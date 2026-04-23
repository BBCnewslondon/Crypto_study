# Methodology & Assumptions

## Section 3: Data Engineering and Analytical Framework

### 3.1  Data Provenance and System Definition

The empirical analysis draws on Kraken Exchange Level-2 (L2) order-book fills and 1-minute OHLCV candlestick data spanning four calendar quarters (Q1–Q4). Each quarter is stored as a collection of headerless CSV files; the L2 records contain `(unix_timestamp, price, volume)` triples, while the candlestick records contain `(unix_timestamp, open, high, low, close, volume, trade_count)` tuples.

We define five *gravitational systems*, each comprising a high-capitalisation "Massive Body" and a structurally dependent "Accretion Disk" asset:

| System | Massive Body | Accretion Disk | Structural Dependency |
|--------|-------------|----------------|----------------------|
| 1 | BTC (XBT) | STX | Stacks settles smart-contracts on Bitcoin L1 |
| 2 | ETH | ARB | Arbitrum is an Ethereum L2 roll-up |
| 3 | SOL | JUP | Jupiter is Solana's dominant DEX aggregator |
| 4 | ADA | SNEK | SNEK is a Cardano-native community token |
| 5 | DOGE (XDG) | SHIB | SHIB emerged as a memetic competitor to DOGE |

> **Note on SNEK:** The Kraken dataset does not include SNEK/USD pairs. System 4 (ADA→SNEK) is therefore excluded from the empirical pipeline but retained in the specification for completeness. If SNEK data becomes available from an alternative venue, the modular architecture permits its immediate integration.

All ticker symbols are mapped to Kraken's internal conventions at the configuration layer (e.g., `BTC → XBTUSD`, `DOGE → XDGUSD`). The quote currency is uniformly USD.

---

### 3.2  Temporal Alignment and Truncation

Raw L2 fills arrive at sub-second resolution. To align them with the candlestick grid and to suppress microstructure noise, we **truncate** each L2 timestamp to the nearest preceding 60-second boundary:

$$
t'_i = \lfloor t_i / 60 \rfloor \times 60
$$

This operation is performed in-memory using Polars' `floordiv` → `mul` chain on the `Int64` timestamp column, avoiding costly datetime parsing. The truncated L2 frame is then **left-joined** to the candlestick frame on $t'$, which attaches the candle's close price ($P_\text{mid}$) to every L2 fill that falls within that minute.

**Justification.** Sixty-second granularity balances two concerns: (i) it is fine enough to detect lags on the order of 1–5 minutes, which prior literature identifies as the dominant regime for crypto lead-lag relationships; (ii) it is coarse enough to contain sufficient L2 observations per bucket for robust volume aggregation.

---

### 3.3  Order-Book Side Classification

The L2 data records executed trades (fills) rather than resting limit orders. To reconstruct a proxy for instantaneous book pressure, we classify each fill as a **resting bid** or **resting ask** using the candle close as a reference:

$$
\text{side}_i =
\begin{cases}
\text{bid}    & \text{if } P_i < P_\text{mid} \\
\text{ask}    & \text{if } P_i > P_\text{mid} \\
\text{at\_mid} & \text{if } P_i = P_\text{mid}
\end{cases}
$$

Fills classified as "at_mid" are retained but excluded from the asymmetric volume aggregation. Per-minute bid and ask volumes are obtained by summation:

$$
V_\text{bid}(t) = \sum_{i:\,\text{side}_i = \text{bid}} v_i, \qquad
V_\text{ask}(t) = \sum_{i:\,\text{side}_i = \text{ask}} v_i
$$

**Assumption.** This heuristic assumes the candle close is a reasonable proxy for the prevailing mid-price at each truncated minute. In practice, the close reflects the *last* trade in that minute, which introduces a small look-ahead bias. We accept this approximation because the primary objective is to measure *multi-minute* lags, where sub-minute timing errors are second-order.

---

### 3.4  Stationarity Transformations

Non-stationary raw price and volume series violate the assumptions of both cross-correlation and Granger causality testing. We apply two transformations.

#### 3.4.1  Log Returns

Raw prices are converted to continuously compounded returns:

$$
r_t = \ln P_t - \ln P_{t-1}
$$

Log returns satisfy three desirable properties:

| Property | Explanation |
|----------|-------------|
| **Variance stabilisation** | Heteroskedasticity in $P_t$ is substantially reduced |
| **Additivity** | Multi-period returns are sums, enabling window-based aggregation |
| **Approximate stationarity** | Log prices are I(1); differencing yields an I(0) series |

We use Polars' vectorised `log()` and `shift(1)` operations; the first observation is `null` and is dropped before downstream analysis.

#### 3.4.2  Order Book Imbalance (OBI)

Absolute bid/ask volumes are non-stationary (they grow with market participation). The **Order Book Imbalance** normalises the instantaneous depth asymmetry into a bounded oscillator:

$$
\text{OBI}(t) = \frac{V_\text{bid}(t) - V_\text{ask}(t)}
                     {V_\text{bid}(t) + V_\text{ask}(t)}
\;\in [-1, +1]
$$

- $\text{OBI} \to +1$ indicates overwhelming buying pressure (all resting volume on the bid side).
- $\text{OBI} \to -1$ indicates overwhelming selling pressure.
- $\text{OBI} = 0$ denotes symmetric depth.

The denominator guard (`V_\text{bid} + V_\text{ask} = 0 \Rightarrow \text{OBI} = 0$) prevents division by zero during illiquid minutes.

---

### 3.5  Signal Processing

#### 3.5.1  Cross-Correlation Function (CCF)

We estimate the sample CCF between the Massive Body's log-return series $\{x_t\}$ and the Accretion Disk's log-return series $\{y_t\}$:

$$
\hat{\rho}(\tau) = \frac{\sum_{t=1}^{N-\tau} (x_t - \bar{x})(y_{t+\tau} - \bar{y})}
                       {\sqrt{\sum_t (x_t - \bar{x})^2 \;\sum_t (y_t - \bar{y})^2}}
$$

where $\tau \in [-60, +60]$ minutes. Positive $\tau$ indicates the major asset **leads** the minor asset by $\tau$ periods.

To detect regime shifts, the CCF is computed in **non-overlapping 30-day epochs**. Within each epoch, we identify the optimal lag:

$$
\tau^* = \arg\max_{\tau > 0} |\hat{\rho}(\tau)|
$$

restricting attention to positive lags (major leads minor), consistent with the gravitational hypothesis.

**Justification.** Thirty-day windows are long enough to contain ≈43,200 one-minute observations (across 24/7 crypto markets), providing statistically stable CCF estimates. The non-overlapping design avoids double-counting.

#### 3.5.2  Granger Causality via Vector Autoregression (VAR)

The CCF measures contemporaneous linear association at a lag. To test for **predictive** causality — whether past values of the major asset's signal improve the forecast of the minor asset's signal — we employ the Granger causality framework.

Consider the bivariate VAR($p$):

$$
\begin{bmatrix} y_t \\ x_t \end{bmatrix}
= \sum_{k=1}^{p} \mathbf{A}_k
\begin{bmatrix} y_{t-k} \\ x_{t-k} \end{bmatrix}
+ \boldsymbol{\epsilon}_t
$$

The null hypothesis "$x$ does not Granger-cause $y$" corresponds to the joint restriction that all coefficients on lagged $x$ in the $y$ equation are zero. We test this using the SSR-based F-statistic from `statsmodels.tsa.stattools.grangercausalitytests` at lag orders $p \in \{1, 2, \dots, 10\}$.

The optimal lag order for the VAR itself is selected by minimising the Akaike Information Criterion (AIC):

$$
\text{AIC}(p) = \ln|\hat{\Sigma}_p| + \frac{2 p d^2}{T}
$$

where $d=2$ (bivariate system), $T$ is the sample length, and $\hat{\Sigma}_p$ is the residual covariance.

**Significance threshold.** We adopt $\alpha = 0.05$ throughout. Given the large sample sizes (≈43k observations per epoch), Bonferroni correction across lag orders would be overly conservative; instead, we report all significant lags and focus on the AIC-optimal model.

---

### 3.6  Out-of-Sample Validation

#### 3.6.1  Strategy

To guard against over-fitting, the analysis employs a **temporal train/test split**:

| Phase | Quarter | Purpose |
|-------|---------|---------|
| Training | Q1 | CCF epoch analysis, Granger testing, lag identification |
| Testing | Q2 | Lag stability check and friction accounting |

The consensus lag $\hat{\tau}$ from the training period is defined as the **mode** of the per-epoch optimal lags $\{\tau^*_1, \dots, \tau^*_K\}$, providing robustness against outlier epochs.

#### 3.6.2  Lag Stability Criterion

On the test quarter, we repeat the rolling CCF procedure and extract the consensus lag $\hat{\tau}_\text{oos}$. The lag is deemed *stable* if:

$$
|\hat{\tau}_\text{train} - \hat{\tau}_\text{oos}| \leq 3 \;\text{minutes}
$$

The ±3-minute tolerance accounts for natural non-stationarity in market microstructure (e.g., liquidity shifts between trading sessions).

#### 3.6.3  Friction Accounting

Even a statistically significant, stable lag yields no economic edge if the predicted price movement falls below the **round-trip mechanical friction** of the exchange. We define:

$$
\text{Friction}_\text{RT} = 2 \times f_\text{taker} + s_\text{spread}
$$

where $f_\text{taker} = 26\;\text{bps}$ (Kraken's standard taker fee) and $s_\text{spread}$ is estimated at 5 bps for major USD pairs.

For each test-quarter epoch, the **mean absolute log-return** at lag $\hat{\tau}$ is compared against the total friction:

$$
\text{Net Edge} = \bigl(\overline{|r_{t+\hat{\tau}}|}\bigr) \times 10{,}000 - \text{Friction}_\text{RT}
$$

An epoch is classified as economically viable if $\text{Net Edge} > 0$. The overall system summary reports the **fraction of profitable epochs** as a measure of strategy persistence.

---

### 3.7  Implementation Notes

| Decision | Rationale |
|----------|-----------|
| **Polars** over pandas | Lazy evaluation and Rust-backed columnar engine provide 2–10× speedup on multi-GB CSV ingestion without exceeding RAM |
| **Statsmodels VAR** | Mature, well-tested VAR and Granger implementations with SSR/Wald/LR test variants |
| **No detrending beyond Δln** | Crypto prices are typically I(1); a single log-difference suffices. Augmented Dickey–Fuller tests can be added as a robustness check |
| **Fixed spread estimate** | True L1 best-bid/best-ask is not available in L2 fill data. A 5 bps estimate is conservative for BTC/USD on Kraken |
| **No slippage model** | Full slippage modelling requires order-book depth at the moment of execution, which is unavailable. The friction estimate is therefore a *lower bound* on true costs |

---

### 3.8  Limitations and Future Work

1. **L2 data reflects fills, not quotes.** The bid/ask classification is a heuristic proxy. Future work should incorporate live L2 snapshots for precise depth reconstruction.

2. **Linear methods only.** CCF and Granger causality assume linear relationships. Non-linear lead-lag dynamics (e.g., during liquidation cascades) may be captured by Transfer Entropy or Wavelet Coherence in subsequent phases.

3. **Rolling windows are non-adaptive.** A fixed 30-day window may obscure regime changes that occur mid-epoch. An adaptive window (e.g., structural-break detection via CUSUM/Bai-Perron) would improve resolution.

4. **ADA→SNEK system excluded.** The Cardano-native SNEK token is not listed on Kraken. Alternative data sources (SundaeSwap, Minswap DEX) could enable this analysis on-chain.
