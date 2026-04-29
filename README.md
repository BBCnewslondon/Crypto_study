# Crypto Lead-Lag Pipeline & Visualisations

## What Was Built

A complete, modular 5-phase Python pipeline plus a **12-figure visualisation suite** that quantifies predictive latency between "Massive Body" crypto assets and their "Accretion Disk" ecosystems.

---

## Visualisation Gallery

### Summary Dashboard

Four-panel overview of all gravitational systems: peak cross-correlation, Granger F-statistics, lag stability, and results table.

![Summary Dashboard](pipeline_results/figures/summary_dashboard.png)

---

### Normalized Time Series Plots

Standard time series of the assets over the course of the dataset, normalized to their initial starting price.

![Major Cryptos Normalized](pipeline_results/figures/major_cryptos_normalized.png)

![Minor Cryptos Normalized](pipeline_results/figures/minor_cryptos_normalized.png)

---

### Cross-Correlation Heatmap

Mean CCF across all systems vs. lag, with optimal lags marked. The dominant signal at lag 0 (contemporaneous correlation) is visible for all systems, with the much weaker but structurally meaningful signal at positive lags (major leads minor).

![CCF Heatmap](pipeline_results/figures/ccf_heatmap.png)

---

### Per-System CCF Correlograms

Each system's cross-correlation function across 30-day epochs. Blue bars = positive lags (major leads), grey = negative lags. Red bar = optimal lag. Blue band = 95% confidence interval.

![BTC -> STX CCF](pipeline_results/figures/ccf_BTC_STX.png)
![ETH -> ARB CCF](pipeline_results/figures/ccf_ETH_ARB.png)
![SOL -> JUP CCF](pipeline_results/figures/ccf_SOL_JUP.png)
![DOGE -> SHIB CCF](pipeline_results/figures/ccf_DOGE_SHIB.png)

---

### Granger Causality F-Statistics

F-statistics at each lag order (1–10) with p-values annotated. All blue bars are significant at α = 0.05. Note the monotonically decreasing F-stats characteristic of a lag-1 dominant process.

![BTC -> STX Granger](pipeline_results/figures/granger_BTC_STX.png)
![ETH -> ARB Granger](pipeline_results/figures/granger_ETH_ARB.png)
![SOL -> JUP Granger](pipeline_results/figures/granger_SOL_JUP.png)
![DOGE -> SHIB Granger](pipeline_results/figures/granger_DOGE_SHIB.png)

---

### Lag Stability: In-Sample vs. Out-of-Sample

Comparison of the consensus lag τ* from Q1 (training) vs Q2 (testing). ETH→ARB is the only system showing drift (12 min → 1 min).

![Lag Stability](pipeline_results/figures/lag_stability_comparison.png)

---

### Friction Analysis

Mean absolute return at the predicted lag vs. round-trip exchange friction (57 bps). No system's signal exceeds the friction threshold, confirming statistical but not economic significance.

![Friction Analysis](pipeline_results/figures/friction_analysis.png)

---

## Results Summary

| System | τ (IS) | τ (OOS) | Stable? | ρ(τ*) | Granger F | Profitable? |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| **BTC → STX** | 1 min | 1 min | Yes | 0.086 | 101.7 | No |
| **ETH → ARB** | 12 min | 1 min | No | 0.094 | 5.4 | No |
| **SOL → JUP** | 1 min | 1 min | Yes | 0.106 | 325.2 | No |
| **DOGE → SHIB** | 1 min | 1 min | Yes | 0.089 | 412.3 | No |

---

## Robustness Analysis (Data Gaps)

To answer the question of whether the measured correlation and lag are robust enough to survive the introduction of data gaps, a new robustness module was introduced. The module simulates:

1. **Random data dropouts** (e.g. from network failures).
2. **Periodic systematic outages** (e.g. routine exchange maintenance creating data voids).

### Random Outages Overlays

The lag and correlation signal gradually decays as data missingness increases, however the optimal lag $\tau^*$ remains remarkably stable up to 20-30% dropout rates for most pairs.

![BTC -> STX Random Gaps](pipeline_results/figures/robustness_random_BTC_STX.png)
![ETH -> ARB Random Gaps](pipeline_results/figures/robustness_random_ETH_ARB.png)
![SOL -> JUP Random Gaps](pipeline_results/figures/robustness_random_SOL_JUP.png)
![DOGE -> SHIB Random Gaps](pipeline_results/figures/robustness_random_DOGE_SHIB.png)

### Periodic Outages Overlays

When injecting block-outages (e.g., 5 min every 1 hour, or 1 hour every 24 hours), the correlation is also negatively impacted but the optimal lag $\tau^*$ correctly clusters near the true value (1 minute for most pairs, exception for ETH->ARB), affirming that the structural discovery holds true even under non-ideal data telemetry conditions.

![BTC -> STX Periodic Gaps](pipeline_results/figures/robustness_periodic_BTC_STX.png)
![ETH -> ARB Periodic Gaps](pipeline_results/figures/robustness_periodic_ETH_ARB.png)
![SOL -> JUP Periodic Gaps](pipeline_results/figures/robustness_periodic_SOL_JUP.png)
![DOGE -> SHIB Periodic Gaps](pipeline_results/figures/robustness_periodic_DOGE_SHIB.png)

---

## Ecosystem Network Graph Analysis

To treat the market as a complex system rather than isolated pairs, we computed pairwise Granger Causality across the entire defined asset list (BTC, ETH, SOL, ADA, DOGE, STX, ARB, JUP, SHIB). The resulting network graph visualizations map the flow of predictive capital.

Nodes are sized by their *out-degree* (total predictive influence exerted on the rest of the market). Edges represent a statistically significant predictive lead (Granger F-test at Lag 1 with $p < 0.05$), with the edge weight and color scaling to the F-statistic strength.

![Ecosystem Network Q1](pipeline_results/figures/ecosystem_network_Q1.png)

---

## Files

| File | Purpose |
|------|---------|
| [pipeline/robustness.py](pipeline/robustness.py) | Tests CCF calculations against data degradation |
| [pipeline/run_robustness.py](pipeline/run_robustness.py) | Standalone script to execute gap analysis |
| [pipeline/network_analysis.py](pipeline/network_analysis.py) | Computes and plots the ecosystem-wide Granger causality network |
| [pipeline/run_network_analysis.py](pipeline/run_network_analysis.py) | Standalone script to execute the network graph analysis |
| [pipeline/visualisation.py](pipeline/visualisation.py) | Visualisation module (14+ figure types) |
| [pipeline_results/figures/](pipeline_results/figures) | Output directory for all PNG figures |
