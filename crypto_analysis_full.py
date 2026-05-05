from __future__ import annotations

from dataclasses import dataclass
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from typing import Dict, Any
from typing import Dict, List, Tuple
from typing import Dict, Tuple, List
from typing import List, Dict, Any, Tuple
from typing import List, Optional
from typing import List, Optional, Tuple
from typing import Optional
import argparse
import json
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import polars as pl
import sys
import time


# ======================================================================
# --- FILE: config.py ---
# ======================================================================


"""
Phase 1: System Definition — Asset Selection & Configuration
=============================================================
Defines the "Massive Body" → "Accretion Disk" pairings and all
global constants used throughout the pipeline.

Kraken uses 'XBT' for Bitcoin and 'XDG' for Dogecoin in their
pair naming. We map human-readable names to Kraken tickers.
"""



# ---------------------------------------------------------------------------
# Ticker mapping: human-readable → Kraken L2/candle file prefix
# ---------------------------------------------------------------------------
TICKER_MAP: Dict[str, str] = {
    "BTC": "XBT",
    "ETH": "ETH",
    "SOL": "SOL",
    "ADA": "ADA",
    "DOGE": "XDG",
    "STX": "STX",
    "ARB": "ARB",
    "JUP": "JUP",
    "SNEK": "SNEK",
    "SHIB": "SHIB",
}

# Quote currency used across all pairs
QUOTE_CURRENCY: str = "USD"


# ---------------------------------------------------------------------------
# Massive Body → Accretion Disk mapping
# ---------------------------------------------------------------------------
SYSTEM_PAIRINGS: Dict[str, str] = {
    # BTC  → STX   (Stacks runs smart-contracts *on* Bitcoin; direct dependency)
    "BTC": "STX",
    # ETH  → ARB   (Arbitrum is an Ethereum L2 rollup; inherits ETH settlement)
    "ETH": "ARB",
    # SOL  → JUP   (Jupiter is the dominant DEX aggregator on Solana)
    "SOL": "JUP",
    # ADA  → SNEK  (SNEK is the leading Cardano-native meme/community token)
    "ADA": "SNEK",
    # DOGE → SHIB  (SHIB emerged as the "Dogecoin killer" meme competitor)
    "DOGE": "SHIB",
}


def get_kraken_pair(asset: str) -> str:
    """Return the Kraken file-prefix for a given human-readable asset name.

    Example
    -------
    >>> get_kraken_pair("BTC")
    'XBTUSD'
    """
    return f"{TICKER_MAP[asset]}{QUOTE_CURRENCY}"


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataPaths:
    """Centralised path registry so no module hard-codes paths."""

    root: Path = Path(r"c:\Users\singh\Crypto_study")

    # Order-book (L2) directories — one per quarter
    l2_dirs: Tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")

    # Candlestick directories — one per quarter
    candle_dirs: Tuple[str, ...] = (
        "Q1_Candlestick",
        "Q2_candlestick",
        "Q3_candlestick",
        "Q4_candlestick",
    )

    def l2_path(self, quarter: str, pair: str) -> Path:
        """Return path to an L2 CSV: e.g. Q1/XBTUSD.csv"""
        return self.root / quarter / f"{pair}.csv"

    def candle_path(self, quarter: str, pair: str, interval: int = 1) -> Path:
        """Return path to a candle CSV: e.g. Q1_Candlestick/XBTUSD_1.csv"""
        return self.root / quarter / f"{pair}_{interval}.csv"

    @property
    def quarter_pairs(self) -> List[Tuple[str, str]]:
        """Yield (l2_dir, candle_dir) tuples aligned by quarter."""
        return list(zip(self.l2_dirs, self.candle_dirs))


# ---------------------------------------------------------------------------
# Pipeline hyper-parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineConfig:
    """All tuneable knobs in one place."""

    # Phase 2 – timestamp truncation granularity (seconds)
    truncation_seconds: int = 60

    # Phase 4 – CCF rolling-window size (calendar days)
    ccf_window_days: int = 30

    # Phase 4 – maximum CCF lag to search (minutes)
    ccf_max_lag: int = 60

    # Phase 4 – VAR model maximum lag order for Granger test
    granger_max_lag: int = 10

    # Phase 4 – Granger significance level
    granger_alpha: float = 0.05

    # Phase 5 – Kraken taker fee (bps → decimal)
    kraken_taker_fee: float = 0.0026  # 26 bps

    # Phase 5 – Kraken maker fee (bps → decimal)
    kraken_maker_fee: float = 0.0016  # 16 bps

    # Phase 5 – train quarter, test quarter
    train_quarter: str = "Q1"
    test_quarter: str = "Q2"


# Singleton instances for convenience
PATHS = DataPaths()
CONFIG = PipelineConfig()


# ======================================================================
# --- FILE: data_engineering.py ---
# ======================================================================


"""
Phase 2: Data Engineering — Ingestion, Truncation & Book Splitting
===================================================================
All heavy lifting uses **polars** for memory-efficient, lazy evaluation.

Workflow per (quarter, pair):
  1. Scan the L2 csv  → truncate timestamps to 60 s
  2. Scan the 1-min candle csv → extract close as P_mid
  3. Left-join candle close onto the order book by truncated timestamp
  4. Classify each limit order as a resting bid (P_i < P_mid)
     or resting ask (P_i > P_mid)
"""






# ---------------------------------------------------------------------------
# Column names (raw CSVs have no headers)
# ---------------------------------------------------------------------------
L2_COLUMNS: list[str] = ["timestamp", "price", "volume"]
CANDLE_COLUMNS: list[str] = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "count",
]


class DataEngineer:
    """Ingests and processes L2 + candlestick data for a single pair/quarter.

    Parameters
    ----------
    pair : str
        Kraken pair name, e.g. ``"XBTUSD"``.
    quarter_l2 : str
        L2 directory name, e.g. ``"Q1"``.
    quarter_candle : str
        Candle directory name, e.g. ``"Q1_Candlestick"``.
    paths : DataPaths
        Filesystem layout.
    config : PipelineConfig
        Pipeline hyper-parameters.
    """

    def __init__(
        self,
        pair: str,
        quarter_l2: str,
        quarter_candle: str,
        paths: DataPaths = PATHS,
        config: PipelineConfig = CONFIG,
    ) -> None:
        self.pair = pair
        self.quarter_l2 = quarter_l2
        self.quarter_candle = quarter_candle
        self.paths = paths
        self.config = config

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_l2(self) -> pl.LazyFrame:
        """Scan the L2 CSV and truncate timestamps to ``truncation_seconds``."""
        path = self.paths.l2_path(self.quarter_l2, self.pair)
        if not path.exists():
            raise FileNotFoundError(f"L2 file not found: {path}")

        return (
            pl.scan_csv(
                path,
                has_header=False,
                new_columns=L2_COLUMNS,
                dtypes={
                    "timestamp": pl.Int64,
                    "price": pl.Float64,
                    "volume": pl.Float64,
                },
            )
            # Truncate to nearest 60-second boundary
            .with_columns(
                (
                    pl.col("timestamp")
                    .floordiv(self.config.truncation_seconds)
                    .mul(self.config.truncation_seconds)
                ).alias("ts_truncated")
            )
        )

    def _load_candles(self) -> pl.LazyFrame:
        """Scan the 1-min candle CSV and keep only (timestamp, close)."""
        path = self.paths.candle_path(
            self.quarter_candle, self.pair, interval=1
        )
        if not path.exists():
            raise FileNotFoundError(f"Candle file not found: {path}")

        return pl.scan_csv(
            path,
            has_header=False,
            new_columns=CANDLE_COLUMNS,
            dtypes={
                "timestamp": pl.Int64,
                "open": pl.Float64,
                "high": pl.Float64,
                "low": pl.Float64,
                "close": pl.Float64,
                "volume": pl.Float64,
                "count": pl.Int64,
            },
        ).select(
            pl.col("timestamp").alias("ts_truncated"),
            pl.col("close").alias("p_mid"),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_enriched_book(self) -> pl.DataFrame:
        """Join L2 orders with candlestick mid-prices and classify side.

        Returns
        -------
        pl.DataFrame
            Columns: ``ts_truncated, price, volume, p_mid, side``
            where ``side ∈ {"bid", "ask", "at_mid"}``.
        """
        l2 = self._load_l2()
        candles = self._load_candles()

        enriched = (
            l2.join(candles, on="ts_truncated", how="left")
            .filter(pl.col("p_mid").is_not_null())
            .with_columns(
                pl.when(pl.col("price") < pl.col("p_mid"))
                .then(pl.lit("bid"))
                .when(pl.col("price") > pl.col("p_mid"))
                .then(pl.lit("ask"))
                .otherwise(pl.lit("at_mid"))
                .alias("side")
            )
            .collect()
        )
        return enriched

    def build_minute_bars(self) -> pl.DataFrame:
        """Return the 1-minute candlestick data as a collected DataFrame.

        Columns: ``ts_truncated, p_mid``
        """
        return self._load_candles().collect()

    def build_book_depth(self) -> pl.DataFrame:
        """Aggregate resting bid/ask volumes per minute.

        Returns
        -------
        pl.DataFrame
            Columns: ``ts_truncated, v_bid, v_ask``
        """
        enriched = self.build_enriched_book()

        bids = (
            enriched.filter(pl.col("side") == "bid")
            .group_by("ts_truncated")
            .agg(pl.col("volume").sum().alias("v_bid"))
        )

        asks = (
            enriched.filter(pl.col("side") == "ask")
            .group_by("ts_truncated")
            .agg(pl.col("volume").sum().alias("v_ask"))
        )

        depth = (
            bids.join(asks, on="ts_truncated", how="outer_coalesce")
            .with_columns(
                pl.col("v_bid").fill_null(0.0),
                pl.col("v_ask").fill_null(0.0),
            )
            .sort("ts_truncated")
        )
        return depth


# ======================================================================
# --- FILE: stationarity.py ---
# ======================================================================


"""
Phase 3: Mathematical Transformation — Achieving Stationarity
==============================================================
Raw prices and absolute order-book volumes are non-stationary.
We convert them to:

  * **Log returns** (price → velocity):
        r_t = ln(P_t) − ln(P_{t−1})

  * **Order Book Imbalance (OBI)** (depth → normalised oscillator):
        OBI = (V_bid − V_ask) / (V_bid + V_ask)   ∈ [−1, 1]
"""



def compute_log_returns(prices: pl.DataFrame, col: str = "p_mid") -> pl.DataFrame:
    """Compute continuous log-returns for a minute-bar DataFrame.

    Parameters
    ----------
    prices : pl.DataFrame
        Must contain ``ts_truncated`` and *col*.
    col : str
        Column name holding the price series.

    Returns
    -------
    pl.DataFrame
        Original columns plus ``log_return``.  First row is null.
    """
    return prices.sort("ts_truncated").with_columns(
        (pl.col(col).log() - pl.col(col).shift(1).log()).alias("log_return")
    )


def compute_obi(depth: pl.DataFrame) -> pl.DataFrame:
    """Compute normalised Order Book Imbalance from bid/ask volumes.

    Parameters
    ----------
    depth : pl.DataFrame
        Must contain ``ts_truncated``, ``v_bid``, ``v_ask``.

    Returns
    -------
    pl.DataFrame
        Original columns plus ``obi``.
        Rows where ``V_bid + V_ask = 0`` are set to ``0.0``.
    """
    return depth.with_columns(
        pl.when((pl.col("v_bid") + pl.col("v_ask")) > 0)
        .then(
            (pl.col("v_bid") - pl.col("v_ask"))
            / (pl.col("v_bid") + pl.col("v_ask"))
        )
        .otherwise(0.0)
        .alias("obi")
    )


def build_stationary_signals(
    prices: pl.DataFrame,
    depth: pl.DataFrame,
) -> pl.DataFrame:
    """Merge log-returns and OBI into a single aligned DataFrame.

    Parameters
    ----------
    prices : pl.DataFrame
        From ``DataEngineer.build_minute_bars()``.
    depth : pl.DataFrame
        From ``DataEngineer.build_book_depth()``.

    Returns
    -------
    pl.DataFrame
        Columns: ``ts_truncated, p_mid, log_return, v_bid, v_ask, obi``
        Sorted by ``ts_truncated``, null-free.
    """
    lr = compute_log_returns(prices)
    obi_df = compute_obi(depth)

    merged = (
        lr.join(obi_df, on="ts_truncated", how="inner")
        .sort("ts_truncated")
        .drop_nulls(subset=["log_return", "obi"])
    )
    return merged


# ======================================================================
# --- FILE: signal_processing.py ---
# ======================================================================


"""
Phase 4: Signal Processing — Isolating the Lag
================================================
Two complementary techniques:

1. **Cross-Correlation Function (CCF)** over 30-day rolling windows
   (epochs) to locate the lag τ where correlation peaks.

2. **Granger Causality** via a Vector Autoregression (VAR) model to
   test whether lagged values of the major asset's signal improve
   prediction of the minor asset's signal.
"""


from numpy.typing import NDArray
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR


# ---------------------------------------------------------------------------
# Data structures for results
# ---------------------------------------------------------------------------
@dataclass
class CCFResult:
    """Result of a single-epoch cross-correlation analysis."""

    epoch_start: int  # Unix timestamp
    epoch_end: int
    n_obs: int        # number of observations
    optimal_lag: int  # minutes
    peak_correlation: float
    lags: NDArray[np.int64]
    correlations: NDArray[np.float64]


@dataclass
class GrangerResult:
    """Result of a Granger causality test at a specific lag order."""

    lag_order: int
    f_statistic: float
    p_value: float
    is_significant: bool


# ---------------------------------------------------------------------------
# Cross-Correlation Function
# ---------------------------------------------------------------------------
def cross_correlate(
    x: NDArray[np.float64],
    y: NDArray[np.float64],
    max_lag: int,
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Compute normalised cross-correlation of *x* (major) → *y* (minor).

    Positive lag means x *leads* y by that many periods.

    Parameters
    ----------
    x, y : 1-D arrays of equal length
    max_lag : maximum lag to evaluate (both directions)

    Returns
    -------
    lags : array of lag indices [-max_lag … +max_lag]
    ccf  : normalised correlation at each lag
    """
    n = len(x)
    x_demean = x - x.mean()
    y_demean = y - y.mean()
    norm = np.sqrt(np.sum(x_demean**2) * np.sum(y_demean**2))
    if norm == 0:
        lags = np.arange(-max_lag, max_lag + 1)
        return lags, np.zeros_like(lags, dtype=np.float64)

    lags_list: list[int] = []
    ccf_list: list[float] = []

    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = np.sum(x_demean[: n - lag] * y_demean[lag:])
        else:
            c = np.sum(x_demean[-lag:] * y_demean[: n + lag])
        lags_list.append(lag)
        ccf_list.append(c / norm)

    return np.array(lags_list), np.array(ccf_list)


def rolling_ccf(
    major_signal: pl.DataFrame,
    minor_signal: pl.DataFrame,
    signal_col: str = "log_return",
    window_days: int = 30,
    max_lag: int = 60,
) -> List[CCFResult]:
    """Compute CCF in non-overlapping 30-day epochs.

    Parameters
    ----------
    major_signal, minor_signal : DataFrames with ``ts_truncated`` + *signal_col*.
    signal_col : column to correlate.
    window_days : epoch length in calendar days.
    max_lag : maximum lag to search in minutes.

    Returns
    -------
    List of ``CCFResult`` for each epoch.
    """
    # Align on common timestamps
    aligned = major_signal.select(
        "ts_truncated",
        pl.col(signal_col).alias("major"),
    ).join(
        minor_signal.select(
            "ts_truncated",
            pl.col(signal_col).alias("minor"),
        ),
        on="ts_truncated",
        how="inner",
    ).sort("ts_truncated").drop_nulls()

    ts = aligned["ts_truncated"].to_numpy()
    major_arr = aligned["major"].to_numpy()
    minor_arr = aligned["minor"].to_numpy()

    window_secs = window_days * 86_400
    results: List[CCFResult] = []
    start = ts[0]
    end_max = ts[-1]

    while start + window_secs <= end_max:
        epoch_end = start + window_secs
        mask = (ts >= start) & (ts < epoch_end)
        x = major_arr[mask]
        y = minor_arr[mask]

        if len(x) > 2 * max_lag:
            lags, ccf = cross_correlate(x, y, max_lag)
            # Focus on positive lags (major leads minor)
            pos_mask = lags > 0
            if np.any(pos_mask):
                best_idx = np.argmax(np.abs(ccf[pos_mask]))
                opt_lag = lags[pos_mask][best_idx]
                peak_corr = ccf[pos_mask][best_idx]
            else:
                opt_lag = 0
                peak_corr = 0.0

            results.append(
                CCFResult(
                    epoch_start=int(start),
                    epoch_end=int(epoch_end),
                    n_obs=len(x),
                    optimal_lag=int(opt_lag),
                    peak_correlation=float(peak_corr),
                    lags=lags,
                    correlations=ccf,
                )
            )

        start = epoch_end  # non-overlapping advance

    return results


# ---------------------------------------------------------------------------
# Granger Causality
# ---------------------------------------------------------------------------
def granger_causality_test(
    major_signal: pl.DataFrame,
    minor_signal: pl.DataFrame,
    signal_col: str = "log_return",
    max_lag: int = 10,
    alpha: float = 0.05,
) -> List[GrangerResult]:
    """Granger-causality: does the major asset Granger-cause the minor?

    Uses ``statsmodels.tsa.stattools.grangercausalitytests``.

    Parameters
    ----------
    major_signal, minor_signal : DataFrames with ``ts_truncated`` + *signal_col*.
    max_lag : maximum lag order to test.
    alpha : significance threshold.

    Returns
    -------
    List of ``GrangerResult`` for lags 1..max_lag.
    """
    aligned = major_signal.select(
        "ts_truncated",
        pl.col(signal_col).alias("major"),
    ).join(
        minor_signal.select(
            "ts_truncated",
            pl.col(signal_col).alias("minor"),
        ),
        on="ts_truncated",
        how="inner",
    ).sort("ts_truncated").drop_nulls()

    # grangercausalitytests expects [y, x] — test if x Granger-causes y
    data = np.column_stack([
        aligned["minor"].to_numpy(),
        aligned["major"].to_numpy(),
    ])

    results: List[GrangerResult] = []
    try:
        test_output = grangercausalitytests(data, maxlag=max_lag, verbose=False)
    except Exception as e:
        print(f"[Granger] Test failed: {e}")
        return results

    for lag in range(1, max_lag + 1):
        ssr_f = test_output[lag][0]["ssr_ftest"]
        f_stat = float(ssr_f[0])
        p_val = float(ssr_f[1])
        results.append(
            GrangerResult(
                lag_order=lag,
                f_statistic=f_stat,
                p_value=p_val,
                is_significant=(p_val < alpha),
            )
        )

    return results


def var_model_analysis(
    major_signal: pl.DataFrame,
    minor_signal: pl.DataFrame,
    signal_col: str = "log_return",
    max_lag: int = 10,
) -> Optional[dict]:
    """Fit a VAR model and extract coefficient diagnostics.

    Returns a dict with ``aic``, ``bic``, ``optimal_order``, and the
    coefficient matrix for the optimal lag.
    """
    aligned = major_signal.select(
        "ts_truncated",
        pl.col(signal_col).alias("major"),
    ).join(
        minor_signal.select(
            "ts_truncated",
            pl.col(signal_col).alias("minor"),
        ),
        on="ts_truncated",
        how="inner",
    ).sort("ts_truncated").drop_nulls()

    data = np.column_stack([
        aligned["minor"].to_numpy(),
        aligned["major"].to_numpy(),
    ])

    try:
        model = VAR(data)
        results = model.select_order(maxlags=max_lag)
        optimal_order = results.aic
        fitted = model.fit(optimal_order)

        return {
            "optimal_order": optimal_order,
            "aic": fitted.aic,
            "bic": fitted.bic,
            "coefficients": fitted.params,
            "summary": str(fitted.summary()),
        }
    except Exception as e:
        print(f"[VAR] Model fitting failed: {e}")
        return None


# ======================================================================
# --- FILE: validation.py ---
# ======================================================================


"""
Phase 5: Reality Testing — Out-of-Sample Validation & Friction
================================================================
Validates that the lag τ identified in-sample (train quarter) persists
in unseen data (test quarter) and that the predicted price movement
exceeds exchange mechanical friction (Kraken fees + bid-ask spread).
"""


from numpy.typing import NDArray




# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class FrictionCheck:
    """Result of the friction accounting check for one epoch."""

    epoch_start: int
    epoch_end: int
    predicted_move_bps: float       # absolute predicted move in basis points
    total_friction_bps: float       # taker fee + spread in basis points
    net_edge_bps: float             # predicted − friction
    is_profitable: bool             # edge > 0?


@dataclass
class ValidationReport:
    """Complete out-of-sample validation for one major→minor system."""

    major_asset: str
    minor_asset: str
    train_quarter: str
    test_quarter: str
    in_sample_lag: int              # τ from train period (minutes)
    in_sample_peak_corr: float
    oos_lag: int                    # τ on test period
    oos_peak_corr: float
    lag_stable: bool                # in-sample τ ≈ out-of-sample τ?
    friction_checks: List[FrictionCheck]
    fraction_profitable: float


# ---------------------------------------------------------------------------
# Epoch chunking
# ---------------------------------------------------------------------------
def identify_consensus_lag(
    ccf_results: List[CCFResult],
) -> int:
    """Pick the lag that appears most often across epochs (mode)."""
    if not ccf_results:
        return 0
    lags = [r.optimal_lag for r in ccf_results]
    values, counts = np.unique(lags, return_counts=True)
    return int(values[np.argmax(counts)])


# ---------------------------------------------------------------------------
# Friction accounting
# ---------------------------------------------------------------------------
def compute_bid_ask_spread_bps(
    depth: pl.DataFrame,
) -> float:
    """Estimate median bid-ask spread in basis points from book depth.

    A rough proxy: we use the ratio v_bid / v_ask as a pressure indicator.
    For a true spread we would need best-bid/best-ask prices; here we
    approximate from the candle close +/- a fraction derived from
    book imbalance.

    Since the L2 data contains fills rather than live quotes, we use a
    conservative fixed estimate of 5 bps (typical for majors on Kraken).
    """
    # Conservative fixed estimate for Kraken tight markets
    return 5.0  # basis points


def run_friction_checks(
    minor_signal: pl.DataFrame,
    predicted_lag: int,
    taker_fee: float,
    depth: Optional[pl.DataFrame] = None,
    window_days: int = 30,
) -> List[FrictionCheck]:
    """Check whether predicted moves at lag τ exceed friction.

    Parameters
    ----------
    minor_signal : DataFrame with ``ts_truncated``, ``log_return``.
    predicted_lag : τ in minutes.
    taker_fee : decimal (e.g. 0.0026 for 26 bps).
    depth : optional book-depth for spread estimation.
    window_days : epoch length.

    Returns
    -------
    List of ``FrictionCheck`` per epoch.
    """
    spread_bps = compute_bid_ask_spread_bps(depth) if depth is not None else 5.0
    total_friction_bps = taker_fee * 10_000 * 2 + spread_bps  # round-trip

    ts = minor_signal["ts_truncated"].to_numpy()
    lr = minor_signal["log_return"].to_numpy()

    window_secs = window_days * 86_400
    results: List[FrictionCheck] = []
    start = ts[0]
    end_max = ts[-1]

    while start + window_secs <= end_max:
        epoch_end = start + window_secs
        mask = (ts >= start) & (ts < epoch_end)
        epoch_lr = lr[mask]

        if len(epoch_lr) > predicted_lag:
            # Mean absolute return at the predicted lag
            shifted = np.abs(epoch_lr[predicted_lag:])
            predicted_move_bps = float(np.mean(shifted)) * 10_000

            net_edge = predicted_move_bps - total_friction_bps
            results.append(
                FrictionCheck(
                    epoch_start=int(start),
                    epoch_end=int(epoch_end),
                    predicted_move_bps=round(predicted_move_bps, 4),
                    total_friction_bps=round(total_friction_bps, 4),
                    net_edge_bps=round(net_edge, 4),
                    is_profitable=(net_edge > 0),
                )
            )
        start = epoch_end

    return results


# ---------------------------------------------------------------------------
# Full OOS validation
# ---------------------------------------------------------------------------
def validate_out_of_sample(
    major_asset: str,
    minor_asset: str,
    train_major: pl.DataFrame,
    train_minor: pl.DataFrame,
    test_major: pl.DataFrame,
    test_minor: pl.DataFrame,
    test_minor_depth: Optional[pl.DataFrame],
    train_quarter: str,
    test_quarter: str,
    max_lag: int = 60,
    window_days: int = 30,
    taker_fee: float = 0.0026,
) -> ValidationReport:
    """End-to-end out-of-sample validation.

    1. Compute CCF on train data → consensus τ
    2. Compute CCF on test data → check τ stability
    3. Run friction checks on test data
    """
    

    # --- In-sample ---
    train_ccf = rolling_ccf(
        train_major, train_minor,
        max_lag=max_lag, window_days=window_days,
    )
    in_sample_lag = identify_consensus_lag(train_ccf)
    in_sample_corr = float(
        np.mean([r.peak_correlation for r in train_ccf])
    ) if train_ccf else 0.0

    # --- Out-of-sample ---
    test_ccf = rolling_ccf(
        test_major, test_minor,
        max_lag=max_lag, window_days=window_days,
    )
    oos_lag = identify_consensus_lag(test_ccf)
    oos_corr = float(
        np.mean([r.peak_correlation for r in test_ccf])
    ) if test_ccf else 0.0

    lag_stable = abs(in_sample_lag - oos_lag) <= 3  # ±3 min tolerance

    # --- Friction ---
    friction_checks = run_friction_checks(
        test_minor,
        predicted_lag=in_sample_lag,
        taker_fee=taker_fee,
        depth=test_minor_depth,
        window_days=window_days,
    )
    profitable_epochs = sum(1 for fc in friction_checks if fc.is_profitable)
    frac = profitable_epochs / len(friction_checks) if friction_checks else 0.0

    return ValidationReport(
        major_asset=major_asset,
        minor_asset=minor_asset,
        train_quarter=train_quarter,
        test_quarter=test_quarter,
        in_sample_lag=in_sample_lag,
        in_sample_peak_corr=round(in_sample_corr, 6),
        oos_lag=oos_lag,
        oos_peak_corr=round(oos_corr, 6),
        lag_stable=lag_stable,
        friction_checks=friction_checks,
        fraction_profitable=round(frac, 4),
    )


# ======================================================================
# --- FILE: visualisation.py ---
# ======================================================================


"""
Scientific Visualisation Module
================================
Generates publication-quality figures from pipeline results.
Uses matplotlib with a tuned academic style.
"""


matplotlib.use("Agg")  # Non-interactive backend for headless rendering
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch

# ---------------------------------------------------------------------------
# Global style configuration — publication-ready
# ---------------------------------------------------------------------------
STYLE_PARAMS = {
    # --- Font ---
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,

    # --- Figure ---
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,

    # --- Axes ---
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#1a1a1a",
    "axes.grid": True,
    "axes.grid.which": "major",
    "axes.spines.top": False,
    "axes.spines.right": False,

    # --- Grid ---
    "grid.color": "#e0e0e0",
    "grid.linewidth": 0.5,
    "grid.alpha": 0.7,

    # --- Ticks ---
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 4,
    "ytick.major.size": 4,

    # --- Lines ---
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
}

# Colour palette — muted, accessible, scientific
PALETTE = {
    "primary":      "#2c6fbb",   # Steel blue
    "secondary":    "#cc4c02",   # Burnt orange
    "tertiary":     "#238b45",   # Forest green
    "quaternary":   "#6a3d9a",   # Purple
    "accent":       "#e31a1c",   # Red (for highlights)
    "neutral":      "#636363",   # Grey
    "light_fill":   "#d0e1f9",   # Light blue fill
    "neg_fill":     "#fdd0a2",   # Light orange fill
    "bg":           "#fafafa",   # Near-white background
    "grid":         "#e0e0e0",
}

SYSTEM_COLOURS = [
    PALETTE["primary"],
    PALETTE["secondary"],
    PALETTE["tertiary"],
    PALETTE["quaternary"],
]


def _apply_style() -> None:
    """Apply the publication style globally."""
    plt.rcParams.update(STYLE_PARAMS)


# ---------------------------------------------------------------------------
# 1.  CCF Correlogram (one per system)
# ---------------------------------------------------------------------------
def plot_ccf_correlogram(
    system_label: str,
    ccf_data: List[Dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Plot the cross-correlation function for each epoch of a system.

    Parameters
    ----------
    system_label : e.g. "BTC -> STX"
    ccf_data : list of dicts with 'lags', 'correlations', 'optimal_lag', 'peak_correlation'
    output_dir : directory to save the figure

    Returns
    -------
    Path to the saved figure.
    """
    _apply_style()
    n_epochs = len(ccf_data)
    cols = min(n_epochs, 2)
    rows = (n_epochs + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols, figsize=(5.5 * cols, 4 * rows),
        sharey=True, squeeze=False,
    )
    axes_flat = axes.flatten()

    for i, epoch in enumerate(ccf_data):
        ax = axes_flat[i]
        lags = np.array(epoch["lags"])
        corrs = np.array(epoch["correlations"])
        opt_lag = epoch["optimal_lag"]
        peak = epoch["peak_correlation"]

        # Stem-like bar plot
        colours = np.where(lags > 0, PALETTE["primary"], PALETTE["neutral"])
        # Highlight the optimal lag bar
        opt_idx = np.where(lags == opt_lag)[0]
        if len(opt_idx) > 0:
            colours[opt_idx[0]] = PALETTE["accent"]

        ax.bar(lags, corrs, width=0.85, color=colours, alpha=0.75, edgecolor="none")

        # Zero line
        ax.axhline(0, color="#333333", linewidth=0.6, zorder=1)
        ax.axvline(0, color="#999999", linewidth=0.5, linestyle="--", alpha=0.5)

        # 95% confidence band (approx 2/sqrt(N))
        n_obs = epoch.get("n_obs", 43200)  # exact from signal_processing if available, else approx for 30-day epoch
        ci = 1.96 / np.sqrt(max(n_obs, 1))
        ax.axhspan(-ci, ci, color=PALETTE["light_fill"], alpha=0.35, zorder=0,
                    label="95% CI")

        # Annotate optimal lag
        ax.annotate(
            f"$\\tau^*$ = {opt_lag} min\n$\\rho$ = {peak:.4f}",
            xy=(opt_lag, peak),
            xytext=(opt_lag + 8, peak + 0.04 if peak > 0 else peak - 0.06),
            fontsize=8.5,
            color=PALETTE["accent"],
            fontweight="bold",
            arrowprops=dict(
                arrowstyle="->",
                color=PALETTE["accent"],
                lw=1.0,
                connectionstyle="arc3,rad=0.2",
            ),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=PALETTE["accent"],
                      alpha=0.9, lw=0.7),
        )

        ax.set_xlabel("Lag $\\tau$ (minutes)")
        if i % cols == 0:
            ax.set_ylabel("Cross-Correlation $\\hat{\\rho}(\\tau)$")
        ax.set_title(f"Epoch {i + 1}", fontweight="medium")
        ax.set_xlim(-65, 65)

    for i in range(n_epochs, len(axes_flat)):
        fig.delaxes(axes_flat[i])

    fig.suptitle(
        f"Cross-Correlation Function:  {system_label}",
        fontsize=13, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    out = output_dir / f"ccf_{system_label.replace(' -> ', '_').replace(' ', '')}.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 2.  Granger Causality F-statistic plot
# ---------------------------------------------------------------------------
def plot_granger_fstats(
    system_label: str,
    granger_data: List[Dict[str, Any]],
    alpha: float,
    output_dir: Path,
) -> Path:
    """Bar chart of Granger F-statistics at each lag order."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    lags = [g["lag_order"] for g in granger_data]
    f_stats = [g["f_statistic"] for g in granger_data]
    p_vals = [g["p_value"] for g in granger_data]
    significant = [g["is_significant"] for g in granger_data]

    bar_colours = [
        PALETTE["primary"] if sig else PALETTE["neutral"]
        for sig in significant
    ]

    bars = ax.bar(lags, f_stats, color=bar_colours, alpha=0.8, edgecolor="white",
                  linewidth=0.5, width=0.7)

    # Annotate p-values on each bar
    for lag_i, bar, p in zip(lags, bars, p_vals):
        if p < 1e-10:
            p_text = f"p < 10$^{{{int(np.floor(np.log10(p)))}}}$"
        elif p < 0.001:
            p_text = f"p = {p:.1e}"
        else:
            p_text = f"p = {p:.3f}"

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(f_stats) * 0.02,
            p_text,
            ha="center", va="bottom",
            fontsize=6.5, color="#444444", rotation=45,
        )

    # Significance marker
    ax.axhline(0, color="#333333", linewidth=0.5)

    ax.set_xlabel("Lag Order $p$")
    ax.set_ylabel("F-statistic")
    ax.set_title(
        f"Granger Causality Test:  {system_label}",
        fontsize=12, fontweight="bold",
    )
    ax.set_xticks(lags)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=PALETTE["primary"], alpha=0.8, label=f"Significant ($\\alpha$ = {alpha})"),
        Patch(facecolor=PALETTE["neutral"], alpha=0.8, label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    out = output_dir / f"granger_{system_label.replace(' -> ', '_').replace(' ', '')}.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 3.  Lag Stability Comparison (all systems)
# ---------------------------------------------------------------------------
def plot_lag_stability(
    systems: List[str],
    in_sample_lags: List[int],
    oos_lags: List[int],
    stability: List[bool],
    output_dir: Path,
) -> Path:
    """Grouped bar chart comparing in-sample vs OOS lag for each system."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(systems))
    width = 0.32

    bars1 = ax.bar(x - width / 2, in_sample_lags, width,
                   color=PALETTE["primary"], alpha=0.85,
                   edgecolor="white", linewidth=0.5,
                   label="In-Sample (Q1)")
    bars2 = ax.bar(x + width / 2, oos_lags, width,
                   color=PALETTE["secondary"], alpha=0.85,
                   edgecolor="white", linewidth=0.5,
                   label="Out-of-Sample (Q2)")

    # Stability markers
    for i, (stable, isl, oosl) in enumerate(zip(stability, in_sample_lags, oos_lags)):
        marker = "STABLE" if stable else "UNSTABLE"
        colour = PALETTE["tertiary"] if stable else PALETTE["accent"]
        y_pos = max(isl, oosl) + 0.8
        ax.text(i, y_pos, marker, ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=colour,
                bbox=dict(boxstyle="round,pad=0.25", fc="white",
                          ec=colour, alpha=0.9, lw=0.8))

    # Tolerance band annotation
    ax.annotate(
        "Tolerance: $|\\tau_{train} - \\tau_{OOS}| \\leq 3$ min",
        xy=(0.98, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=8, style="italic",
        color=PALETTE["neutral"],
    )

    ax.set_xlabel("Gravitational System")
    ax.set_ylabel("Optimal Lag $\\tau^*$ (minutes)")
    ax.set_title("Lag Stability:  In-Sample vs. Out-of-Sample",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(systems, fontsize=9)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.set_ylim(0, max(max(in_sample_lags), max(oos_lags)) * 1.5 + 2)

    fig.tight_layout()
    out = output_dir / "lag_stability_comparison.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 4.  Friction Analysis (all systems)
# ---------------------------------------------------------------------------
def plot_friction_analysis(
    systems: List[str],
    predicted_bps: List[float],
    friction_bps: float,
    output_dir: Path,
) -> Path:
    """Horizontal bar chart: predicted edge vs. mechanical friction."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 4))

    y = np.arange(len(systems))
    bar_height = 0.5

    # Predicted move bars
    bars = ax.barh(y, predicted_bps, bar_height,
                   color=PALETTE["primary"], alpha=0.8,
                   edgecolor="white", linewidth=0.5,
                   label="Mean |Return| at $\\tau^*$ (bps)")

    # Friction threshold line
    ax.axvline(friction_bps, color=PALETTE["accent"], linewidth=2,
               linestyle="--", label=f"Round-trip Friction ({friction_bps:.0f} bps)",
               zorder=5)

    # Fill the "unprofitable" zone
    ax.axvspan(0, friction_bps, color=PALETTE["neg_fill"], alpha=0.2, zorder=0)
    ax.axvspan(friction_bps, ax.get_xlim()[1] if ax.get_xlim()[1] > friction_bps
               else friction_bps * 1.5,
               color="#d5efdb", alpha=0.2, zorder=0)

    # Labels
    ax.text(friction_bps / 2, len(systems) - 0.1, "UNPROFITABLE",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            color=PALETTE["accent"], alpha=0.6)
    ax.text(friction_bps + (max(predicted_bps + [friction_bps * 1.3]) - friction_bps) / 2,
            len(systems) - 0.1,
            "PROFITABLE", ha="center", va="bottom", fontsize=8.5,
            fontweight="bold", color=PALETTE["tertiary"], alpha=0.6)

    # Value annotations
    for i, (bp, sys) in enumerate(zip(predicted_bps, systems)):
        net = bp - friction_bps
        net_text = f"{bp:.1f} bps  (net: {net:+.1f})"
        colour = PALETTE["tertiary"] if net > 0 else PALETTE["accent"]
        ax.text(bp + 1, i, net_text, va="center", fontsize=8, color=colour,
                fontweight="medium")

    ax.set_yticks(y)
    ax.set_yticklabels(systems, fontsize=10)
    ax.set_xlabel("Basis Points (bps)")
    ax.set_title("Friction Accounting:  Predicted Signal vs. Exchange Costs",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=8.5)
    ax.set_xlim(0, max(max(predicted_bps), friction_bps) * 1.6)
    ax.invert_yaxis()

    fig.tight_layout()
    out = output_dir / "friction_analysis.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 5.  Summary Dashboard (multi-panel)
# ---------------------------------------------------------------------------
def plot_summary_dashboard(
    results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Multi-panel summary figure combining key metrics."""
    _apply_style()

    # Extract data from results
    systems = []
    in_sample_corrs = []
    oos_corrs = []
    granger_max_f = []
    in_lags = []
    oos_lags_list = []

    for key, data in results.items():
        if "error" in data and "validation" not in data:
            continue
        v = data.get("validation", {})
        if not v:
            continue
        label = f"{data['major']} -> {data['minor']}"
        systems.append(label)
        in_sample_corrs.append(abs(v.get("in_sample_peak_corr", 0)))
        oos_corrs.append(abs(v.get("oos_peak_corr", 0)))
        in_lags.append(v.get("in_sample_lag", 0))
        oos_lags_list.append(v.get("oos_lag", 0))

        # Max Granger F
        g_data = data.get("granger", [])
        if g_data:
            granger_max_f.append(max(g["f_statistic"] for g in g_data))
        else:
            granger_max_f.append(0)

    if not systems:
        return output_dir / "summary_dashboard.png"

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    colours = SYSTEM_COLOURS[:len(systems)]

    # --- Panel A: Peak CCF Correlation ---
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(systems))
    w = 0.32
    ax1.bar(x - w / 2, in_sample_corrs, w, color=PALETTE["primary"],
            alpha=0.8, label="In-Sample", edgecolor="white", linewidth=0.5)
    ax1.bar(x + w / 2, oos_corrs, w, color=PALETTE["secondary"],
            alpha=0.8, label="Out-of-Sample", edgecolor="white", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, fontsize=8)
    ax1.set_ylabel("$|\\hat{\\rho}(\\tau^*)|$")
    ax1.set_title("(a) Peak Cross-Correlation", fontweight="bold", fontsize=11)
    ax1.legend(fontsize=7.5, framealpha=0.9)

    # --- Panel B: Granger F-statistic (max) ---
    ax2 = fig.add_subplot(gs[0, 1])
    bars_g = ax2.bar(x, granger_max_f, 0.55, color=colours, alpha=0.8,
                     edgecolor="white", linewidth=0.5)
    for i, (bar, fval) in enumerate(zip(bars_g, granger_max_f)):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{fval:.1f}", ha="center", va="bottom", fontsize=8,
                 fontweight="medium", color="#333333")
    ax2.set_xticks(x)
    ax2.set_xticklabels(systems, fontsize=8)
    ax2.set_ylabel("F-statistic")
    ax2.set_title("(b) Max Granger F-Statistic (Lag 1)", fontweight="bold", fontsize=11)

    # --- Panel C: Lag Stability ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.bar(x - w / 2, in_lags, w, color=PALETTE["primary"],
            alpha=0.8, label="In-Sample $\\tau$", edgecolor="white", linewidth=0.5)
    ax3.bar(x + w / 2, oos_lags_list, w, color=PALETTE["secondary"],
            alpha=0.8, label="OOS $\\tau$", edgecolor="white", linewidth=0.5)
    for i, (il, ol) in enumerate(zip(in_lags, oos_lags_list)):
        stable = abs(il - ol) <= 3
        marker = "STABLE" if stable else "DRIFT"
        col = PALETTE["tertiary"] if stable else PALETTE["accent"]
        ax3.text(i, max(il, ol) + 0.5, marker, ha="center", fontsize=7.5,
                 fontweight="bold", color=col)
    ax3.set_xticks(x)
    ax3.set_xticklabels(systems, fontsize=8)
    ax3.set_ylabel("$\\tau^*$ (minutes)")
    ax3.set_title("(c) Lag Stability Across Quarters", fontweight="bold", fontsize=11)
    ax3.legend(fontsize=7.5, framealpha=0.9)

    # --- Panel D: Summary table ---
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")

    table_data = []
    col_labels = ["System", "$\\tau_{IS}$", "$\\tau_{OOS}$", "Stable?",
                  "$\\rho_{IS}$", "Granger F"]
    for i, s in enumerate(systems):
        stable_str = "Yes" if abs(in_lags[i] - oos_lags_list[i]) <= 3 else "No"
        table_data.append([
            s,
            f"{in_lags[i]} min",
            f"{oos_lags_list[i]} min",
            stable_str,
            f"{in_sample_corrs[i]:.4f}",
            f"{granger_max_f[i]:.1f}",
        ])

    table = ax4.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colColours=[PALETTE["light_fill"]] * len(col_labels),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1, 1.6)

    # Style header row
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_text_props(fontweight="bold", fontsize=8.5)
        cell.set_edgecolor("#cccccc")

    # Style data rows
    for i in range(len(table_data)):
        for j in range(len(col_labels)):
            cell = table[i + 1, j]
            cell.set_edgecolor("#dddddd")
            # Highlight stability column
            if j == 3:
                if table_data[i][3] == "Yes":
                    cell.set_facecolor("#d5efdb")
                else:
                    cell.set_facecolor("#fdd0a2")

    ax4.set_title("(d) Results Summary", fontweight="bold", fontsize=11, pad=15)

    fig.suptitle(
        "Crypto Lead-Lag Analysis:  Gravitational System Diagnostics",
        fontsize=14, fontweight="bold", y=0.98,
    )

    out = output_dir / "summary_dashboard.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# 6.  CCF Heatmap — all systems on one plot
# ---------------------------------------------------------------------------
def plot_ccf_heatmap(
    results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Heatmap of mean CCF across systems vs. lag."""
    _apply_style()

    systems = []
    mean_ccf_arrays = []

    for key, data in results.items():
        if "ccf" not in data or not data["ccf"]:
            continue
        label = f"{data['major']} -> {data['minor']}"
        systems.append(label)

        # Average correlations across epochs
        all_corrs = [np.array(epoch["correlations"]) for epoch in data["ccf"]]
        mean_corr = np.mean(all_corrs, axis=0)
        mean_ccf_arrays.append(mean_corr)

    if not systems:
        return output_dir / "ccf_heatmap.png"

    lags = np.array(data["ccf"][0]["lags"])  # same for all
    ccf_matrix = np.array(mean_ccf_arrays)

    fig, ax = plt.subplots(figsize=(10, 3.5))

    # Use a diverging colourmap centered at zero
    vmax = np.max(np.abs(ccf_matrix[:, lags != 0]))  # exclude lag-0 for scale
    im = ax.imshow(
        ccf_matrix, aspect="auto",
        cmap="RdBu_r", vmin=-vmax, vmax=vmax,
        extent=[lags[0] - 0.5, lags[-1] + 0.5, len(systems) - 0.5, -0.5],
        interpolation="nearest",
    )

    # Vertical line at lag = 0
    ax.axvline(0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.6)

    ax.set_yticks(range(len(systems)))
    ax.set_yticklabels(systems, fontsize=9)
    ax.set_xlabel("Lag $\\tau$ (minutes)")
    ax.set_title("Mean Cross-Correlation Heatmap Across Systems",
                 fontsize=12, fontweight="bold")

    # Colourbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("$\\hat{\\rho}(\\tau)$", fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Mark optimal lags with arrows
    for i, (key, data) in enumerate(
        [(k, v) for k, v in results.items() if "ccf" in v and v["ccf"]]
    ):
        opt_lags = [ep["optimal_lag"] for ep in data["ccf"]]
        from collections import Counter
        mode_lag = Counter(opt_lags).most_common(1)[0][0]
        ax.plot(mode_lag, i, marker="v", color=PALETTE["accent"],
                markersize=8, zorder=5)
        ax.annotate(f"$\\tau^*$={mode_lag}", xy=(mode_lag, i),
                    xytext=(mode_lag + 6, i - 0.15),
                    fontsize=7.5, color=PALETTE["accent"], fontweight="bold")

    fig.tight_layout()
    out = output_dir / "ccf_heatmap.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


# ===================================================================
# Main entry point
# ===================================================================
def generate_all_figures(
    results_path: Path,
    output_dir: Optional[Path] = None,
) -> List[Path]:
    """Load results JSON and generate all figures.

    Parameters
    ----------
    results_path : Path to ``full_results.json``
    output_dir : directory for figures (defaults to ``pipeline_results/figures/``)

    Returns
    -------
    List of paths to generated figures.
    """
    with open(results_path) as f:
        results = json.load(f)

    if output_dir is None:
        output_dir = results_path.parent / "figures"
    output_dir.mkdir(exist_ok=True)

    figures: List[Path] = []

    # --- Per-system figures ---
    valid_systems = []
    in_sample_lags = []
    oos_lags_all = []
    stability_flags = []
    predicted_bps_all = []

    for key, data in results.items():
        if "ccf" not in data or not data["ccf"]:
            continue

        label = f"{data['major']} -> {data['minor']}"

        # CCF correlogram
        fig_path = plot_ccf_correlogram(label, data["ccf"], output_dir)
        figures.append(fig_path)
        print(f"  [+] {fig_path.name}")

        # Granger F-stats
        if "granger" in data and data["granger"]:
            fig_path = plot_granger_fstats(label, data["granger"], 0.05, output_dir)
            figures.append(fig_path)
            print(f"  [+] {fig_path.name}")

        # Collect validation data
        v = data.get("validation", {})
        if v:
            valid_systems.append(label)
            in_sample_lags.append(v.get("in_sample_lag", 0))
            oos_lags_all.append(v.get("oos_lag", 0))
            stability_flags.append(v.get("lag_stable", False))

            # Mean predicted bps across friction checks
            friction = v.get("friction_checks", [])
            if friction:
                mean_pred = np.mean([fc["predicted_move_bps"] for fc in friction])
            else:
                mean_pred = 0.0
            predicted_bps_all.append(mean_pred)

    # --- Cross-system figures ---
    if valid_systems:
        fig_path = plot_lag_stability(
            valid_systems, in_sample_lags, oos_lags_all, stability_flags, output_dir
        )
        figures.append(fig_path)
        print(f"  [+] {fig_path.name}")

        fig_path = plot_friction_analysis(
            valid_systems, predicted_bps_all, 57.0, output_dir
        )
        figures.append(fig_path)
        print(f"  [+] {fig_path.name}")

    # Heatmap
    fig_path = plot_ccf_heatmap(results, output_dir)
    figures.append(fig_path)
    print(f"  [+] {fig_path.name}")

    # Summary dashboard
    fig_path = plot_summary_dashboard(results, output_dir)
    figures.append(fig_path)
    print(f"  [+] {fig_path.name}")

    return figures
# ---------------------------------------------------------------------------
# 7.  Robustness Data Gaps plots
# ---------------------------------------------------------------------------
def plot_robustness_comparison(
    system_label: str,
    results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Plot the degradation of CCF metrics under random data gaps."""
    _apply_style()
    
    fractions = []
    opt_lags = []
    corrs = []
    lags_std = []
    corrs_std = []
    
    if results.get("baseline"):
        bl_lags = [ep.optimal_lag for ep in results["baseline"]]
        bl_corrs = [ep.peak_correlation for ep in results["baseline"]]
        bl_corr_mean = np.mean(bl_corrs)
        import statistics
        bl_lag_mode = statistics.mode(bl_lags) if bl_lags else 0
        fractions.append(0.0)
        opt_lags.append(bl_lag_mode)
        corrs.append(bl_corr_mean)
        lags_std.append(np.std(bl_lags))
        corrs_std.append(np.std(bl_corrs))
        
    random_res = results.get("random", {})
    for frac_str in sorted(random_res.keys(), key=float):
        frac = float(frac_str)
        eps = random_res[frac_str]
        if eps:
            c_vals = [ep.peak_correlation for ep in eps]
            l_vals = [ep.optimal_lag for ep in eps]
            import statistics
            l_mode = statistics.mode(l_vals) if l_vals else 0
            
            fractions.append(frac)
            opt_lags.append(l_mode)
            corrs.append(np.mean(c_vals))
            lags_std.append(np.std(l_vals))
            corrs_std.append(np.std(c_vals))
            
    fig, ax1 = plt.subplots(figsize=(6, 4.5))
    
    color = PALETTE["primary"]
    ax1.set_xlabel("Missing Data Fraction")
    ax1.set_ylabel("Peak Cross-Correlation $\\rho$", color=color)
    ax1.plot(fractions, corrs, marker="o", linestyle="-", color=color, linewidth=2)
    ax1.fill_between(fractions, np.array(corrs) - np.array(corrs_std), np.array(corrs) + np.array(corrs_std), color=color, alpha=0.2, label="$\pm 1 \sigma$ (Correlation)")
    ax1.tick_params(axis='y', labelcolor=color)
    if corrs:
        ax1.set_ylim(0, max(np.array(corrs) + np.array(corrs_std)) * 1.1)
    
    ax2 = ax1.twinx()  
    color2 = PALETTE["secondary"]
    ax2.set_ylabel("Optimal Lag $\\tau^*$ (minutes)", color=color2)  
    ax2.plot(fractions, opt_lags, marker="s", linestyle="--", color=color2, linewidth=2)
    ax2.fill_between(fractions, np.array(opt_lags) - np.array(lags_std), np.array(opt_lags) + np.array(lags_std), color=color2, alpha=0.2, label="$\pm 1 \sigma$ (Lag)")
    ax2.tick_params(axis='y', labelcolor=color2)
    
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color("#333333")
    
    plt.title(f"Random Drop-out Robustness: {system_label}", fontweight="bold", fontsize=12)
    fig.tight_layout()
    
    out = output_dir / f"robustness_random_{system_label.replace(' -> ', '_').replace(' ', '')}.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out

def plot_robustness_periodic(
    system_label: str,
    results: Dict[str, Any],
    output_dir: Path,
) -> Path:
    """Plot the degradation of CCF metrics under periodic data gaps."""
    _apply_style()
    
    periodic_res = results.get("periodic", {})
    if not periodic_res:
         return output_dir / f"robustness_periodic_{system_label.replace(' -> ', '_').replace(' ', '')}.png"
         
    labels = []
    opt_lags = []
    corrs = []
    
    if results.get("baseline"):
        bl_lags = [ep.optimal_lag for ep in results["baseline"]]
        bl_corrs = [ep.peak_correlation for ep in results["baseline"]]
        import statistics
        bl_lag_mode = statistics.mode(bl_lags) if bl_lags else 0
        labels.append("Baseline (0m)")
        opt_lags.append(bl_lag_mode)
        corrs.append(np.mean(bl_corrs))

    for key, eps in periodic_res.items():
        if eps:
            c_vals = [ep.peak_correlation for ep in eps]
            l_vals = [ep.optimal_lag for ep in eps]
            import statistics
            l_mode = statistics.mode(l_vals) if l_vals else 0
            
            labels.append(key)
            corrs.append(np.mean(c_vals))
            opt_lags.append(l_mode)
            
    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    color = PALETTE["primary"]
    rects1 = ax1.bar(x - width/2, corrs, width, label='Peak Correlation $\\rho$', color=color, alpha=0.8)
    ax1.set_ylabel("Peak Cross-Correlation $\\rho$", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")

    ax2 = ax1.twinx()  
    color2 = PALETTE["secondary"]
    rects2 = ax2.bar(x + width/2, opt_lags, width, label='Optimal Lag $\\tau^*$', color=color2, alpha=0.8)
    ax2.set_ylabel("Optimal Lag $\\tau^*$ (minutes)", color=color2)  
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color("#333333")
    
    for i, v in enumerate(opt_lags):
         ax2.text(i + width/2, v + 0.5, str(v), color='#333333', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.title(f"Periodic Outage Robustness: {system_label}\nPeriod_Gap (mins)", fontweight="bold", fontsize=12)
    fig.tight_layout()

    out = output_dir / f"robustness_periodic_{system_label.replace(' -> ', '_').replace(' ', '')}.png"
    fig.savefig(out, facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    

    results_file = PATHS.root / "pipeline_results" / "full_results.json"
    print(f"Loading results from {results_file}...")
    figs = generate_all_figures(results_file)
    print(f"\nGenerated {len(figs)} figures.")


# ======================================================================
# --- FILE: plot_timeseries.py ---
# ======================================================================






def get_pair(ticker: str) -> str:
    return f"{TICKER_MAP[ticker]}{QUOTE_CURRENCY}"

def load_full_year_data(ticker: str) -> pd.DataFrame:
    pair = get_pair(ticker)
    dfs = []
    
    for quarter_dir in PATHS.candle_dirs:
        path = PATHS.candle_path(quarter_dir, pair, interval=1440)
        if path.exists():
            df = pd.read_csv(path, header=None, names=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
            dfs.append(df)
            
    if not dfs:
        return pd.DataFrame()
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df["time"] = pd.to_datetime(full_df["time"], unit="s")
    full_df = full_df.sort_values("time").drop_duplicates(subset=["time"])
    return full_df

def plot_normalized_series(tickers, title, filename):
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    
    plotted_any = False
    for ticker in tickers:
        df = load_full_year_data(ticker)
        if df.empty:
            print(f"No data found for {ticker}")
            continue
            
        first_valid = df["close"].dropna().iloc[0]
        df["norm_close"] = df["close"] / first_valid
        
        ax.plot(df["time"], df["norm_close"], label=ticker, linewidth=1.5)
        plotted_any = True
        
    if plotted_any:
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_ylabel("Normalized Price (Base = 1.0)")
        ax.set_xlabel("Date")
        ax.legend(framealpha=0.9, loc="upper left")
        fig.tight_layout()
        
        out_dir = PATHS.root / "pipeline_results" / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        fig.savefig(out_path, facecolor="white")
        print(f"Saved plot to {out_path}")
    else:
        print(f"Could not plot {filename} - no data")

if __name__ == "__main__":
    majors = ["BTC", "ETH", "SOL", "ADA", "DOGE"]
    minors = ["STX", "ARB", "JUP", "SNEK", "SHIB"]
    
    plot_normalized_series(majors, "Major Cryptos: Normalized Price", "major_cryptos_normalized.png")
    plot_normalized_series(minors, "Minor Cryptos: Normalized Price", "minor_cryptos_normalized.png")


# ======================================================================
# --- FILE: network_analysis.py ---
# ======================================================================


"""
Phase 6: Ecosystem Network Graph Analysis
===========================================
Treats the cryptocurrency market as a complex system and maps out the
flow of predictive capital. Computes pairwise Granger Causality across
all defined assets to construct a directed network graph.
"""









class EcosystemNetworkAnalyzer:
    """Orchestrates ecosystem-wide analysis to build a directed graph."""

    def __init__(
        self,
        quarter_l2: str,
        quarter_candle: str,
    ) -> None:
        self.quarter_l2 = quarter_l2
        self.quarter_candle = quarter_candle
        self.assets = list(TICKER_MAP.keys())
        self.signals: Dict[str, pl.DataFrame] = {}
        self.graph = nx.DiGraph()

    def load_and_preprocess_data(self) -> None:
        """Load candlestick data for all assets and compute log returns."""
        print(f"Loading data for {len(self.assets)} assets in {self.quarter_l2}...")
        for asset in self.assets:
            pair = get_kraken_pair(asset)
            try:
                engineer = DataEngineer(
                    pair=pair,
                    quarter_l2=self.quarter_l2,
                    quarter_candle=self.quarter_candle,
                    paths=PATHS,
                    config=CONFIG,
                )
                bars = engineer.build_minute_bars()
                stationary = compute_log_returns(bars, col="p_mid").drop_nulls()
                
                # We only need ts_truncated and log_return for Granger
                self.signals[asset] = stationary.select(["ts_truncated", "log_return"])
            except FileNotFoundError:
                print(f"  [Warning] Missing data for {asset} ({pair}). Skipping.")

    def compute_network(self, alpha: float = 0.05) -> nx.DiGraph:
        """Compute pairwise Granger Causality and build the directed graph."""
        if not self.signals:
            self.load_and_preprocess_data()

        print(f"Computing pairwise Granger Causality across {len(self.signals)} assets...")
        
        valid_assets = list(self.signals.keys())
        self.graph.add_nodes_from(valid_assets)

        total_pairs = len(valid_assets) * (len(valid_assets) - 1)
        completed = 0

        for major in valid_assets:
            for minor in valid_assets:
                if major == minor:
                    continue
                
                # Test if major Granger-causes minor
                results = granger_causality_test(
                    major_signal=self.signals[major],
                    minor_signal=self.signals[minor],
                    signal_col="log_return",
                    max_lag=1,  # Focus on lag 1 for the network edge
                    alpha=alpha,
                )
                
                if results and results[0].is_significant:
                    # Add edge: major -> minor
                    weight = results[0].f_statistic
                    self.graph.add_edge(major, minor, weight=weight)

                completed += 1
                if completed % 10 == 0:
                    print(f"  ... processed {completed}/{total_pairs} pairs")

        print(f"Network built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")
        return self.graph

    def plot_network(self, output_dir: Path) -> Path:
        """Visualize the network graph and save to disk."""
        if self.graph.number_of_nodes() == 0:
            self.compute_network()

        plt.figure(figsize=(14, 10), facecolor='white')
        ax = plt.gca()
        
        # Calculate out-degree (weighted) to size the nodes
        out_degrees = dict(self.graph.out_degree(weight='weight'))
        max_degree = max(out_degrees.values()) if out_degrees else 1.0
        
        node_sizes = [
            (out_degrees.get(node, 0) / max_degree) * 4000 + 1000
            for node in self.graph.nodes()
        ]
        
        # Calculate edge widths based on F-statistic
        edge_weights = [self.graph[u][v]['weight'] for u, v in self.graph.edges()]
        max_weight = max(edge_weights) if edge_weights else 1.0
        edge_widths = [(w / max_weight) * 4.0 + 0.5 for w in edge_weights]
        edge_colors = [w for w in edge_weights]

        # Use spring layout for a nice organic look
        pos = nx.spring_layout(self.graph, k=1.5, seed=42)

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, pos,
            node_size=node_sizes,
            node_color="#2c3e50",
            alpha=0.9,
            edgecolors="white",
            linewidths=2,
            ax=ax
        )
        
        # Draw edges with color mapping
        nx.draw_networkx_edges(
            self.graph, pos,
            width=edge_widths,
            edge_color=edge_colors,
            edge_cmap=plt.cm.viridis,
            edge_vmin=0,
            edge_vmax=max_weight,
            arrowsize=20,
            arrowstyle="->",
            connectionstyle="arc3,rad=0.1",
            alpha=0.7,
            ax=ax
        )
        
        # Add a colorbar for edge weights
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=max_weight))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.05)
        cbar.set_label('Granger F-Statistic (Strength of Lead)', rotation=270, labelpad=20)
            
        # Draw labels
        nx.draw_networkx_labels(
            self.graph, pos,
            font_size=12,
            font_color="white",
            font_weight="bold",
            font_family="sans-serif",
            ax=ax
        )

        plt.title(
            f"Crypto Ecosystem Predictive Network ({self.quarter_l2})\n"
            f"Nodes sized by total predictive influence. Edges show Granger Causality (Lag 1).",
            fontsize=16,
            pad=20
        )
        plt.axis('off')
        plt.tight_layout()

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"ecosystem_network_{self.quarter_l2}.png"
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved network visualization to {out_path}")
        return out_path


# ======================================================================
# --- FILE: robustness.py ---
# ======================================================================


"""
Robustness Testing Module
==========================
Analyzes how data gaps (missing observations) affect the CCF and 
Granger causality lag measurements. Can simulate both random drop-outs 
and periodic system downtime.
"""




def inject_random_gaps(df: pl.DataFrame, drop_fraction: float, seed: int = 42) -> pl.DataFrame:
    """Randomly drop a percentage of rows to simulate API network issues.
    
    Parameters
    ----------
    df : pl.DataFrame
        Signal dataframe (must contain timestamp and metric components)
    drop_fraction : float
        Percentage of rows to drop (0.0 to <1.0)
        
    Returns
    -------
    pl.DataFrame
        DataFrame with random rows dropped.
    """
    if drop_fraction <= 0.0:
        return df
    if drop_fraction >= 1.0:
        return df.clear()
        
    np.random.seed(seed)
    n = len(df)
    drop_indices = np.random.choice(n, size=int(n * drop_fraction), replace=False)
    
    # Create mask to keep rows not in drop_indices
    keep_indices = np.setdiff1d(np.arange(n), drop_indices)
    return df[keep_indices]


def inject_periodic_gaps(df: pl.DataFrame, period_minutes: int, gap_length_minutes: int) -> pl.DataFrame:
    """Drop blocks of rows systematically to simulate exchange maintenance or systematic ingestion outages.
    
    Parameters
    ----------
    df : pl.DataFrame
        Signal dataframe (must contain at least 'ts_truncated')
    period_minutes : int
        The period of the cycle (e.g., 60 for hourly).
    gap_length_minutes : int
        The duration of downtime in each period (e.g., 5 min downtime every 60m).
        
    Returns
    -------
    pl.DataFrame
        DataFrame with systematic drops.
    """
    if gap_length_minutes <= 0 or period_minutes <= 0:
        return df
        
    period_sec = period_minutes * 60
    gap_sec = gap_length_minutes * 60
    
    # We drop if the timestamp modulo period is less than the gap duration
    return df.filter((pl.col("ts_truncated") % period_sec) >= gap_sec)


def run_robustness_analysis(
    major_train: pl.DataFrame, 
    minor_train: pl.DataFrame,
    random_drop_fractions: List[float],
    periodic_cases: List[Tuple[int, int]],
    window_days: int = 30,
    max_lag: int = 60
) -> Dict[str, Any]:
    """Run CCF on clean and degraded datasets and return comparative metrics.
    
    Parameters
    ----------
    major_train : pl.DataFrame
        Major asset signal dataframe
    minor_train : pl.DataFrame
        Minor asset signal dataframe (will have gaps applied to it)
    random_drop_fractions : List[float]
        Severity levels for random dropping (e.g. [0.1, 0.2, 0.5])
    periodic_cases : List[Tuple[int, int]]
        List of (period_minutes, gap_length_minutes) for systematic drops.
        
    Returns
    -------
    Dictionary containing baseline and degraded CCF results.
    """
    
    results = {
        "baseline": [],
        "random": {},
        "periodic": {}
    }
    
    print("\n   [+] Computing CCF Baseline...")
    baseline_ccf = rolling_ccf(major_train, minor_train, window_days=window_days, max_lag=max_lag)
    results["baseline"] = baseline_ccf
    
    print("   [+] Simulating Random Gaps...")
    for frac in random_drop_fractions:
        minor_gappy = inject_random_gaps(minor_train, frac)
        ccf_res = rolling_ccf(major_train, minor_gappy, window_days=window_days, max_lag=max_lag)
        results["random"][f"{frac:.2f}"] = ccf_res
        
    print("   [+] Simulating Periodic Gaps...")
    for period, gap in periodic_cases:
        minor_gappy = inject_periodic_gaps(minor_train, period, gap)
        ccf_res = rolling_ccf(major_train, minor_gappy, window_days=window_days, max_lag=max_lag)
        results["periodic"][f"{period}m_{gap}m"] = ccf_res

    return results


# ======================================================================
# --- FILE: run_pipeline.py ---
# ======================================================================


"""
run_pipeline.py — Full 5-Phase Orchestrator
=============================================
Executes Phases 1–5 for every major→minor pairing and writes
results to ``pipeline_results/``.

Usage::

    python -m pipeline.run_pipeline
"""










RESULTS_DIR = PATHS.root / "pipeline_results"
RESULTS_DIR.mkdir(exist_ok=True)


def _serialize(obj: Any) -> Any:
    """JSON-serialisable conversion for numpy/dataclass types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _serialize(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, list):
        return [_serialize(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    return obj


def run_single_pair(
    major_name: str,
    minor_name: str,
) -> Dict[str, Any]:
    """Execute the full pipeline for one major→minor system."""
    major_pair = get_kraken_pair(major_name)
    minor_pair = get_kraken_pair(minor_name)

    print(f"\n{'='*60}")
    print(f"  System: {major_name} ({major_pair}) -> {minor_name} ({minor_pair})")
    print(f"{'='*60}")

    # Quarter mapping
    q_map = PATHS.quarter_pairs  # [(l2_dir, candle_dir), ...]
    train_l2, train_candle = q_map[0]  # Q1
    test_l2, test_candle = q_map[1]    # Q2

    report: Dict[str, Any] = {
        "major": major_name,
        "minor": minor_name,
        "major_pair": major_pair,
        "minor_pair": minor_pair,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # ── Phase 2: Data Engineering ──────────────────────────────────
    print("\n[Phase 2] Ingesting & processing L2 + candlestick data...")
    t0 = time.perf_counter()

    try:
        major_eng_train = DataEngineer(major_pair, train_l2, train_candle)
        minor_eng_train = DataEngineer(minor_pair, train_l2, train_candle)

        major_prices_train = major_eng_train.build_minute_bars()
        minor_prices_train = minor_eng_train.build_minute_bars()

        major_depth_train = major_eng_train.build_book_depth()
        minor_depth_train = minor_eng_train.build_book_depth()

        print(f"  Train major bars: {len(major_prices_train):,}")
        print(f"  Train minor bars: {len(minor_prices_train):,}")
    except FileNotFoundError as e:
        print(f"  [WARN] Skipping (file not found): {e}")
        report["error"] = str(e)
        return report

    try:
        major_eng_test = DataEngineer(major_pair, test_l2, test_candle)
        minor_eng_test = DataEngineer(minor_pair, test_l2, test_candle)

        major_prices_test = major_eng_test.build_minute_bars()
        minor_prices_test = minor_eng_test.build_minute_bars()

        minor_depth_test = minor_eng_test.build_book_depth()

        print(f"  Test major bars:  {len(major_prices_test):,}")
        print(f"  Test minor bars:  {len(minor_prices_test):,}")
    except FileNotFoundError as e:
        print(f"  [WARN] Skipping test quarter (file not found): {e}")
        report["error"] = str(e)
        return report

    print(f"  Phase 2 completed in {time.perf_counter() - t0:.1f}s")

    # ── Phase 3: Stationarity Transforms ──────────────────────────
    print("\n[Phase 3] Computing log-returns & OBI...")
    t0 = time.perf_counter()

    major_train = build_stationary_signals(major_prices_train, major_depth_train)
    minor_train = build_stationary_signals(minor_prices_train, minor_depth_train)
    major_test = build_stationary_signals(major_prices_test, minor_depth_test)

    # For test minor we only need minute bars + log returns
    
    minor_test_lr = compute_log_returns(minor_prices_test).drop_nulls()

    print(f"  Train major stationary rows: {len(major_train):,}")
    print(f"  Train minor stationary rows: {len(minor_train):,}")
    print(f"  Phase 3 completed in {time.perf_counter() - t0:.1f}s")

    # ── Phase 4: Signal Processing ────────────────────────────────
    print("\n[Phase 4] Running CCF + Granger causality...")
    t0 = time.perf_counter()

    # CCF
    ccf_results = rolling_ccf(
        major_train,
        minor_train,
        max_lag=CONFIG.ccf_max_lag,
        window_days=CONFIG.ccf_window_days,
    )
    print(f"  CCF epochs: {len(ccf_results)}")
    for cr in ccf_results:
        print(f"    tau = {cr.optimal_lag:+3d} min  |  rho = {cr.peak_correlation:+.4f}")

    report["ccf"] = [_serialize(r) for r in ccf_results]

    # Granger
    granger_results = granger_causality_test(
        major_train,
        minor_train,
        max_lag=CONFIG.granger_max_lag,
        alpha=CONFIG.granger_alpha,
    )
    sig_lags = [g for g in granger_results if g.is_significant]
    print(f"  Granger significant lags: {[g.lag_order for g in sig_lags]}")
    report["granger"] = [_serialize(g) for g in granger_results]

    # VAR
    var_info = var_model_analysis(
        major_train, minor_train, max_lag=CONFIG.granger_max_lag
    )
    if var_info:
        print(f"  VAR optimal order (AIC): {var_info['optimal_order']}")
        report["var_optimal_order"] = var_info["optimal_order"]
        report["var_aic"] = var_info["aic"]

    print(f"  Phase 4 completed in {time.perf_counter() - t0:.1f}s")

    # ── Phase 5: Out-of-Sample Validation ─────────────────────────
    print("\n[Phase 5] Out-of-sample validation & friction...")
    t0 = time.perf_counter()

    # Build test-quarter stationary signals for major
    try:
        major_depth_test = major_eng_test.build_book_depth()
        major_test_stat = build_stationary_signals(major_prices_test, major_depth_test)
    except Exception:
        major_test_stat = compute_log_returns(major_prices_test).drop_nulls()

    minor_test_stat = minor_test_lr

    oos_report = validate_out_of_sample(
        major_asset=major_name,
        minor_asset=minor_name,
        train_major=major_train,
        train_minor=minor_train,
        test_major=major_test_stat,
        test_minor=minor_test_stat,
        test_minor_depth=minor_depth_test,
        train_quarter=CONFIG.train_quarter,
        test_quarter=CONFIG.test_quarter,
        max_lag=CONFIG.ccf_max_lag,
        window_days=CONFIG.ccf_window_days,
        taker_fee=CONFIG.kraken_taker_fee,
    )

    print(f"  In-sample tau:  {oos_report.in_sample_lag} min")
    print(f"  OOS tau:        {oos_report.oos_lag} min")
    print(f"  Lag stable:   {oos_report.lag_stable}")
    print(f"  Profitable:   {oos_report.fraction_profitable:.1%}")
    print(f"  Phase 5 completed in {time.perf_counter() - t0:.1f}s")

    report["validation"] = _serialize(oos_report)

    return report


def main() -> None:
    """Run the pipeline for all configured systems."""
    all_reports: Dict[str, Any] = {}

    for major, minor in SYSTEM_PAIRINGS.items():
        try:
            report = run_single_pair(major, minor)
            all_reports[f"{major}__{minor}"] = report
        except Exception as exc:
            print(f"\n[X] Fatal error for {major}->{minor}: {exc}")
            all_reports[f"{major}__{minor}"] = {"error": str(exc)}

    # Write consolidated results
    out_path = RESULTS_DIR / "full_results.json"
    with open(out_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=_serialize)
    print(f"\n\n[OK] Results saved to {out_path}")


if __name__ == "__main__":
    main()


# ======================================================================
# --- FILE: run_network_analysis.py ---
# ======================================================================


"""
Execute the Ecosystem Network Graph Analysis.

This script runs the pairwise Granger Causality tests across all
assets in the ecosystem for a specified quarter and generates a
network graph visualization.
"""





def main():
    parser = argparse.ArgumentParser(description="Ecosystem Network Graph Analysis")
    parser.add_argument(
        "--quarter",
        type=str,
        default="Q1",
        help="The quarter to analyze (e.g., Q1, Q2). Default is Q1."
    )
    args = parser.parse_args()

    quarter_l2 = args.quarter
    # The candle directories in the project have the format Q1_Candlestick, Q2_candlestick, etc.
    if args.quarter == "Q1":
        quarter_candle = "Q1_Candlestick"
    else:
        quarter_candle = f"{args.quarter}_candlestick"

    print("=" * 60)
    print(f"Starting Ecosystem Network Graph Analysis for {args.quarter}")
    print("=" * 60)

    analyzer = EcosystemNetworkAnalyzer(
        quarter_l2=quarter_l2,
        quarter_candle=quarter_candle,
    )
    
    # Run the analysis and build the graph
    analyzer.compute_network(alpha=0.05)
    
    # Plot the results
    output_dir = PATHS.root / "pipeline_results" / "figures"
    analyzer.plot_network(output_dir=output_dir)

    print("=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()


# ======================================================================
# --- FILE: run_robustness.py ---
# ======================================================================


"""
Orchestrator for Robustness Analysis
====================================
Runs the data gap robustness tests on specified crypto pairs and generates plots.
"""








RESULTS_DIR = PATHS.root / "pipeline_results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def run_single_pair_robustness(major_name: str, minor_name: str) -> None:
    major_pair = get_kraken_pair(major_name)
    minor_pair = get_kraken_pair(minor_name)

    print(f"\n{'='*60}")
    print(f"  Robustness Analysis: {major_name} -> {minor_name}")
    print(f"{'='*60}")

    # Use Q1 (train) for robustness
    q_map = PATHS.quarter_pairs
    train_l2, train_candle = q_map[0]

    try:
        print("[Phase 2] Ingesting L2 + candlestick data...")
        major_eng = DataEngineer(major_pair, train_l2, train_candle)
        minor_eng = DataEngineer(minor_pair, train_l2, train_candle)

        major_prices = major_eng.build_minute_bars()
        minor_prices = minor_eng.build_minute_bars()
        major_depth = major_eng.build_book_depth()
        minor_depth = minor_eng.build_book_depth()

        print("[Phase 3] Computing stationarity transforms...")
        major_train = build_stationary_signals(major_prices, major_depth)
        minor_train = build_stationary_signals(minor_prices, minor_depth)

    except FileNotFoundError as e:
        print(f"  [WARN] Skipping (file not found): {e}")
        return

    # Define Gap scenarios
    random_fractions = [0.05, 0.10, 0.20, 0.30, 0.50]
    # (period_minutes, gap_length_minutes) e.g. 5 min every hour, 15 min every 4 hours, etc.
    periodic_cases = [
        (60, 5),      # 5m every 1h
        (240, 15),    # 15m every 4h
        (1440, 60),   # 1h every 24h
        (1440, 120),  # 2h every 24h
    ]

    print("[Phase 4] Running robustness CCF variations...")
    t0 = time.perf_counter()
    results = run_robustness_analysis(
        major_train, 
        minor_train, 
        random_drop_fractions=random_fractions,
        periodic_cases=periodic_cases,
        window_days=CONFIG.ccf_window_days,
        max_lag=CONFIG.ccf_max_lag
    )
    print(f"  Analysis completed in {time.perf_counter() - t0:.1f}s")
    
    # Generate Plots
    system_label = f"{major_name} -> {minor_name}"
    print(f"  Generating comparative plots for {system_label}...")
    f1 = plot_robustness_comparison(system_label, results, FIGURES_DIR)
    f2 = plot_robustness_periodic(system_label, results, FIGURES_DIR)
    print(f"  -> Saved {f1.name}")
    print(f"  -> Saved {f2.name}")


def main() -> None:
    # Run the robustness logic for all systems defined in the configuration.
    for major, minor in SYSTEM_PAIRINGS.items():
        try:
            run_single_pair_robustness(major, minor)
        except Exception as exc:
            print(f"\n[X] Fatal error for {major}->{minor}: {exc}")

if __name__ == "__main__":
    main()


# ======================================================================
# --- FILE: run_all.py ---
# ======================================================================


"""
Crypto Accretion Disk Analysis — Master Orchestrator
===================================================
A single entry point to execute the entire research pipeline.

Phases:
1. Normalized Price Visualization
2. Core Lead-Lag Pipeline (CCF, Granger, VAR, OOS Validation)
3. Ecosystem Network Graph Analysis
4. Data-Gap Robustness Analysis
5. Summary Dashboard Generation
"""

import sys
import time
from pathlib import Path
import json

# Ensure project root is in path for imports








def run_timeseries_plots():
    print("\n" + "="*70)
    print(" PHASE 1: GENERATING NORMALIZED PRICE PLOTS")
    print("="*70)
    majors = ["BTC", "ETH", "SOL", "ADA", "DOGE"]
    minors = ["STX", "ARB", "JUP", "SNEK", "SHIB"]
    
    plot_normalized_series(majors, "Major Cryptos: Normalized Price", "major_cryptos_normalized.png")
    plot_normalized_series(minors, "Minor Cryptos: Normalized Price", "minor_cryptos_normalized.png")

def run_main_pipeline():
    print("\n" + "="*70)
    print(" PHASE 2: CORE LEAD-LAG PIPELINE (ALL SYSTEMS)")
    print("="*70)
    
    results_dir = PATHS.root / "pipeline_results"
    results_dir.mkdir(exist_ok=True)
    
    all_reports = {}
    for major, minor in SYSTEM_PAIRINGS.items():
        try:
            report = run_single_pair(major, minor)
            all_reports[f"{major}__{minor}"] = report
        except Exception as exc:
            print(f"\n[X] Fatal error for {major}->{minor}: {exc}")
            all_reports[f"{major}__{minor}"] = {"error": str(exc)}

    out_path = results_dir / "full_results.json"
    with open(out_path, "w") as f:
        json.dump(all_reports, f, indent=2, default=_serialize)
    print(f"\n[OK] Consolidated results saved to {out_path}")
    return out_path

def run_network_graph():
    print("\n" + "="*70)
    print(" PHASE 3: ECOSYSTEM NETWORK GRAPH ANALYSIS")
    print("="*70)
    
    analyzer = EcosystemNetworkAnalyzer(
        quarter_l2="Q1",
        quarter_candle="Q1_Candlestick"
    )
    analyzer.compute_network(alpha=0.05)
    
    output_dir = PATHS.root / "pipeline_results" / "figures"
    analyzer.plot_network(output_dir=output_dir)

def run_robustness_tests():
    print("\n" + "="*70)
    print(" PHASE 4: DATA-GAP ROBUSTNESS ANALYSIS")
    print("="*70)
    
    for major, minor in SYSTEM_PAIRINGS.items():
        try:
            run_single_pair_robustness(major, minor)
        except Exception as exc:
            print(f"\n[X] Error in robustness for {major}->{minor}: {exc}")

def run_visualisation(results_path):
    print("\n" + "="*70)
    print(" PHASE 5: GENERATING FINAL VISUALIZATIONS")
    print("="*70)
    generate_all_figures(results_path)

def main():
    start_time = time.time()
    
    # 1. Plots
    run_timeseries_plots()
    
    # 2. Pipeline
    results_path = run_main_pipeline()
    
    # 3. Network
    run_network_graph()
    
    # 4. Robustness
    run_robustness_tests()
    
    # 5. Visualisation (Summary Dashboard, Heatmaps, etc.)
    run_visualisation(results_path)
    
    total_time = time.time() - start_time
    print("\n" + "="*70)
    print(f" FULL PROJECT EXECUTION COMPLETE IN {total_time/60:.2f} MINUTES")
    print(f" All artifacts available in: {PATHS.root / 'pipeline_results'}")
    print("="*70)

if __name__ == "__main__":
    main()
