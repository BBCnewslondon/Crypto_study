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
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import polars as pl
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
