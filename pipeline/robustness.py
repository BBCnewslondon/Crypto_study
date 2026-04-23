"""
Robustness Testing Module
==========================
Analyzes how data gaps (missing observations) affect the CCF and 
Granger causality lag measurements. Can simulate both random drop-outs 
and periodic system downtime.
"""
from __future__ import annotations

import numpy as np
import polars as pl
from typing import List, Dict, Any, Tuple

from pipeline.signal_processing import rolling_ccf, CCFResult

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
