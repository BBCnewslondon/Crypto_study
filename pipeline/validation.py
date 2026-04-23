"""
Phase 5: Reality Testing — Out-of-Sample Validation & Friction
================================================================
Validates that the lag τ identified in-sample (train quarter) persists
in unseen data (test quarter) and that the predicted price movement
exceeds exchange mechanical friction (Kraken fees + bid-ask spread).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .signal_processing import cross_correlate, CCFResult


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
    from .signal_processing import rolling_ccf

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
