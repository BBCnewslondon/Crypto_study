"""
run_pipeline.py — Full 5-Phase Orchestrator
=============================================
Executes Phases 1–5 for every major→minor pairing and writes
results to ``pipeline_results/``.

Usage::

    python -m pipeline.run_pipeline
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

import polars as pl
import numpy as np

from pipeline.config import (
    SYSTEM_PAIRINGS,
    CONFIG,
    PATHS,
    get_kraken_pair,
)
from pipeline.data_engineering import DataEngineer
from pipeline.stationarity import build_stationary_signals
from pipeline.signal_processing import (
    rolling_ccf,
    granger_causality_test,
    var_model_analysis,
)
from pipeline.validation import validate_out_of_sample


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
    from pipeline.stationarity import compute_log_returns
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
