"""
Orchestrator for Robustness Analysis
====================================
Runs the data gap robustness tests on specified crypto pairs and generates plots.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from pipeline.config import SYSTEM_PAIRINGS, CONFIG, PATHS, get_kraken_pair
from pipeline.data_engineering import DataEngineer
from pipeline.stationarity import build_stationary_signals
from pipeline.robustness import run_robustness_analysis
from pipeline.visualisation import plot_robustness_comparison, plot_robustness_periodic

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
