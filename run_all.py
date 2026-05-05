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
root_dir = Path(__file__).parent.absolute()
sys.path.append(str(root_dir))

from pipeline.config import PATHS, SYSTEM_PAIRINGS, CONFIG
from pipeline.plot_timeseries import plot_normalized_series
from pipeline.run_pipeline import run_single_pair, _serialize
from pipeline.network_analysis import EcosystemNetworkAnalyzer
from pipeline.run_robustness import run_single_pair_robustness
from pipeline.visualisation import generate_all_figures

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
