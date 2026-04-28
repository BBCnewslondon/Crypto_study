"""
Execute the Ecosystem Network Graph Analysis.

This script runs the pairwise Granger Causality tests across all
assets in the ecosystem for a specified quarter and generates a
network graph visualization.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.config import PATHS
from pipeline.network_analysis import EcosystemNetworkAnalyzer

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
