"""
Phase 6: Ecosystem Network Graph Analysis
===========================================
Treats the cryptocurrency market as a complex system and maps out the
flow of predictive capital. Computes pairwise Granger Causality across
all defined assets to construct a directed network graph.
"""
from __future__ import annotations

from typing import Dict, Tuple, List
from pathlib import Path

import polars as pl
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from pipeline.config import TICKER_MAP, PATHS, CONFIG, get_kraken_pair
from pipeline.data_engineering import DataEngineer
from pipeline.stationarity import compute_log_returns
from pipeline.signal_processing import granger_causality_test


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
