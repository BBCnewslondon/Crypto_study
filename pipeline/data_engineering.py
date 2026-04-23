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
from __future__ import annotations

from pathlib import Path
from typing import Optional

import polars as pl

from .config import PATHS, CONFIG, get_kraken_pair, DataPaths, PipelineConfig


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
