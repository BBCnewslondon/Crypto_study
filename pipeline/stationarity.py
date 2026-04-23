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
from __future__ import annotations

import polars as pl
import numpy as np


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
