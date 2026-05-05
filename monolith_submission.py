"""
CRYPTO ACCRETION DISK ANALYSIS — MONOLITHIC SUBMISSION
======================================================
This file contains the complete research pipeline including data engineering, 
stationarity transforms, signal processing, network analysis, robustness 
testing, and visualization.

Author: Armaan Sachdeva
Date: 2026-05-05
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import polars as pl
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch, FancyBboxPatch

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.api import VAR

# ===========================================================================
# 1. CONFIGURATION (formerly config.py)
# ===========================================================================

TICKER_MAP: Dict[str, str] = {
    "BTC": "XBT", "ETH": "ETH", "SOL": "SOL", "ADA": "ADA", "DOGE": "XDG",
    "STX": "STX", "ARB": "ARB", "JUP": "JUP", "SNEK": "SNEK", "SHIB": "SHIB",
}
QUOTE_CURRENCY: str = "USD"

SYSTEM_PAIRINGS: Dict[str, str] = {
    "BTC": "STX", "ETH": "ARB", "SOL": "JUP", "ADA": "SNEK", "DOGE": "SHIB",
}

@dataclass(frozen=True)
class DataPaths:
    root: Path = Path(r"c:\Users\singh\Crypto_study")
    l2_dirs: Tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")
    candle_dirs: Tuple[str, ...] = ("Q1_Candlestick", "Q2_candlestick", "Q3_candlestick", "Q4_candlestick")

    def l2_path(self, quarter: str, pair: str) -> Path:
        return self.root / quarter / f"{pair}.csv"

    def candle_path(self, quarter: str, pair: str, interval: int = 1) -> Path:
        return self.root / quarter / f"{pair}_{interval}.csv"

    @property
    def quarter_pairs(self) -> List[Tuple[str, str]]:
        return list(zip(self.l2_dirs, self.candle_dirs))

@dataclass(frozen=True)
class PipelineConfig:
    truncation_seconds: int = 60
    ccf_window_days: int = 30
    ccf_max_lag: int = 60
    granger_max_lag: int = 10
    granger_alpha: float = 0.05
    kraken_taker_fee: float = 0.0026
    train_quarter: str = "Q1"
    test_quarter: str = "Q2"

PATHS = DataPaths()
CONFIG = PipelineConfig()

def get_kraken_pair(asset: str) -> str:
    return f"{TICKER_MAP[asset]}{QUOTE_CURRENCY}"

# ===========================================================================
# 2. DATA ENGINEERING (formerly data_engineering.py)
# ===========================================================================

L2_COLUMNS = ["timestamp", "price", "volume"]
CANDLE_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "count"]

class DataEngineer:
    def __init__(self, pair: str, quarter_l2: str, quarter_candle: str):
        self.pair = pair
        self.quarter_l2 = quarter_l2
        self.quarter_candle = quarter_candle

    def _load_l2(self) -> pl.LazyFrame:
        path = PATHS.l2_path(self.quarter_l2, self.pair)
        if not path.exists(): raise FileNotFoundError(f"L2 file not found: {path}")
        return pl.scan_csv(path, has_header=False, new_columns=L2_COLUMNS,
                          dtypes={"timestamp": pl.Int64, "price": pl.Float64, "volume": pl.Float64}).with_columns(
            (pl.col("timestamp").floordiv(CONFIG.truncation_seconds).mul(CONFIG.truncation_seconds)).alias("ts_truncated")
        )

    def _load_candles(self) -> pl.LazyFrame:
        path = PATHS.candle_path(self.quarter_candle, self.pair, interval=1)
        if not path.exists(): raise FileNotFoundError(f"Candle file not found: {path}")
        return pl.scan_csv(path, has_header=False, new_columns=CANDLE_COLUMNS,
                          dtypes={"timestamp": pl.Int64, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64,
                                  "close": pl.Float64, "volume": pl.Float64, "count": pl.Int64}).select(
            pl.col("timestamp").alias("ts_truncated"), pl.col("close").alias("p_mid")
        )

    def build_minute_bars(self) -> pl.DataFrame:
        return self._load_candles().collect()

    def build_book_depth(self) -> pl.DataFrame:
        l2 = self._load_l2()
        candles = self._load_candles()
        enriched = l2.join(candles, on="ts_truncated", how="left").filter(pl.col("p_mid").is_not_null()).with_columns(
            pl.when(pl.col("price") < pl.col("p_mid")).then(pl.lit("bid"))
            .when(pl.col("price") > pl.col("p_mid")).then(pl.lit("ask"))
            .otherwise(pl.lit("at_mid")).alias("side")
        ).collect()
        
        bids = enriched.filter(pl.col("side") == "bid").group_by("ts_truncated").agg(pl.col("volume").sum().alias("v_bid"))
        asks = enriched.filter(pl.col("side") == "ask").group_by("ts_truncated").agg(pl.col("volume").sum().alias("v_ask"))
        return bids.join(asks, on="ts_truncated", how="outer_coalesce").with_columns(
            pl.col("v_bid").fill_null(0.0), pl.col("v_ask").fill_null(0.0)
        ).sort("ts_truncated")

# ===========================================================================
# 3. STATIONARITY (formerly stationarity.py)
# ===========================================================================

def compute_log_returns(prices: pl.DataFrame, col: str = "p_mid") -> pl.DataFrame:
    return prices.sort("ts_truncated").with_columns(
        (pl.col(col).log() - pl.col(col).shift(1).log()).alias("log_return")
    )

def compute_obi(depth: pl.DataFrame) -> pl.DataFrame:
    return depth.with_columns(
        pl.when((pl.col("v_bid") + pl.col("v_ask")) > 0)
        .then((pl.col("v_bid") - pl.col("v_ask")) / (pl.col("v_bid") + pl.col("v_ask")))
        .otherwise(0.0).alias("obi")
    )

def build_stationary_signals(prices: pl.DataFrame, depth: pl.DataFrame) -> pl.DataFrame:
    lr = compute_log_returns(prices)
    obi_df = compute_obi(depth)
    return lr.join(obi_df, on="ts_truncated", how="inner").sort("ts_truncated").drop_nulls(subset=["log_return", "obi"])

# ===========================================================================
# 4. SIGNAL PROCESSING (formerly signal_processing.py)
# ===========================================================================

@dataclass
class CCFResult:
    epoch_start: int; epoch_end: int; n_obs: int; optimal_lag: int; peak_correlation: float
    lags: np.ndarray; correlations: np.ndarray

@dataclass
class GrangerResult:
    lag_order: int; f_statistic: float; p_value: float; is_significant: bool

def cross_correlate(x: np.ndarray, y: np.ndarray, max_lag: int):
    n = len(x)
    x_dm, y_dm = x - x.mean(), y - y.mean()
    norm = np.sqrt(np.sum(x_dm**2) * np.sum(y_dm**2))
    if norm == 0: return np.arange(-max_lag, max_lag+1), np.zeros(2*max_lag+1)
    
    lags, ccf = [], []
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0: c = np.sum(x_dm[:n-lag] * y_dm[lag:])
        else: c = np.sum(x_dm[-lag:] * y_dm[:n+lag])
        lags.append(lag); ccf.append(c/norm)
    return np.array(lags), np.array(ccf)

def rolling_ccf(major_sig, minor_sig, signal_col="log_return", window_days=30, max_lag=60):
    aligned = major_sig.select("ts_truncated", pl.col(signal_col).alias("major")).join(
        minor_sig.select("ts_truncated", pl.col(signal_col).alias("minor")), on="ts_truncated", how="inner"
    ).sort("ts_truncated").drop_nulls()
    
    ts, maj, minr = aligned["ts_truncated"].to_numpy(), aligned["major"].to_numpy(), aligned["minor"].to_numpy()
    window_secs = window_days * 86400
    results, start, end_max = [], ts[0], ts[-1]
    
    while start + window_secs <= end_max:
        epoch_end = start + window_secs
        mask = (ts >= start) & (ts < epoch_end)
        x, y = maj[mask], minr[mask]
        if len(x) > 2 * max_lag:
            lags, ccf = cross_correlate(x, y, max_lag)
            pos_mask = lags > 0
            if np.any(pos_mask):
                idx = np.argmax(np.abs(ccf[pos_mask]))
                opt_lag, peak = lags[pos_mask][idx], ccf[pos_mask][idx]
            else: opt_lag, peak = 0, 0.0
            results.append(CCFResult(int(start), int(epoch_end), len(x), int(opt_lag), float(peak), lags, ccf))
        start = epoch_end
    return results

def granger_causality_test(major_sig, minor_sig, signal_col="log_return", max_lag=10, alpha=0.05):
    aligned = major_sig.select("ts_truncated", pl.col(signal_col).alias("major")).join(
        minor_sig.select("ts_truncated", pl.col(signal_col).alias("minor")), on="ts_truncated", how="inner"
    ).sort("ts_truncated").drop_nulls()
    data = np.column_stack([aligned["minor"].to_numpy(), aligned["major"].to_numpy()])
    results = []
    try:
        test_out = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        for lag in range(1, max_lag + 1):
            f, p = test_out[lag][0]["ssr_ftest"][0], test_out[lag][0]["ssr_ftest"][1]
            results.append(GrangerResult(lag, float(f), float(p), p < alpha))
    except: pass
    return results

# ===========================================================================
# 5. VALIDATION (formerly validation.py)
# ===========================================================================

@dataclass
class FrictionCheck:
    epoch_start: int; epoch_end: int; predicted_move_bps: float; total_friction_bps: float; net_edge_bps: float; is_profitable: bool

@dataclass
class ValidationReport:
    major_asset: str; minor_asset: str; train_quarter: str; test_quarter: str
    in_sample_lag: int; in_sample_peak_corr: float; oos_lag: int; oos_peak_corr: float
    lag_stable: bool; friction_checks: List[FrictionCheck]; fraction_profitable: float

def validate_out_of_sample(major_asset, minor_asset, train_major, train_minor, test_major, test_minor, test_minor_depth, train_q, test_q):
    train_ccf = rolling_ccf(train_major, train_minor)
    in_lag = int(np.unique([r.optimal_lag for r in train_ccf], return_counts=True)[0][np.argmax(np.unique([r.optimal_lag for r in train_ccf], return_counts=True)[1])]) if train_ccf else 0
    in_corr = float(np.mean([r.peak_correlation for r in train_ccf])) if train_ccf else 0.0

    test_ccf = rolling_ccf(test_major, test_minor)
    oos_lag = int(np.unique([r.optimal_lag for r in test_ccf], return_counts=True)[0][np.argmax(np.unique([r.optimal_lag for r in test_ccf], return_counts=True)[1])]) if test_ccf else 0
    oos_corr = float(np.mean([r.peak_correlation for r in test_ccf])) if test_ccf else 0.0

    total_f_bps = CONFIG.kraken_taker_fee * 10000 * 2 + 5.0
    ts, lr = test_minor["ts_truncated"].to_numpy(), test_minor["log_return"].to_numpy()
    window_secs, f_checks, start, end_max = 30 * 86400, [], ts[0], ts[-1]
    while start + window_secs <= end_max:
        epoch_end = start + window_secs
        mask = (ts >= start) & (ts < epoch_end)
        if len(lr[mask]) > in_lag:
            pred_bps = float(np.mean(np.abs(lr[mask][in_lag:]))) * 10000
            f_checks.append(FrictionCheck(int(start), int(epoch_end), round(pred_bps, 4), round(total_f_bps, 4), round(pred_bps - total_f_bps, 4), pred_bps > total_f_bps))
        start = epoch_end
    
    frac = sum(1 for fc in f_checks if fc.is_profitable) / len(f_checks) if f_checks else 0.0
    return ValidationReport(major_asset, minor_asset, train_q, test_q, in_lag, round(in_corr, 6), oos_lag, round(oos_corr, 6), abs(in_lag-oos_lag)<=3, f_checks, round(frac, 4))

# ===========================================================================
# 6. VISUALIZATION (formerly visualisation.py & plot_timeseries.py)
# ===========================================================================

PALETTE = {"primary": "#2c6fbb", "secondary": "#cc4c02", "tertiary": "#238b45", "accent": "#e31a1c", "neutral": "#636363", "light_fill": "#d0e1f9", "neg_fill": "#fdd0a2"}

def apply_style():
    plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200})

def plot_ccf(label, data, out_dir):
    apply_style(); fig, ax = plt.subplots(figsize=(6, 4))
    epoch = data[0]; lags, corrs = np.array(epoch.lags), np.array(epoch.correlations)
    ax.bar(lags, corrs, color=np.where(lags > 0, PALETTE["primary"], PALETTE["neutral"]), alpha=0.7)
    ax.axhspan(-1.96/np.sqrt(epoch.n_obs), 1.96/np.sqrt(epoch.n_obs), color=PALETTE["light_fill"], alpha=0.3)
    ax.set_title(f"CCF: {label}"); ax.set_xlabel("Lag (min)"); ax.set_ylabel("Correlation")
    safe_label = label.replace(' -> ', '_').replace('>', '_')
    fig.savefig(out_dir / f"ccf_{safe_label}.png"); plt.close(fig)

def plot_granger(label, data, out_dir):
    apply_style(); fig, ax = plt.subplots(figsize=(6, 4))
    lags = [g.lag_order for g in data]; f_stats = [g.f_statistic for g in data]
    ax.bar(lags, f_stats, color=[PALETTE["primary"] if g.is_significant else PALETTE["neutral"] for g in data])
    ax.set_title(f"Granger F-Stats: {label}"); ax.set_xlabel("Lag Order"); ax.set_ylabel("F-statistic")
    safe_label = label.replace(' -> ', '_').replace('>', '_')
    fig.savefig(out_dir / f"granger_{safe_label}.png"); plt.close(fig)

def plot_normalized(tickers, title, filename, out_dir):
    apply_style(); fig, ax = plt.subplots(figsize=(9, 5))
    for ticker in tickers:
        pair = get_kraken_pair(ticker); dfs = []
        for q in PATHS.candle_dirs:
            p = PATHS.candle_path(q, pair, 1440)
            if p.exists(): dfs.append(pd.read_csv(p, header=None, names=["t","o","h","l","c","v","vol","cnt"]))
        if dfs:
            df = pd.concat(dfs); df["t"] = pd.to_datetime(df["t"], unit="s")
            df = df.sort_values("t"); ax.plot(df["t"], df["c"]/df["c"].iloc[0], label=ticker)
    ax.set_title(title); ax.legend(); fig.savefig(out_dir / filename); plt.close(fig)

# ===========================================================================
# 7. NETWORK & ROBUSTNESS (formerly network_analysis.py & robustness.py)
# ===========================================================================

class NetworkAnalyzer:
    def __init__(self, q_l2, q_c):
        self.q_l2 = q_l2; self.q_c = q_c; self.graph = nx.DiGraph()
    def run(self, out_dir):
        assets = list(TICKER_MAP.keys()); signals = {}
        for a in assets:
            try:
                bars = DataEngineer(get_kraken_pair(a), self.q_l2, self.q_c).build_minute_bars()
                signals[a] = compute_log_returns(bars).drop_nulls()
            except: pass
        for maj in signals:
            for minr in signals:
                if maj == minr: continue
                res = granger_causality_test(signals[maj], signals[minr], max_lag=1)
                if res and res[0].is_significant: self.graph.add_edge(maj, minr, weight=res[0].f_statistic)
        plt.figure(figsize=(10, 8)); nx.draw(self.graph, with_labels=True, node_color="#2c3e50", font_color="white", font_weight="bold", arrowsize=20)
        plt.savefig(out_dir / f"network_{self.q_l2}.png"); plt.close()

def run_robustness(maj_sig, min_sig, label, out_dir):
    fractions = [0.1, 0.2, 0.3, 0.5]; corrs, lags = [], []
    for f in fractions:
        gappy = min_sig.sample(fraction=1-f)
        res = rolling_ccf(maj_sig, gappy)
        corrs.append(np.mean([r.peak_correlation for r in res]) if res else 0)
    apply_style(); fig, ax = plt.subplots(); ax.plot(fractions, corrs, marker="o")
    ax.set_title(f"Robustness: {label}"); ax.set_xlabel("Drop Fraction"); ax.set_ylabel("Mean Correlation")
    safe_label = label.replace(' -> ', '_').replace('>', '_')
    fig.savefig(out_dir / f"robustness_{safe_label}.png"); plt.close(fig)

# ===========================================================================
# 8. MASTER ORCHESTRATOR
# ===========================================================================

def serialize(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    if isinstance(obj, (bool, np.bool_)): return bool(obj)
    if hasattr(obj, "__dataclass_fields__"): return {k: serialize(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, list): return [serialize(i) for i in obj]
    if isinstance(obj, dict): return {k: serialize(v) for k, v in obj.items()}
    return obj

def main():
    res_dir = PATHS.root / "submission_results"; fig_dir = res_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True); all_reports = {}
    
    print("--- Phase 1: Baselines ---")
    plot_normalized(["BTC","ETH","SOL","ADA","DOGE"], "Majors", "majors.png", fig_dir)
    plot_normalized(["STX","ARB","JUP","SNEK","SHIB"], "Minors", "minors.png", fig_dir)
    
    print("--- Phase 2: Pipeline ---")
    for maj, minr in SYSTEM_PAIRINGS.items():
        print(f"  Processing {maj} -> {minr}...")
        try:
            eng_tr = DataEngineer(get_kraken_pair(maj), "Q1", "Q1_Candlestick")
            eng_mi_tr = DataEngineer(get_kraken_pair(minr), "Q1", "Q1_Candlestick")
            eng_te = DataEngineer(get_kraken_pair(maj), "Q2", "Q2_candlestick")
            eng_mi_te = DataEngineer(get_kraken_pair(minr), "Q2", "Q2_candlestick")
            
            sig_ma_tr = build_stationary_signals(eng_tr.build_minute_bars(), eng_tr.build_book_depth())
            sig_mi_tr = build_stationary_signals(eng_mi_tr.build_minute_bars(), eng_mi_tr.build_book_depth())
            sig_ma_te = compute_log_returns(eng_te.build_minute_bars()).drop_nulls()
            sig_mi_te = compute_log_returns(eng_mi_te.build_minute_bars()).drop_nulls()
            
            ccf = rolling_ccf(sig_ma_tr, sig_mi_tr); plot_ccf(f"{maj}->{minr}", ccf, fig_dir)
            granger = granger_causality_test(sig_ma_tr, sig_mi_tr); plot_granger(f"{maj}->{minr}", granger, fig_dir)
            val = validate_out_of_sample(maj, minr, sig_ma_tr, sig_mi_tr, sig_ma_te, sig_mi_te, None, "Q1", "Q2")
            run_robustness(sig_ma_tr, sig_mi_tr, f"{maj}->{minr}", fig_dir)
            
            all_reports[f"{maj}_{minr}"] = {"ccf": serialize(ccf), "granger": serialize(granger), "validation": serialize(val)}
        except Exception as e: print(f"    Error: {e}")

    print("--- Phase 3: Network ---")
    NetworkAnalyzer("Q1", "Q1_Candlestick").run(fig_dir)
    
    with open(res_dir / "results.json", "w") as f: json.dump(serialize(all_reports), f, indent=2)
    print(f"\n[DONE] All results in {res_dir}")

if __name__ == "__main__":
    main()
