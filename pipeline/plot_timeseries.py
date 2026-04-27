import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from pipeline.config import TICKER_MAP, QUOTE_CURRENCY, PATHS
from pipeline.visualisation import _apply_style, PALETTE

def get_pair(ticker: str) -> str:
    return f"{TICKER_MAP[ticker]}{QUOTE_CURRENCY}"

def load_full_year_data(ticker: str) -> pd.DataFrame:
    pair = get_pair(ticker)
    dfs = []
    
    for quarter_dir in PATHS.candle_dirs:
        path = PATHS.candle_path(quarter_dir, pair, interval=1440)
        if path.exists():
            df = pd.read_csv(path, header=None, names=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
            dfs.append(df)
            
    if not dfs:
        return pd.DataFrame()
        
    full_df = pd.concat(dfs, ignore_index=True)
    full_df["time"] = pd.to_datetime(full_df["time"], unit="s")
    full_df = full_df.sort_values("time").drop_duplicates(subset=["time"])
    return full_df

def plot_normalized_series(tickers, title, filename):
    _apply_style()
    fig, ax = plt.subplots(figsize=(9, 5))
    
    plotted_any = False
    for ticker in tickers:
        df = load_full_year_data(ticker)
        if df.empty:
            print(f"No data found for {ticker}")
            continue
            
        first_valid = df["close"].dropna().iloc[0]
        df["norm_close"] = df["close"] / first_valid
        
        ax.plot(df["time"], df["norm_close"], label=ticker, linewidth=1.5)
        plotted_any = True
        
    if plotted_any:
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_ylabel("Normalized Price (Base = 1.0)")
        ax.set_xlabel("Date")
        ax.legend(framealpha=0.9, loc="upper left")
        fig.tight_layout()
        
        out_dir = PATHS.root / "pipeline_results" / "figures"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename
        fig.savefig(out_path, facecolor="white")
        print(f"Saved plot to {out_path}")
    else:
        print(f"Could not plot {filename} - no data")

if __name__ == "__main__":
    majors = ["BTC", "ETH", "SOL", "ADA", "DOGE"]
    minors = ["STX", "ARB", "JUP", "SNEK", "SHIB"]
    
    plot_normalized_series(majors, "Major Cryptos: Normalized Price", "major_cryptos_normalized.png")
    plot_normalized_series(minors, "Minor Cryptos: Normalized Price", "minor_cryptos_normalized.png")
