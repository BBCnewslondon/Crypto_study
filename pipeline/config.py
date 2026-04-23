"""
Phase 1: System Definition — Asset Selection & Configuration
=============================================================
Defines the "Massive Body" → "Accretion Disk" pairings and all
global constants used throughout the pipeline.

Kraken uses 'XBT' for Bitcoin and 'XDG' for Dogecoin in their
pair naming. We map human-readable names to Kraken tickers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Ticker mapping: human-readable → Kraken L2/candle file prefix
# ---------------------------------------------------------------------------
TICKER_MAP: Dict[str, str] = {
    "BTC": "XBT",
    "ETH": "ETH",
    "SOL": "SOL",
    "ADA": "ADA",
    "DOGE": "XDG",
    "STX": "STX",
    "ARB": "ARB",
    "JUP": "JUP",
    "SNEK": "SNEK",
    "SHIB": "SHIB",
}

# Quote currency used across all pairs
QUOTE_CURRENCY: str = "USD"


# ---------------------------------------------------------------------------
# Massive Body → Accretion Disk mapping
# ---------------------------------------------------------------------------
SYSTEM_PAIRINGS: Dict[str, str] = {
    # BTC  → STX   (Stacks runs smart-contracts *on* Bitcoin; direct dependency)
    "BTC": "STX",
    # ETH  → ARB   (Arbitrum is an Ethereum L2 rollup; inherits ETH settlement)
    "ETH": "ARB",
    # SOL  → JUP   (Jupiter is the dominant DEX aggregator on Solana)
    "SOL": "JUP",
    # ADA  → SNEK  (SNEK is the leading Cardano-native meme/community token)
    "ADA": "SNEK",
    # DOGE → SHIB  (SHIB emerged as the "Dogecoin killer" meme competitor)
    "DOGE": "SHIB",
}


def get_kraken_pair(asset: str) -> str:
    """Return the Kraken file-prefix for a given human-readable asset name.

    Example
    -------
    >>> get_kraken_pair("BTC")
    'XBTUSD'
    """
    return f"{TICKER_MAP[asset]}{QUOTE_CURRENCY}"


# ---------------------------------------------------------------------------
# Filesystem layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DataPaths:
    """Centralised path registry so no module hard-codes paths."""

    root: Path = Path(r"c:\Users\singh\Crypto_study")

    # Order-book (L2) directories — one per quarter
    l2_dirs: Tuple[str, ...] = ("Q1", "Q2", "Q3", "Q4")

    # Candlestick directories — one per quarter
    candle_dirs: Tuple[str, ...] = (
        "Q1_Candlestick",
        "Q2_candlestick",
        "Q3_candlestick",
        "Q4_candlestick",
    )

    def l2_path(self, quarter: str, pair: str) -> Path:
        """Return path to an L2 CSV: e.g. Q1/XBTUSD.csv"""
        return self.root / quarter / f"{pair}.csv"

    def candle_path(self, quarter: str, pair: str, interval: int = 1) -> Path:
        """Return path to a candle CSV: e.g. Q1_Candlestick/XBTUSD_1.csv"""
        return self.root / quarter / f"{pair}_{interval}.csv"

    @property
    def quarter_pairs(self) -> List[Tuple[str, str]]:
        """Yield (l2_dir, candle_dir) tuples aligned by quarter."""
        return list(zip(self.l2_dirs, self.candle_dirs))


# ---------------------------------------------------------------------------
# Pipeline hyper-parameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PipelineConfig:
    """All tuneable knobs in one place."""

    # Phase 2 – timestamp truncation granularity (seconds)
    truncation_seconds: int = 60

    # Phase 4 – CCF rolling-window size (calendar days)
    ccf_window_days: int = 30

    # Phase 4 – maximum CCF lag to search (minutes)
    ccf_max_lag: int = 60

    # Phase 4 – VAR model maximum lag order for Granger test
    granger_max_lag: int = 10

    # Phase 4 – Granger significance level
    granger_alpha: float = 0.05

    # Phase 5 – Kraken taker fee (bps → decimal)
    kraken_taker_fee: float = 0.0026  # 26 bps

    # Phase 5 – Kraken maker fee (bps → decimal)
    kraken_maker_fee: float = 0.0016  # 16 bps

    # Phase 5 – train quarter, test quarter
    train_quarter: str = "Q1"
    test_quarter: str = "Q2"


# Singleton instances for convenience
PATHS = DataPaths()
CONFIG = PipelineConfig()
