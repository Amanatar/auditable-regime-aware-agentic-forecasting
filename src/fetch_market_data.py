"""
Offline Market Data Cache Builder for TRACE-Fin.
Repository: Amanatar/auditable-regime-aware-agentic-forecasting
Author: Principal Quant & ML Systems Engineer

Fetches daily OHLCV (2018-01-01 to 2026-10-30) for:
  - SPY   (US Large Cap Core Equity)
  - QQQ   (US Tech / Growth Equity)
  - ^NSEI (NIFTY 50 Indian Benchmark Index)
  - TLT   (20+ Year US Treasury Bond ETF - Defensive Fixed Income)
  - GLD   (SPDR Gold Shares ETF - Safe Haven / Inflation Hedge)

Saves a compact (< 1.5 MB) deterministic Parquet cache to:
  data/market_cache/daily_2018_2026.parquet
"""

from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "data" / "market_cache"
PARQUET_PATH = CACHE_DIR / "daily_2018_2026.parquet"
MANIFEST_PATH = CACHE_DIR / "daily_2018_2026_manifest.json"

SYMBOLS = ["SPY", "QQQ", "^NSEI", "TLT", "GLD"]
START_DATE = "2018-01-01"
END_DATE = "2026-10-31"  # Exclusive in yfinance -> covers through 2026-10-30


def fetch_and_cache_market_data(
    symbols: list[str] = SYMBOLS,
    start: str = START_DATE,
    end: str = END_DATE,
    out_path: Path = PARQUET_PATH
) -> Path:
    print("=" * 75)
    print("TRACE-FIN OFFLINE MARKET DATA CACHE BUILDER (2018-01-01 to 2026-10-30)")
    print("=" * 75)

    frames = []
    for sym in symbols:
        print(f"  Fetching {sym:<8} [{start} -> 2026-10-30]...", end=" ", flush=True)
        df = yf.download(
            sym,
            start=start,
            end=end,
            auto_adjust=False,
            multi_level_index=False,
            progress=False
        )
        if df.empty:
            raise RuntimeError(f"Failed to download data for {sym}")

        # Handle possible MultiIndex columns depending on yfinance version
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df["date"] = pd.to_datetime(df[date_col]).dt.strftime("%Y-%m-%d")

        clean = pd.DataFrame({
            "date": df["date"],
            "symbol": sym,
            "open": pd.to_numeric(df["Open"], errors="coerce").astype("float32"),
            "high": pd.to_numeric(df["High"], errors="coerce").astype("float32"),
            "low": pd.to_numeric(df["Low"], errors="coerce").astype("float32"),
            "close": pd.to_numeric(df["Close"], errors="coerce").astype("float32"),
            "adj_close": pd.to_numeric(df.get("Adj Close", df["Close"]), errors="coerce").astype("float32"),
            "volume": pd.to_numeric(df["Volume"], errors="coerce").fillna(0).astype("int64"),
        })
        # Strictly filter to <= 2026-10-30 and drop NaN close
        clean = clean[(clean["date"] >= "2018-01-01") & (clean["date"] <= "2026-10-30")]
        clean = clean.dropna(subset=["close"]).drop_duplicates(subset=["date", "symbol"])
        print(f"{len(clean)} rows ({clean['date'].min()} -> {clean['date'].max()})")
        frames.append(clean)

    full_df = pd.concat(frames, ignore_index=True)
    full_df = full_df.sort_values(["date", "symbol"]).reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_parquet(out_path, index=False, compression="snappy")

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"\n[OUTPUT] Saved Parquet cache: {out_path} ({size_mb:.3f} MB, {len(full_df)} total rows)")
    assert size_mb < 1.5, f"Cache size {size_mb:.3f} MB exceeds 1.5 MB limit!"

    manifest = {
        "symbols": symbols,
        "requested_start": START_DATE,
        "requested_end": "2026-10-30",
        "actual_min_date": str(full_df["date"].min()),
        "actual_max_date": str(full_df["date"].max()),
        "total_rows": int(len(full_df)),
        "size_mb": round(size_mb, 4),
        "generated_utc": datetime.now(timezone.utc).isoformat()
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OUTPUT] Saved manifest: {MANIFEST_PATH}")
    return out_path


if __name__ == "__main__":
    fetch_and_cache_market_data()
