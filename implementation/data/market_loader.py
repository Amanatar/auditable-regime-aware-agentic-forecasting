"""Real multi-regime market corpus loader and strictly causal point-in-time feature engineering.

Supports:
- Deterministic ingestion of daily OHLCV across 5 assets (SPY, QQQ, ^NSEI, RELIANCE.NS, HDFCBANK.NS)
- Frozen snapshot storage in data/eval/multi_asset_daily_2021_2026.csv committed to Git
- Strictly causal point-in-time feature pipeline with trailing log returns, realized volatility
  (Yang-Zhang and close-to-close), vol-of-vol ratios, drawdown from trailing peak, and causal rolling z-scores
- Deliberate leaked pipeline mode (full-sample fitting and forward-shifted windows) for ablation studies
- SHA-256 point-in-time filtration audit hashing integrated with cutoff_validator.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT_PATH = REPO_ROOT / "data" / "eval" / "multi_asset_daily_2021_2026.csv"
DEFAULT_MANIFEST_PATH = REPO_ROOT / "data" / "eval" / "multi_asset_manifest.json"

DEFAULT_SYMBOLS: List[str] = [
    "SPY",           # S&P 500 ETF (US Core Large Cap)
    "QQQ",           # Nasdaq 100 ETF (US Tech / Growth)
    "^NSEI",         # NIFTY 50 Index (Indian Benchmark)
    "RELIANCE.NS",   # Reliance Industries Ltd (Indian Conglomerate / Energy)
    "HDFCBANK.NS",   # HDFC Bank Ltd (Indian Financials / Banking)
]

DEFAULT_START = "2021-01-01"
DEFAULT_END = "2026-01-01"

ASSET_METADATA: Dict[str, Dict[str, str]] = {
    "SPY": {"name": "SPDR S&P 500 ETF Trust", "market": "US", "currency": "USD", "asset_class": "ETF"},
    "QQQ": {"name": "Invesco QQQ Trust", "market": "US", "currency": "USD", "asset_class": "ETF"},
    "^NSEI": {"name": "NIFTY 50", "market": "IN", "currency": "INR", "asset_class": "Index"},
    "RELIANCE.NS": {"name": "Reliance Industries Limited", "market": "IN", "currency": "INR", "asset_class": "Equity"},
    "HDFCBANK.NS": {"name": "HDFC Bank Limited", "market": "IN", "currency": "INR", "asset_class": "Equity"},
}


# ============================================================================
# 1. MARKET INGESTION & FROZEN SNAPSHOT STORAGE
# ============================================================================

def fetch_symbol_ohlcv(symbol: str, start: str = DEFAULT_START, end: str = DEFAULT_END) -> List[Dict[str, Any]]:
    """Fetch daily OHLCV from yfinance, standardizing to a clean list of row dicts."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is required to download raw market data. "
            "Install via `pip install yfinance` or load from frozen snapshot."
        ) from e

    # Download with auto_adjust=False to get raw Open/High/Low/Close plus Adj Close
    df = yf.download(symbol, start=start, end=end, auto_adjust=False, multi_level_index=False, progress=False)
    if df.empty:
        raise ValueError(f"No market data returned from yfinance for symbol {symbol} in [{start}, {end}].")

    rows: List[Dict[str, Any]] = []
    for dt_index, row in df.iterrows():
        # Date formatting
        date_str = dt_index.strftime("%Y-%m-%d") if hasattr(dt_index, "strftime") else str(dt_index)[:10]
        
        o = float(row.get("Open", row.get("open", 0.0)))
        h = float(row.get("High", row.get("high", 0.0)))
        l = float(row.get("Low", row.get("low", 0.0)))
        c = float(row.get("Close", row.get("close", 0.0)))
        ac = float(row.get("Adj Close", row.get("adj_close", c)))
        v = int(row.get("Volume", row.get("volume", 0)) or 0)

        rows.append({
            "symbol": symbol,
            "date": date_str,
            "open": round(o, 6),
            "high": round(h, 6),
            "low": round(l, 6),
            "close": round(c, 6),
            "adj_close": round(ac, 6),
            "volume": v,
        })

    # Sort strictly by date ascending
    rows.sort(key=lambda r: r["date"])
    return rows


def build_market_snapshot(
    symbols: List[str] = DEFAULT_SYMBOLS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    output_path: Union[Path, str] = DEFAULT_SNAPSHOT_PATH,
    manifest_path: Union[Path, str] = DEFAULT_MANIFEST_PATH,
) -> Path:
    """Download OHLCV across all assets and save a deterministic frozen CSV and JSON manifest."""
    output_file = Path(output_path)
    manifest_file = Path(manifest_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    symbol_stats: Dict[str, Dict[str, Any]] = {}

    for sym in symbols:
        sym_rows = fetch_symbol_ohlcv(sym, start=start, end=end)
        all_rows.extend(sym_rows)
        symbol_stats[sym] = {
            "metadata": ASSET_METADATA.get(sym, {}),
            "row_count": len(sym_rows),
            "start_date": sym_rows[0]["date"] if sym_rows else None,
            "end_date": sym_rows[-1]["date"] if sym_rows else None,
            "first_close": sym_rows[0]["close"] if sym_rows else None,
            "last_close": sym_rows[-1]["close"] if sym_rows else None,
        }

    # Sort deterministically by (symbol, date)
    all_rows.sort(key=lambda r: (r["symbol"], r["date"]))

    # Write clean CSV
    fieldnames = ["symbol", "date", "open", "high", "low", "close", "adj_close", "volume"]
    with output_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Compute SHA-256 of frozen file
    file_bytes = output_file.read_bytes()
    snapshot_sha256 = hashlib.sha256(file_bytes).hexdigest()

    # Write manifest
    manifest_data = {
        "dataset_name": "TRACE-Fin Real Multi-Regime Market Corpus (Frozen Snapshot)",
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "start": start,
        "end": end,
        "total_rows": len(all_rows),
        "file_path": str(output_file.relative_to(REPO_ROOT)).replace("\\", "/"),
        "file_sha256": snapshot_sha256,
        "symbols": symbols,
        "symbol_stats": symbol_stats,
    }
    manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    return output_file


def load_market_snapshot(
    path: Union[Path, str] = DEFAULT_SNAPSHOT_PATH,
    symbol: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Load market rows from the frozen snapshot file.
    
    If the file does not exist locally, attempts build_market_snapshot().
    """
    snapshot_file = Path(path)
    if not snapshot_file.exists():
        build_market_snapshot(output_path=snapshot_file)

    rows: List[Dict[str, Any]] = []
    with snapshot_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if symbol is not None and row["symbol"] != symbol:
                continue
            rows.append({
                "symbol": row["symbol"],
                "date": row["date"],
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "adj_close": float(row["adj_close"]),
                "volume": int(float(row["volume"] or 0)),
            })

    rows.sort(key=lambda r: (r["symbol"], r["date"]))
    return rows


# ============================================================================
# 2. STRICTLY CAUSAL POINT-IN-TIME FEATURE CALCULATORS
# ============================================================================

def compute_trailing_log_returns(
    closes: List[float],
    lags: Sequence[int] = (1, 5, 21, 63),
) -> Dict[int, List[Optional[float]]]:
    """Compute strictly causal trailing log returns: ln(Close_t / Close_{t - k}).
    
    For index t < k, returns None to avoid lookahead contamination.
    """
    n = len(closes)
    result: Dict[int, List[Optional[float]]] = {lag: [None] * n for lag in lags}
    for lag in lags:
        for t in range(lag, n):
            c_prev = closes[t - lag]
            c_curr = closes[t]
            if c_prev > 0 and c_curr > 0:
                result[lag][t] = round(math.log(c_curr / c_prev), 6)
    return result


def compute_close_to_close_volatility(
    closes: List[float],
    window: int,
) -> List[Optional[float]]:
    """Compute annualized close-to-close realized volatility over strictly trailing window [t - window + 1, t].
    
    Formula: sqrt(252 * sample_variance(daily_log_returns)).
    """
    n = len(closes)
    vols: List[Optional[float]] = [None] * n
    if n < window + 1:
        return vols

    # Compute 1-day log returns
    r1 = [None] + [math.log(closes[i] / closes[i - 1]) for i in range(1, n)]

    for t in range(window, n):
        # trailing slice of window daily returns: r1[t - window + 1 ... t]
        chunk = [r for r in r1[t - window + 1 : t + 1] if r is not None]
        if len(chunk) < window or window < 2:
            continue
        mean_r = sum(chunk) / window
        var_r = sum((x - mean_r) ** 2 for x in chunk) / (window - 1)
        ann_vol = math.sqrt(max(0.0, var_r) * 252.0)
        vols[t] = round(ann_vol, 6)

    return vols


def compute_yang_zhang_volatility(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    window: int,
) -> List[Optional[float]]:
    """Compute annualized Yang-Zhang (2000) realized volatility over strictly trailing window [t - window + 1, t].
    
    Yang-Zhang handles opening jumps, trend drift, and continuous volatility.
    Overnight jump: o_i = ln(Open_i / Close_{i-1})
    Open-to-close: c_i = ln(Close_i / Open_i)
    Rogers-Satchell: rs_i = ln(High_i / Close_i)*ln(High_i / Open_i) + ln(Low_i / Close_i)*ln(Low_i / Open_i)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    sigma_yz^2 = var(o) + k * var(c) + (1 - k) * mean(rs)
    Annualized: sqrt(252 * sigma_yz^2).
    """
    n = len(closes)
    vols: List[Optional[float]] = [None] * n
    if n < window + 1 or window < 2:
        return vols

    k = 0.34 / (1.34 + (window + 1.0) / (window - 1.0))

    for t in range(window, n):
        sub_c_prev = closes[t - window : t]
        sub_o = opens[t - window + 1 : t + 1]
        sub_h = highs[t - window + 1 : t + 1]
        sub_l = lows[t - window + 1 : t + 1]
        sub_c = closes[t - window + 1 : t + 1]

        overnight = [math.log(o / cp) for o, cp in zip(sub_o, sub_c_prev) if cp > 0 and o > 0]
        open_to_close = [math.log(c / o) for c, o in zip(sub_c, sub_o) if o > 0 and c > 0]
        rs = [
            math.log(h / c) * math.log(h / o) + math.log(l / c) * math.log(l / o)
            for h, l, c, o in zip(sub_h, sub_l, sub_c, sub_o)
            if c > 0 and o > 0 and h > 0 and l > 0
        ]

        if len(overnight) < window or len(open_to_close) < window or len(rs) < window:
            continue

        mean_o = sum(overnight) / window
        var_o = sum((x - mean_o) ** 2 for x in overnight) / (window - 1)

        mean_c = sum(open_to_close) / window
        var_c = sum((x - mean_c) ** 2 for x in open_to_close) / (window - 1)

        mean_rs = sum(rs) / window
        var_yz = var_o + k * var_c + (1.0 - k) * mean_rs
        ann_vol = math.sqrt(max(0.0, var_yz) * 252.0)
        vols[t] = round(ann_vol, 6)

    return vols


def compute_drawdown_from_peak(
    closes: List[float],
    window: int,
) -> List[Optional[float]]:
    """Compute drawdown from trailing peak over [t - window + 1, t]: (Close_t - Peak_t) / Peak_t."""
    n = len(closes)
    drawdowns: List[Optional[float]] = [None] * n
    for t in range(window - 1, n):
        chunk = closes[t - window + 1 : t + 1]
        peak = max(chunk)
        if peak > 0:
            drawdowns[t] = round((closes[t] - peak) / peak, 6)
    return drawdowns


def compute_causal_rolling_zscore(
    series: List[Optional[float]],
    window: int,
    eps: float = 1e-8,
) -> List[Optional[float]]:
    """Compute rolling z-score strictly over trailing window [t - window + 1, t].
    
    NEVER fits on future or full-sample statistics!
    z_t = (x_t - mean(x_{trailing})) / (std(x_{trailing}) + eps)
    """
    n = len(series)
    zscores: List[Optional[float]] = [None] * n
    for t in range(window - 1, n):
        chunk = [x for x in series[t - window + 1 : t + 1] if x is not None]
        if len(chunk) < window or window < 2:
            continue
        val = series[t]
        if val is None:
            continue
        mean_val = sum(chunk) / len(chunk)
        var_val = sum((x - mean_val) ** 2 for x in chunk) / (len(chunk) - 1)
        std_val = math.sqrt(max(0.0, var_val))
        zscores[t] = round((val - mean_val) / (std_val + eps), 6)
    return zscores


# ============================================================================
# 3. COMPLETE FEATURE PIPELINE (CAUSAL vs DELIBERATELY LEAKED)
# ============================================================================

def engineer_point_in_time_features(
    records: List[Dict[str, Any]],
    leaked_pipeline_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Engineer point-in-time quantitative features for an asset series.
    
    Args:
        records: List of daily OHLCV rows for a single symbol, sorted by date.
        leaked_pipeline_mode: If True, deliberately introduces:
            1) Full-sample global mean and standard deviation fitting (contaminating
               every historical fold with post-cutoff distribution parameters).
            2) Forward shift of returns and prices by t + 1 (incorporating tomorrow's
               market movement into today's feature vector).
            3) Provenance timestamp indicating input timestamp exceeds row date.
    
    Returns:
        List of feature row dictionaries enriched with audit metadata.
    """
    if not records:
        return []

    # Ensure records are sorted by date
    sorted_records = sorted(records, key=lambda r: r["date"])
    dates = [r["date"] for r in sorted_records]
    opens = [float(r["open"]) for r in sorted_records]
    highs = [float(r["high"]) for r in sorted_records]
    lows = [float(r["low"]) for r in sorted_records]
    closes = [float(r["close"]) for r in sorted_records]
    volumes = [float(r["volume"]) for r in sorted_records]
    symbol = sorted_records[0]["symbol"]
    n = len(sorted_records)

    # 1. Trailing Log Returns (1d, 5d, 21d, 63d)
    log_ret_dict = compute_trailing_log_returns(closes, lags=(1, 5, 21, 63))
    ret_1d = log_ret_dict[1]
    ret_5d = log_ret_dict[5]
    ret_21d = log_ret_dict[21]
    ret_63d = log_ret_dict[63]

    # 2. Realized Volatilities (10d, 21d, 63d)
    vol_c2c_10d = compute_close_to_close_volatility(closes, window=10)
    vol_c2c_21d = compute_close_to_close_volatility(closes, window=21)
    vol_c2c_63d = compute_close_to_close_volatility(closes, window=63)

    vol_yz_10d = compute_yang_zhang_volatility(opens, highs, lows, closes, window=10)
    vol_yz_21d = compute_yang_zhang_volatility(opens, highs, lows, closes, window=21)
    vol_yz_63d = compute_yang_zhang_volatility(opens, highs, lows, closes, window=63)

    # 3. Volatility-of-Volatility Ratios & Drawdowns
    vol_ratio_10_63: List[Optional[float]] = [None] * n
    vol_ratio_21_63: List[Optional[float]] = [None] * n
    vol_yz_ratio_10_63: List[Optional[float]] = [None] * n
    vol_yz_ratio_21_63: List[Optional[float]] = [None] * n
    vol_of_vol_std_63d: List[Optional[float]] = [None] * n

    for t in range(n):
        if vol_c2c_10d[t] is not None and vol_c2c_63d[t] is not None and vol_c2c_63d[t] > 0:
            vol_ratio_10_63[t] = round(vol_c2c_10d[t] / vol_c2c_63d[t], 6)
        if vol_c2c_21d[t] is not None and vol_c2c_63d[t] is not None and vol_c2c_63d[t] > 0:
            vol_ratio_21_63[t] = round(vol_c2c_21d[t] / vol_c2c_63d[t], 6)
        if vol_yz_10d[t] is not None and vol_yz_63d[t] is not None and vol_yz_63d[t] > 0:
            vol_yz_ratio_10_63[t] = round(vol_yz_10d[t] / vol_yz_63d[t], 6)
        if vol_yz_21d[t] is not None and vol_yz_63d[t] is not None and vol_yz_63d[t] > 0:
            vol_yz_ratio_21_63[t] = round(vol_yz_21d[t] / vol_yz_63d[t], 6)

        # Vol-of-vol standard deviation over trailing 63 days of 21d realized vol
        if t >= 63:
            v_chunk = [v for v in vol_c2c_21d[t - 63 + 1 : t + 1] if v is not None]
            if len(v_chunk) >= 60:
                m_v = sum(v_chunk) / len(v_chunk)
                s_v = math.sqrt(sum((x - m_v) ** 2 for x in v_chunk) / (len(v_chunk) - 1))
                vol_of_vol_std_63d[t] = round(s_v, 6)

    dd_peak_21d = compute_drawdown_from_peak(closes, window=21)
    dd_peak_63d = compute_drawdown_from_peak(closes, window=63)
    dd_peak_252d = compute_drawdown_from_peak(closes, window=252)

    # 4. Strictly Causal Rolling Z-Scores (NEVER full-sample StandardScaler fit!)
    zscore_ret_1d = compute_causal_rolling_zscore(ret_1d, window=63)
    zscore_ret_5d = compute_causal_rolling_zscore(ret_5d, window=63)
    zscore_vol_c2c_21d = compute_causal_rolling_zscore(vol_c2c_21d, window=63)
    zscore_vol_yz_21d = compute_causal_rolling_zscore(vol_yz_21d, window=63)
    zscore_vol_volume = compute_causal_rolling_zscore([v for v in volumes], window=21)

    # 5. Handle Deliberately Leaked Pipeline Mode (Ablation Benchmark)
    if leaked_pipeline_mode:
        # Contamination 1: Full-sample global scaler fit across the entire timeline
        valid_ret1 = [x for x in ret_1d if x is not None]
        global_mean_ret1 = sum(valid_ret1) / len(valid_ret1) if valid_ret1 else 0.0
        global_std_ret1 = math.sqrt(sum((x - global_mean_ret1) ** 2 for x in valid_ret1) / max(1, len(valid_ret1) - 1)) if len(valid_ret1) > 1 else 1.0

        valid_vol21 = [x for x in vol_c2c_21d if x is not None]
        global_mean_vol21 = sum(valid_vol21) / len(valid_vol21) if valid_vol21 else 0.0
        global_std_vol21 = math.sqrt(sum((x - global_mean_vol21) ** 2 for x in valid_vol21) / max(1, len(valid_vol21) - 1)) if len(valid_vol21) > 1 else 1.0

        leaked_zscore_ret_1d = [
            round((x - global_mean_ret1) / (global_std_ret1 + 1e-8), 6) if x is not None else None
            for x in ret_1d
        ]
        leaked_zscore_vol_21d = [
            round((x - global_mean_vol21) / (global_std_vol21 + 1e-8), 6) if x is not None else None
            for x in vol_c2c_21d
        ]

    # Assemble enriched output rows
    output_rows: List[Dict[str, Any]] = []
    for t in range(n):
        dt = dates[t]
        
        if not leaked_pipeline_mode:
            # Strictly Causal Filtration Mode F_t
            available_at = f"{dt}T23:59:59+00:00"
            input_max_date = dt
            row_dict: Dict[str, Any] = {
                "symbol": symbol,
                "date": dt,
                "available_at": available_at,
                "input_max_date": input_max_date,
                "is_causal": True,
                "open": opens[t],
                "high": highs[t],
                "low": lows[t],
                "close": closes[t],
                "volume": int(volumes[t]),
                "log_return_1d": ret_1d[t],
                "log_return_5d": ret_5d[t],
                "log_return_21d": ret_21d[t],
                "log_return_63d": ret_63d[t],
                "realized_vol_c2c_10d": vol_c2c_10d[t],
                "realized_vol_c2c_21d": vol_c2c_21d[t],
                "realized_vol_c2c_63d": vol_c2c_63d[t],
                "realized_vol_yz_10d": vol_yz_10d[t],
                "realized_vol_yz_21d": vol_yz_21d[t],
                "realized_vol_yz_63d": vol_yz_63d[t],
                "vol_of_vol_ratio_10_63": vol_ratio_10_63[t],
                "vol_of_vol_ratio_21_63": vol_ratio_21_63[t],
                "vol_of_vol_yz_ratio_10_63": vol_yz_ratio_10_63[t],
                "vol_of_vol_yz_ratio_21_63": vol_yz_ratio_21_63[t],
                "vol_of_vol_std_63d": vol_of_vol_std_63d[t],
                "drawdown_from_peak_21d": dd_peak_21d[t],
                "drawdown_from_peak_63d": dd_peak_63d[t],
                "drawdown_from_peak_252d": dd_peak_252d[t],
                "zscore_log_return_1d_63w": zscore_ret_1d[t],
                "zscore_log_return_5d_63w": zscore_ret_5d[t],
                "zscore_vol_c2c_21d_63w": zscore_vol_c2c_21d[t],
                "zscore_vol_yz_21d_63w": zscore_vol_yz_21d[t],
                "zscore_volume_21w": zscore_vol_volume[t],
            }
        else:
            # Deliberately Leaked Mode:
            # Shift features forward by t + 1 (so index t peeks into tomorrow's return)
            # and stamp with t + 1's observation timestamp
            next_t = min(n - 1, t + 1)
            next_dt = dates[next_t]
            available_at = f"{next_dt}T23:59:59+00:00" if next_t > t else f"{dt}T23:59:59+00:00"
            input_max_date = next_dt

            row_dict = {
                "symbol": symbol,
                "date": dt,
                "available_at": available_at,
                "input_max_date": input_max_date,
                "is_causal": False,
                "open": opens[t],
                "high": highs[t],
                "low": lows[t],
                "close": closes[t],
                "volume": int(volumes[t]),
                # Lookahead shift: row t receives t + 1's return and features
                "log_return_1d": ret_1d[next_t],
                "log_return_5d": ret_5d[next_t],
                "log_return_21d": ret_21d[next_t],
                "log_return_63d": ret_63d[next_t],
                "realized_vol_c2c_10d": vol_c2c_10d[next_t],
                "realized_vol_c2c_21d": vol_c2c_21d[next_t],
                "realized_vol_c2c_63d": vol_c2c_63d[next_t],
                "realized_vol_yz_10d": vol_yz_10d[next_t],
                "realized_vol_yz_21d": vol_yz_21d[next_t],
                "realized_vol_yz_63d": vol_yz_63d[next_t],
                "vol_of_vol_ratio_10_63": vol_ratio_10_63[next_t],
                "vol_of_vol_ratio_21_63": vol_ratio_21_63[next_t],
                "vol_of_vol_yz_ratio_10_63": vol_yz_ratio_10_63[next_t],
                "vol_of_vol_yz_ratio_21_63": vol_yz_ratio_21_63[next_t],
                "vol_of_vol_std_63d": vol_of_vol_std_63d[next_t],
                "drawdown_from_peak_21d": dd_peak_21d[next_t],
                "drawdown_from_peak_63d": dd_peak_63d[next_t],
                "drawdown_from_peak_252d": dd_peak_252d[next_t],
                # Contaminated by full-sample scaler
                "zscore_log_return_1d_63w": leaked_zscore_ret_1d[t],
                "zscore_log_return_5d_63w": zscore_ret_5d[next_t],
                "zscore_vol_c2c_21d_63w": leaked_zscore_vol_21d[t],
                "zscore_vol_yz_21d_63w": zscore_vol_yz_21d[next_t],
                "zscore_volume_21w": zscore_vol_volume[next_t],
            }

        output_rows.append(row_dict)

    return output_rows


# ============================================================================
# 4. POINT-IN-TIME FILTRATION & SHA-256 AUDIT HASHING
# ============================================================================

def compute_point_in_time_filtration_hash(
    records: List[Dict[str, Any]],
    cutoff: str,
    symbol: Optional[str] = None,
) -> str:
    """Compute a deterministic SHA-256 hash of the filtration F_cutoff.
    
    Filters records to date <= cutoff, formats canonical fields, and hashes.
    """
    cutoff_dt = datetime.fromisoformat(cutoff.replace("Z", "+00:00"))
    
    # Filter rows <= cutoff
    canonical_rows: List[Dict[str, Any]] = []
    for r in records:
        if symbol is not None and r.get("symbol") != symbol:
            continue
        row_dt = datetime.fromisoformat(str(r.get("available_at") or r["date"]).replace("Z", "+00:00"))
        if row_dt <= cutoff_dt:
            # Canonical representation
            canonical_rows.append({
                "symbol": r["symbol"],
                "date": r["date"],
                "close": r["close"],
                "ret_1d": r.get("log_return_1d"),
                "vol_yz_21d": r.get("realized_vol_yz_21d"),
                "z_ret_1d": r.get("zscore_log_return_1d_63w"),
            })

    canonical_rows.sort(key=lambda r: (r["symbol"], r["date"]))
    encoded = json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


# ============================================================================
# 5. CLI & WORKFLOW ENTRY POINT
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="TRACE-Fin Real Market Ingestion & Feature Engineering")
    parser.add_argument("--fetch", action="store_true", help="Download raw OHLCV from yfinance and save snapshot.")
    parser.add_argument("--snapshot", default=str(DEFAULT_SNAPSHOT_PATH), help="Path to frozen snapshot CSV.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH), help="Path to manifest JSON.")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS, help="List of symbols to process.")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end", default=DEFAULT_END, help="End date (YYYY-MM-DD).")
    args = parser.parse_args()

    snapshot_path = Path(args.snapshot)
    manifest_path = Path(args.manifest)

    if args.fetch or not snapshot_path.exists():
        print(f"[TRACE-Fin] Fetching 5+ years of market data for {len(args.symbols)} assets: {args.symbols}...")
        build_market_snapshot(
            symbols=args.symbols,
            start=args.start,
            end=args.end,
            output_path=snapshot_path,
            manifest_path=manifest_path,
        )
        print(f"[TRACE-Fin] Frozen snapshot saved to: {snapshot_path}")

    # Load and print summary stats
    all_rows = load_market_snapshot(snapshot_path)
    symbols = sorted({r["symbol"] for r in all_rows})
    print(f"\n{'='*70}")
    print(f"TRACE-Fin Multi-Asset Market Corpus Summary (Total Rows: {len(all_rows)})")
    print(f"{'='*70}")
    for sym in symbols:
        sym_rows = [r for r in all_rows if r["symbol"] == sym]
        print(f"  * Symbol: {sym:12s} | Rows: {len(sym_rows):5d} | Range: {sym_rows[0]['date']} to {sym_rows[-1]['date']} | Start Close: {sym_rows[0]['close']:.2f} | End Close: {sym_rows[-1]['close']:.2f}")

    # Run sample causal feature pipeline and print cutoff hash
    sample_cutoff = "2024-12-31T23:59:59+00:00"
    print(f"\n[TRACE-Fin] Engineering strictly causal point-in-time features...")
    causal_features = engineer_point_in_time_features(all_rows[:1255], leaked_pipeline_mode=False)
    audit_hash = compute_point_in_time_filtration_hash(causal_features, sample_cutoff)
    print(f"[TRACE-Fin] Sample Causal Filtration SHA-256 (Cutoff: {sample_cutoff}): {audit_hash}")


if __name__ == "__main__":
    main()
