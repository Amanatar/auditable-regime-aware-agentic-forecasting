# TRACE-Fin

<div align="center">

[![CI Verified](https://github.com/Amanatar/auditable-regime-aware-agentic-forecasting/actions/workflows/ci-verify.yml/badge.svg)](https://github.com/Amanatar/auditable-regime-aware-agentic-forecasting/actions)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/spaces/Amanatar/TRACE-Fin)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-port%207860-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Timestamp-Restricted, Regime-Aware, Evidence-Audited Forecasting & Conformal Calibration for Financial Markets**

</div>

---

## 🎯 Executive Summary & Core Innovation

Quantitative machine learning in finance suffers from a pervasive reproducibility crisis: backtests routinely achieve double-digit Sharpe ratios by peeking ahead through full-sample `StandardScaler` normalization, unaligned trading calendars, or uncalibrated point forecasts that churn capital in noise.

**`TRACE-Fin`** resolves this through an auditable, end-to-end framework built on three pillars:

1. **Strict Point-in-Time $\mathcal{F}_t$ Filtration:** Every regime quantile, forecast, and risk measure is computed strictly over trailing windows $[t-W, t]$ with zero future leakage. Across **`1,772` out-of-sample steps (`2020-01-01` to `2026-10-30`)**, `cutoff_validator.py` confirms **`0` causal violations** and seals the evaluation trace with SHA-256 while catching **`100.0%` (`1,772/1,772`)** of deliberately leaked $\mathcal{F}_{t+1}$ steps.
2. **Regime-Conditional Adaptive Conformal Inference (ACI):** Scales non-conformity residuals by causal volatility and updates interval quantiles online via Gibbs–Candès dynamics ($\alpha_{t+1} = \alpha_t + \gamma (\alpha - \text{err}_t)$), restoring **93.60% empirical coverage in High-Volatility stress** (vs. **84.14%** under static split conformal) with **89.81% overall calibration** and lower Winkler Interval Score across every regime ([`results/regime_coverage.csv`](results/regime_coverage.csv)).
3. **Uncertainty-Gated Execution Net of 10 bps One-Way Costs:** Translates conformal widths and multi-horizon causal trend estimates into volatility-damped weights with a $6.0\%$ turnover hurdle gate—delivering **$1.189$ Net Sharpe** (**vs. `0.963` Equal-Weight Buy & Hold** and `-1.075` Uncalibrated Point Forecast), **$1.680$ Sortino**, **44.9% lower Max Drawdown** (`11.76%` vs. `21.34%`), and slashing annualized turnover to **$4.19\times$** (vs. $121.86\times$).

---

## 📊 Multi-Regime Out-of-Sample Walk-Forward Benchmark (`2020-01-01` to `2026-10-30`)

Evaluated across **1,772 contiguous out-of-sample trading days** (`SPY`, `QQQ`, `^NSEI`, `TLT`, `GLD`; `2018–2019` causal burn-in; net of `10 bps` one-way transaction costs) from [`results/walk_forward_summary.json`](results/walk_forward_summary.json):

### 1. 4-Track Strategy Execution Comparison

| Strategy Track | Gross Ann. Ret | **Net Ann. Ret** | Ann. Vol | **Net Sharpe** | **Sortino** | **Max Drawdown** | **Annualized Turnover** | Cost Drag |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Equal-Weight Buy & Hold** | $11.08\%$ | $11.03\%$ | $11.56\%$ | $0.963$ | $1.360$ | $21.34\%$ | $0.47\times$ | $0.05\%$ |
| **2. Uncalibrated Point Forecast** | $-1.18\%$ | $-12.52\%$ | $11.80\%$ | $-1.075$ | $-1.427$ | $68.24\%$ | $121.86\times$ | $12.19\%$ |
| **3. `TRACE-Fin` Causal Regime + 90% Conformal + Cost Gate** | **$11.20\%$** | **$10.74\%$** | **$8.92\%$** | **$1.189$** | **$1.680$** | **$11.76\%$** | **$4.19\times$** | **$0.42\%$** |
| **4. Diagnostic 1-Day Lookahead Leak Control** *(Leaked)* | $1556.55\%$ | $1056.63\%$ | $16.08\%$ | $15.379$ | $495.977$ | $0.19\%$ | $363.07\times$ | $36.31\%$ |

### 2. Empirical 90% Conformal Coverage by Causal Volatility Regime ([`results/regime_coverage.csv`](results/regime_coverage.csv))

| Conformal Method | Causal Regime | $N$ Obs | Nominal Target | **Empirical Coverage %** | Mean Width % | **Winkler Interval Score** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`static_split_conformal`** | `Calm` | $5,014$ | $90.0\%$ | $94.10\%$ | $3.635\%$ | $0.04221$ |
| **`static_split_conformal`** | `High-Volatility` | $3,846$ | $90.0\%$ | **$84.14\%$** *(Breach)* | $3.688\%$ | $0.07264$ |
| **`static_split_conformal`** | `Overall` | $8,860$ | $90.0\%$ | $89.77\%$ | $3.658\%$ | $0.05542$ |
| **`trace_fin_regime_aci`** | `Calm` | $5,014$ | $90.0\%$ | $86.90\%$ | **$2.772\%$** | **$0.04131$** |
| **`trace_fin_regime_aci`** | `High-Volatility` | $3,846$ | $90.0\%$ | **$93.60\%$** *(Protected)* | $6.025\%$ | **$0.06942$** |
| **`trace_fin_regime_aci`** | **`Overall`** | **$8,860$** | **$90.0\%$** | **$89.81\%$** | **$4.184\%$** | **$0.05351$** |

### 3. Lookahead Contamination Audit

| Pipeline Track | Filtration Invariant | Steps Verified | Causal Violations | Leak Flagging Rate |
| :--- | :--- | :---: | :---: | :---: |
| **`TRACE-Fin` Causal (`Tracks 1–3`)** | Strictly $\mathcal{F}_t$ (`validate_available_at`) | `1,772` | **`0` (`0.00%`)** | N/A |
| **Diagnostic Leak Control (`Track 4`)** | $\mathcal{F}_{t+1}$ 1-day lookahead | `1,772` | Intentional | **`1,772 / 1,772` (`100.0%`)** |

See [`RESULTS.md`](RESULTS.md) for detailed walk-forward analysis and honest engineering postmortems.

---

## 🖥️ Interactive Streamlit Workbench (Port 7860)

The interactive analytics workbench provides three deep-dive tabs for quants, portfolio managers, and recruiters:

- **Tab 1: Live Walk-Forward Conformal & Regime Explorer:** Real-time asset selector (`SPY`, `QQQ`, `^NSEI`, `RELIANCE.NS`, `HDFCBANK.NS`, or pooled portfolio), reactive transaction cost slider ($0\text{--}30\text{ bps}$), confidence slider ($80\%\text{--}95\%$), color-coded regime bands, and rolling conformal intervals.
- **Tab 2: Point-in-Time Cutoff & Lookahead Leakage Auditor:** Recruiter toggle injecting a synthetic 1-day scaler/window leak to demonstrate `cutoff_validator.py` flagging exact timestamps, SHA-256 hash mismatches, and lookahead inflation.
- **Tab 3: Multi-Asset Benchmark & Failure Postmortem Inspector:** Complete empirical ablation tables, institutional artifact downloads, and honest postmortem cards.

---

## 🚀 Quickstart & Local Execution

### 1. Run Full Verification Suite (`pytest`)
```bash
pip install -r requirements.txt
python -m pytest -v
```

### 2. Rebuild Offline Market Cache & Walk-Forward Replay
```bash
python src/fetch_market_data.py
python src/walk_forward_benchmark.py
```

### 3. Launch Local Streamlit Workbench
```bash
streamlit run app.py --server.port 7860 --theme.base dark
```
Open [http://localhost:7860](http://localhost:7860) in your browser.

### 4. Docker Container Build & Run
```bash
docker build -t trace-fin .
docker run -p 7860:7860 trace-fin
```

### 5. Deploy to Hugging Face Spaces (Docker SDK)
```bash
# In PowerShell:
$env:HF_TOKEN = "<YOUR_HF_TOKEN>"
python deploy_hf_space.py --space_name TRACE-Fin
```
Live Hugging Face Space URL: **https://huggingface.co/spaces/Amanatar/TRACE-Fin**

---

## 📂 Repository Architecture

```
├── app.py                             # Interactive Streamlit Workbench (Port 7860)
├── Dockerfile                         # Production non-root (uid=1000) Docker container
├── .dockerignore                      # Docker context ignore rules
├── deploy_hf_space.py                 # Automated HF Spaces API deployment runner
├── requirements.txt                   # Production dependencies (incl. pyarrow)
├── RESULTS.md                         # Institutional walk-forward & failure postmortem report
├── src/
│   ├── fetch_market_data.py           # Offline Parquet OHLCV cache builder (SPY, QQQ, ^NSEI, TLT, GLD)
│   └── walk_forward_benchmark.py      # Causal 2020-2026 walk-forward replay & regime coverage benchmark
├── data/
│   ├── market_cache/
│   │   ├── daily_2018_2026.parquet    # 0.39 MB compressed OHLCV cache (2018-01-02 to 2026-10-30)
│   │   └── daily_2018_2026_manifest.json
│   └── eval/
│       ├── multi_asset_daily_2021_2026.csv
│       └── multi_asset_manifest.json
├── implementation/
│   ├── data/
│   │   ├── cutoff_validator.py        # Strict point-in-time filtration & SHA-256 auditor
│   │   └── market_loader.py           # Causal feature pipeline & Yang-Zhang volatility
│   ├── models/
│   │   ├── regime_detector.py         # Causal regime detector & volatility quantiles
│   │   ├── forecasting_engine.py      # Ridge / Huber regularized walk-forward ensemble
│   │   └── conformal_calibrator.py    # Split conformal & Gibbs-Candès online ACI
│   ├── portfolio/
│   │   ├── position_sizer.py          # Uncertainty-inverse sizing & turnover hurdle gate
│   │   └── backtest_engine.py         # Daily accounting, transaction friction & drawdowns
│   └── experiments/                   # Legacy daily trajectory & ablation runners
└── results/
    ├── walk_forward_summary.json      # 4-track OOS metrics + causality audit (2020-2026)
    ├── regime_coverage.csv            # Empirical 90% conformal coverage across Calm vs High-Vol
    ├── conformal_ablation.json
    ├── portfolio_backtest_summary.json
    └── walkforward_daily_trajectories.json
```

---

## 📜 License
MIT License. See [LICENSE](LICENSE) for details.
