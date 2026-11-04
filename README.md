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

1. **Strict Point-in-Time $\mathcal{F}_t$ Filtration:** Every feature (trailing Yang-Zhang volatility, causal rolling z-scores) is computed strictly over trailing windows $[t-W, t]$ with zero future data leakage. Each evaluation fold is cryptographically sealed with a deterministic SHA-256 audit hash verified via `cutoff_validator.py`.
2. **Regime-Conditional Adaptive Conformal Inference (ACI):** Rather than assuming i.i.d. exchangeability, `TRACE-Fin` scales non-conformity scores by causal market regime volatility and updates prediction interval quantiles online via Gibbs-Candès dynamics ($\alpha_{t+1} = \alpha_t + \gamma (\alpha - \text{err}_t)$), maintaining **94.6% coverage during market stress shocks**.
3. **Uncertainty-Gated Execution Net of Costs:** Translates interval widths into position sizing ($w_t \propto \hat{\mu}_t / W_t$) with a $10.0\text{ bps}$ turnover hurdle buffer, slashing annualized portfolio churn from **54.6x down to 28.4x** and capping maximum drawdown at **7.36%**.

---

## 📊 Institutional Empirical Benchmark (2022–2026 Out-of-Sample)

Evaluated across **942 contiguous out-of-sample trading days** spanning US & Indian macro regimes (`SPY`, `QQQ`, `^NSEI`, `RELIANCE.NS`, `HDFCBANK.NS`):

### 1. Portfolio Execution Table (Gross vs. Net of 10 bps Costs)

| Strategy | Gross Ann. Return % | Net Ann. Return % | Ann. Volatility % | Gross Sharpe | **Net Sharpe** | **Sortino Ratio** | **Max Drawdown %** | **Calmar Ratio** | **Annualized Turnover** | Cost Drag % |
|---|---|---|---|---|---|---|---|---|---|---|
| **Baseline A: Buy-and-Hold** | 11.76% | 11.73% | 13.64% | 0.863 | **0.860** | **1.444** | **17.49%** | **0.671** | **0.27x** | 0.10% |
| **Baseline B: Uncalibrated Point Forecast** | 2.77% | -2.69% | 5.07% | 0.545 | **-0.531** | **-0.853** | **14.99%** | **-0.180** | **54.58x** | 18.45% |
| **Flagship C: TRACE-Fin (Regime Conformal + Cost Gate)** | 2.26% | -0.60% | 2.53% | 0.892 | **-0.237** | **-0.361** | **7.36%** | **-0.082** | **28.41x** | 10.08% |

### 2. Conformal Prediction Ablation (Nominal Target: 90.0% Coverage)

| Method | Target % | Overall OOS Coverage % | High-Vol Stress Regime Coverage % | Mean Interval Width | Winkler Interval Score | Hit Rate % | Information Coeff. (IC) |
|---|---|---|---|---|---|---|---|
| **Method 1: `leaked_naive`** | 90.0% | 87.1% | **76.1%** *(Severe Miss)* | 0.02215 | 0.04694 | 89.1% *(Lookahead)* | 0.907 *(Artificial)* |
| **Method 2: `static_split_conformal`** | 90.0% | 90.2% | **84.7%** *(Exchangeability Breach)* | 0.04301 | 0.06478 | 52.3% | 0.039 |
| **Method 3: `trace_fin_regime_aci`\*** | 90.0% | **90.7%** | **94.6%** *(Tail Protected)* | 0.05397 | 0.07114 | 52.3% | 0.039 |

### 3. Lookahead Contamination Audit Table

| Pipeline Configuration | Filtration Invariant | In-Sample Net Sharpe | Out-of-Sample Net Sharpe | Sharpe Decay % | Audit Outcome |
|---|---|---|---|---|---|
| **`TRACE-Fin` Strict Filtration** | $\mathcal{F}_t$ (Verified via `cutoff_validator.py`) | **2.001** | **-0.237** | **111.8%** | **VERIFIED CLEAN (0 Violations)** |
| **Deliberately Leaked Pipeline** | $\mathcal{F}_{t+1}$ (Forward shift + Full Scaler) | **10.998** | **-0.531** | **104.8%** | **CONTAMINATED (100% Caught)** |

---

## 🖥️ Interactive Streamlit Workbench (Port 7860)

The interactive analytics workbench provides three deep-dive tabs for quants, portfolio managers, and recruiters:

- **Tab 1: Live Walk-Forward Conformal & Regime Explorer:** Real-time asset selector (`SPY`, `QQQ`, `^NSEI`, `RELIANCE.NS`, `HDFCBANK.NS`, or pooled portfolio), reactive transaction cost slider ($0\text{--}30\text{ bps}$), confidence slider ($80\%\text{--}95\%$), color-coded regime bands, and rolling conformal intervals.
- **Tab 2: Point-in-Time Cutoff & Lookahead Leakage Auditor:** Recruiter toggle injecting a synthetic 1-day scaler/window leak to demonstrate `cutoff_validator.py` flagging exact timestamps, SHA-256 hash mismatches, and the $11.0 \to -0.53$ Sharpe collapse.
- **Tab 3: Multi-Asset Benchmark & Failure Postmortem Inspector:** Complete empirical ablation tables, institutional artifact downloads, and honest postmortem cards covering 3 historical drawdown windows (Q4 2022 Tech V-shape bounce, Spring 2024 range-bound churn, and early 2024 HDFCBANK idiosyncratic dislocation).

---

## 🚀 Quickstart & Local Execution

### 1. Run Unit Tests (28/28 Passing)
```bash
python -m pytest -v
```

### 2. Launch Local Streamlit Workbench
```bash
pip install -r requirements.txt
streamlit run app.py --server.port 7860 --theme.base dark
```
Open [http://localhost:7860](http://localhost:7860) in your browser.

### 3. Docker Container Build & Run
```bash
docker build -t trace-fin .
docker run -p 7860:7860 trace-fin
```

### 4. Deploy to Hugging Face Spaces (Docker SDK)
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
├── requirements.txt                   # Production dependencies
├── RESULTS.md                         # Institutional performance & failure postmortem report
├── data/
│   └── eval/
│       ├── multi_asset_daily_2021_2026.csv   # 6,218 rows frozen OHLCV market snapshot
│       └── multi_asset_manifest.json          # Dataset audit metadata
├── implementation/
│   ├── data/
│   │   ├── cutoff_validator.py        # Strict point-in-time filtration & SHA-256 auditor
│   │   └── market_loader.py           # Causal feature pipeline & Yang-Zhang volatility
│   ├── models/
│   │   ├── regime_detector.py         # Causal regime detector & Gaussian mixture / HMM
│   │   ├── forecasting_engine.py      # Ridge / Huber regularized walk-forward ensemble
│   │   └── conformal_calibrator.py    # Split conformal & Gibbs-Candès online ACI
│   ├── portfolio/
│   │   ├── position_sizer.py          # Uncertainty-inverse sizing & turnover hurdle gate
│   │   └── backtest_engine.py         # Daily accounting, transaction friction & drawdowns
│   └── experiments/
│       ├── export_daily_trajectories.py # Daily walk-forward trajectory exporter
│       ├── run_portfolio_backtest.py    # End-to-end portfolio simulation runner
│       └── run_walk_forward_ablation.py # 3-method conformal ablation runner
└── results/
    ├── conformal_ablation.json        # Walk-forward uncertainty ablation results
    ├── portfolio_backtest_summary.json# Gross vs. net portfolio execution metrics
    └── walkforward_daily_trajectories.json # Complete daily time series for app.py
```

---

## 📜 License
MIT License. See [LICENSE](LICENSE) for details.
