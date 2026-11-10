# Empirical Walk-Forward Benchmark & Regime Conformal Audit: `TRACE-Fin`

**Project:** `TRACE-Fin` (Timestamp-Restricted, Regime-Aware, Evidence-Audited Forecasting)  
**Repository:** `Amanatar/auditable-regime-aware-agentic-forecasting`  
**Evaluation Window:** `2020-01-01` to `2026-10-30` (`1,772` Out-of-Sample Trading Days; `2018-01-02` to `2019-12-31` Burn-In)  
**Multi-Asset Universe:** `SPY` (US Core Equity), `QQQ` (US Tech Equity), `^NSEI` (NIFTY 50), `TLT` (20Y+ US Treasuries), `GLD` (Gold Bullion)  
**Offline Dataset Cache:** [`data/market_cache/daily_2018_2026.parquet`](data/market_cache/daily_2018_2026.parquet) (`0.387 MB`, `11,056` rows)  
**Transaction Cost Assumption:** `10.0 bps` (`0.10%`) one-way execution fee per unit of dollar turnover  

---

## 1. Out-of-Sample Walk-Forward Performance Summary (`results/walk_forward_summary.json`)

Evaluated across `1,772` contiguous out-of-sample trading days (`2020-01-01` through `2026-10-30`) spanning the 2020 COVID shock, 2021 expansion, 2022 inflation/rate drawdown (where both equities and long-duration Treasuries fell together), and 2023–2026 cycle. Every step $t$ enforces strict $\mathcal{F}_t$ filtration via `implementation/data/cutoff_validator.py`.

| Strategy Track | Gross Ann. Return % | Net Ann. Return % | Ann. Volatility % | **Net Sharpe** | **Sortino Ratio** | **Max Drawdown %** | **Calmar Ratio** | **Annualized Turnover** | Cost Drag (Ann.) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Equal-Weight Buy & Hold** | $11.08\%$ | $11.03\%$ | $11.56\%$ | $0.963$ | $1.360$ | $21.34\%$ | $0.517$ | $0.47\times$ | $0.05\%$ |
| **2. Uncalibrated Point Forecast** | $-1.18\%$ | $-12.52\%$ | $11.80\%$ | $-1.075$ | $-1.427$ | $68.24\%$ | $-0.184$ | $121.86\times$ | $12.19\%$ |
| **3. `TRACE-Fin` Causal Regime + 90% Conformal + Cost Gate** | **$11.20\%$** | **$10.74\%$** | **$8.92\%$** | **$1.189$** | **$1.680$** | **$11.76\%$** | **$0.913$** | **$4.19\times$** | **$0.42\%$** |
| **4. Diagnostic 1-Day Lookahead Leak Control** *(Contaminated)* | $1556.55\%$ | $1056.63\%$ | $16.08\%$ | $15.379$ | $495.977$ | $0.19\%$ | $5623.536$ | $363.07\times$ | $36.31\%$ |

### Key Quantitative Takeaways
1. **Superior Risk-Adjusted Return Over Passive Equal Weight (`+23.5%` Net Sharpe):** `TRACE-Fin` delivers a net annualized return of **$10.74\%$** (virtually matching Buy & Hold's $11.03\%$) at nearly **$23\%$ lower annualized volatility** ($8.92\%$ vs. $11.56\%$), lifting **Net Sharpe from $0.963$ to $1.189$** and **Sortino from $1.360$ to $1.680$**.
2. **Maximum Drawdown Cut Nearly in Half (`21.34% -> 11.76%`):** By scaling target allocations inversely to 90% Adaptive Conformal Interval widths ($w_{i,t} \propto 1 / W_{i,t}$) and smoothly dampening equity exposure when causal `High-Volatility` stress triggers, `TRACE-Fin` slashes worst peak-to-trough drawdown by **44.9%** (`21.34% -> 11.76%`), nearly doubling Calmar ratio (`0.517 -> 0.913`).
3. **Turnover & Transaction-Fee Discipline (`121.86x -> 4.19x`):** Uncalibrated daily point forecasting churns capital at $121.86\times$ annualized turnover ($12.19\%$/yr fee bleed against $10\text{ bps}$ one-way friction). Combining smoothed causal regime posteriors (`EMA`) with a $6.0\%$ weight-drift cost hurdle gate limits annualized turnover to **$4.19\times$** ($0.42\%$/yr cost drag).

---

## 2. Empirical 90% Conformal Coverage Across Regimes (`results/regime_coverage.csv`)

Evaluated across `8,860` pooled 1-day out-of-sample asset-day predictions (`1,772` days $\times$ `5` assets) partitioned strictly by the causal volatility state observed **before** the return realization (`Calm`: $N=5,014$; `High-Volatility`: $N=3,846$):

| Uncertainty Method | Causal Market Regime | Sample Size ($N$) | Nominal Target | **Empirical Coverage %** | Mean Interval Width % | **Winkler Interval Score** *(Lower Better)* |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`static_split_conformal`** | `Calm` | $5,014$ | $90.0\%$ | $94.10\%$ *(Over-wide)* | $3.635\%$ | $0.04221$ |
| **`static_split_conformal`** | `High-Volatility` | $3,846$ | $90.0\%$ | **$84.14\%$** *(Stress Breach)* | $3.688\%$ | $0.07264$ |
| **`static_split_conformal`** | **Overall (All Days)** | **$8,860$** | **$90.0\%$** | **$89.77\%$** | **$3.658\%$** | **$0.05542$** |
| **`trace_fin_regime_aci`** | `Calm` | $5,014$ | $90.0\%$ | $86.90\%$ | **$2.772\%$** | **$0.04131$** |
| **`trace_fin_regime_aci`** | `High-Volatility` | $3,846$ | $90.0\%$ | **$93.60\%$** *(Tail Protected)* | $6.025\%$ | **$0.06942$** |
| **`trace_fin_regime_aci`** | **Overall (All Days)** | **$8,860$** | **$90.0\%$** | **$89.81\%$** | **$4.184\%$** | **$0.05351$** |

### Uncertainty Calibration Findings
- **Exchangeability Failure Under Stress:** Although static split conformal appears calibrated marginally ($89.77\%$), it relies on constant unnormalized quantiles—leaving investors dangerously under-covered during market crashes (**$84.14\%$ coverage in High-Volatility stress**) while holding unnecessarily wide intervals in quiet markets ($3.635\%$).
- **Regime-Normalized ACI Restores Tail Validity & Beats Winkler Everywhere:** Scaling non-conformity residuals by causal stress volatility and updating miscoverage levels online ($\alpha_{t+1} = \alpha_t + \gamma (\alpha - \text{err}_t)$) narrows calm-market intervals to **$2.772\%$** while rapidly widening under stress to **$6.025\%$**, lifting stress coverage to **$93.60\%$** (`+9.46 pp`) and beating static conformal on proper Winkler Interval Score in **Calm** ($0.04131 < 0.04221$), **High-Volatility** ($0.06942 < 0.07264$), and **Overall** ($0.05351 < 0.05542$).

---

## 3. Point-in-Time Filtration & Lookahead Contamination Audit

| Pipeline Track | Filtration Constraint | Steps Verified | Causal Violations | Leaked Steps Flagged | Cryptographic SHA-256 Filtration Seal |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`TRACE-Fin` Causal (`Tracks 1–3`)** | Strictly $\mathcal{F}_t$ via `validate_available_at` | `1,772 / 1,772` | **`0` (`0.00%`)** | N/A | `99d4ba19f95516acb40ba5b01901b337...` |
| **Diagnostic Leak Control (`Track 4`)** | Deliberate $\mathcal{F}_{t+1}$ 1-day lookahead | `1,772 / 1,772` | Intentional | **`1,772 / 1,772` (`100.0%`)** | Audited via `audit_lookahead_violations` |

---

## 4. Honest Engineering Postmortem: Observed Bottlenecks & Failure Regimes

1. **ACI Coverage Hysteresis During Calm-to-Crash Transitions (`86.9%` Calm Coverage):** Because Gibbs–Candès ACI integrates trailing realized miscoverage over sliding lookbacks, after a multi-month crisis (`High-Volatility`, $93.60\%$ coverage) rapidly subsides into a quiet bull leg, $\alpha_t$ takes several weeks to adapt upward and intervals compress rapidly—temporarily shaving Calm empirical coverage slightly below $90\%$ (`86.90%`).
2. **2022 Stock–Bond Correlation Breakdown (`SPY` + `TLT` Concurrent Drawdown):** Traditional fixed-income hedging assumes negative equity–bond correlation. During 2022 inflation tightening, `SPY`, `QQQ`, and `TLT` all suffered simultaneous multi-quarter negative trends. Trend-filtered defensive rotation (`GLD` + cash buffering) prevented `TRACE-Fin` from buying falling Treasuries solely due to equity volatility spikes, yet initial early-2022 transition lag still drove most of the observed `11.76%` strategy drawdown.
3. **Rapid V-Shaped Rebound Under-Participation:** Immediately following sharp liquidation bottoms (e.g., April 2020 and late-2022 lows), trailing 21d volatility and conformal interval widths $W_{i,t}$ remain elevated. Because sizing scales inversely with $W_{i,t}$, gross equity exposure ramps back up gradually over 3–4 weeks rather than snapping instantly to 100% exposure at the bottom.
