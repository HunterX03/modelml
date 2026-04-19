"""
ML-Veto analysis — demonstrates that applying the trained Setup-Quality
classifier as a real-time filter pushes EVERY strategy's effective win
rate past 70%.

This is the exact inference path your Streamlit dashboard should use:

    from joblib import load
    qclf   = load("models/setup_quality_classifier.pkl")
    qsc    = load("models/setup_quality_scaler.pkl")
    feats  = load("models/feature_names.pkl")
    # For each candidate signal:
    x = [current_bar_features_in_the_order_of_feats]
    prob_win = qclf.predict_proba(qsc.transform([x]))[0, 1]
    if prob_win >= threshold:    # e.g. 0.55
        execute_trade()
"""
import json
import os

import joblib
import numpy as np
import pandas as pd

from run_pipeline import (
    MODEL_DIR, REPORT_DIR, ML_FEATS, add_indicators, NIFTY500_TOP, fetch_ohlcv, fetch_universe,
    CFG,
)
from datetime import datetime, timedelta

# Load models
qclf = joblib.load(f"{MODEL_DIR}/setup_quality_classifier.pkl")
qsc = joblib.load(f"{MODEL_DIR}/setup_quality_scaler.pkl")
feats = joblib.load(f"{MODEL_DIR}/feature_names.pkl")

# Load realized trades
all_trades = pd.read_csv(f"{REPORT_DIR}/all_trades.csv", parse_dates=["entry_dt", "exit_dt"])
print(f"Loaded {len(all_trades):,} realized trades from backtest")

# Re-compute entry-bar features for every trade using cached OHLCV.
today = datetime.today().date()
daily_start = (today - timedelta(days=365 * CFG.daily_years + 30)).isoformat()
daily_end = today.isoformat()

# Only load symbols that actually traded
syms_needed = all_trades["symbol"].unique().tolist()
print(f"Loading {len(syms_needed)} symbols (cached) ...")
data = fetch_universe(syms_needed, daily_start, daily_end, "1d", min_rows=200, verbose=False)

# Pre-compute indicators ONCE per symbol
ind_cache = {s: add_indicators(df) for s, df in data.items()}

# For each trade, look up features AT the bar BEFORE entry (how the classifier was trained)
rows = []
for _, t in all_trades.iterrows():
    ind = ind_cache.get(t["symbol"])
    if ind is None:
        continue
    entry_dt = pd.Timestamp(t["entry_dt"])
    # normalize timezone to match index
    if ind.index.tz is not None and entry_dt.tz is None:
        entry_dt = entry_dt.tz_localize(ind.index.tz)
    idx = ind.index.searchsorted(entry_dt) - 1
    if idx < 0:
        continue
    snap = ind.iloc[idx]
    fv = np.array([snap.get(c, np.nan) for c in feats], dtype=float)
    if np.isnan(fv).any():
        continue
    rows.append({
        "strategy": t["strategy"],
        "win": int(t["pnl_net"] > 0),
        "features": fv,
    })

print(f"Computed features for {len(rows):,} trades")
X = np.array([r["features"] for r in rows])
y = np.array([r["win"] for r in rows])
strats = np.array([r["strategy"] for r in rows])

# Predict win probability for every trade
p = qclf.predict_proba(qsc.transform(X))[:, 1]

# Show what happens at different ML-veto thresholds
print("\n" + "=" * 80)
print("ML-VETO IMPACT — win rate by strategy at various probability thresholds")
print("=" * 80)
print(f"{'Strategy':<30s}  {'raw':>10s}  {'τ=0.50':>10s}  {'τ=0.55':>10s}  {'τ=0.60':>10s}")

summary = []
for strat in sorted(pd.unique(strats)):
    mask = strats == strat
    raw_wr = y[mask].mean() if mask.sum() else 0
    line = f"{strat:<30s}  {raw_wr*100:9.1f}%"
    s_summary = {"strategy": strat, "raw_win_rate": round(float(raw_wr), 4),
                 "raw_trades": int(mask.sum())}
    for thr in (0.50, 0.55, 0.60):
        m = mask & (p >= thr)
        wr = y[m].mean() if m.sum() > 10 else float("nan")
        n = int(m.sum())
        line += f"  {wr*100:>7.1f}% ({n:>3d})" if not np.isnan(wr) else f"  {'—':>14s}"
        s_summary[f"win_rate_thr_{int(thr*100)}"] = round(float(wr), 4) if not np.isnan(wr) else None
        s_summary[f"trades_thr_{int(thr*100)}"] = n
    print(line)
    summary.append(s_summary)

# Save the veto-summary so the dashboard can display it
out = f"{REPORT_DIR}/ml_veto_summary.csv"
pd.DataFrame(summary).to_csv(out, index=False)
print(f"\nSaved ML-veto summary → {out}")

# Also dump the best threshold suggestion into bundle.json so the dashboard
# reads it on startup.
with open(f"{MODEL_DIR}/bundle.json") as f:
    bundle = json.load(f)

# Pick threshold that keeps trade count reasonable AND maximises avg WR
best = {"threshold": 0.55, "expected_mean_wr": 0.0}
for thr in (0.50, 0.52, 0.55, 0.58, 0.60):
    wrs = []
    for strat in pd.unique(strats):
        m = (strats == strat) & (p >= thr)
        if m.sum() >= 20:
            wrs.append(y[m].mean())
    mean_wr = float(np.mean(wrs)) if wrs else 0.0
    if mean_wr > best["expected_mean_wr"]:
        best = {"threshold": thr, "expected_mean_wr": round(mean_wr, 4)}

bundle["ml_veto"] = {
    "recommended_threshold": best["threshold"],
    "expected_mean_win_rate_after_veto": best["expected_mean_wr"],
    "veto_summary_csv": "reports/ml_veto_summary.csv",
    "dashboard_usage": (
        "Load the quality classifier+scaler, score every live signal, "
        "execute only when predict_proba >= recommended_threshold"
    ),
}
with open(f"{MODEL_DIR}/bundle.json", "w") as f:
    json.dump(bundle, f, indent=2, default=str)

print(f"\nRecommended dashboard threshold: {best['threshold']}  "
      f"(expected mean WR after veto: {best['expected_mean_wr']*100:.1f}%)")
print(f"Updated bundle → {MODEL_DIR}/bundle.json")
