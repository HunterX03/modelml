"""
Monthly walk-forward retrainer — keeps the ML bundle fresh as market
regimes drift.

Usage
-----
One-off:
    python walk_forward_retrainer.py

Cron (1st of every month, 03:00 AM):
    0 3 1 * *  cd /path/to/note_repo && /usr/bin/python3 walk_forward_retrainer.py >> logs/retrainer.log 2>&1

What it does
------------
1. Archives the current /models and /reports under /archive/<YYYY-MM-DD>/
   so you can always roll back.
2. Re-runs the full pipeline on the newest 3-year window + 2-year forward.
3. Validates the new bundle on a held-out 6-month window (walk-forward).
4. Only promotes the new bundle if its out-of-sample mean win-rate
   (after ML-veto at τ=0.60) is ≥ 70%; otherwise keeps the old bundle
   and logs a drift warning.
5. Writes a summary to reports/retrain_history.jsonl so the dashboard
   can display model freshness.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
ARCHIVE_DIR = os.path.join(BASE_DIR, "archive")
LOG_DIR = os.path.join(BASE_DIR, "logs")
HISTORY_FILE = os.path.join(REPORT_DIR, "retrain_history.jsonl")

MIN_ACCEPTABLE_WR = 0.70          # minimum out-of-sample mean WR to promote
OOS_HOLDOUT_MONTHS = 6            # last 6 months held out for walk-forward test
ML_VETO_THRESHOLD = 0.60

for d in (ARCHIVE_DIR, LOG_DIR):
    os.makedirs(d, exist_ok=True)


def log(msg: str) -> None:
    ts = datetime.now().isoformat(timespec="seconds")
    print(f"[{ts}] {msg}", flush=True)


def archive_current_bundle() -> str:
    """Snapshot existing bundle so we can rollback if new one under-performs."""
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    target = os.path.join(ARCHIVE_DIR, stamp)
    os.makedirs(target, exist_ok=True)
    for src in (MODEL_DIR, REPORT_DIR):
        if os.path.exists(src):
            dest = os.path.join(target, os.path.basename(src))
            shutil.copytree(src, dest, dirs_exist_ok=True)
    log(f"Archived current bundle → {target}")
    return target


def run_pipeline() -> bool:
    """Re-run the full backtest + ML training pipeline."""
    log("Running run_pipeline.py (may take 7-30 min) ...")
    proc = subprocess.run(
        [sys.executable, os.path.join(BASE_DIR, "run_pipeline.py")],
        capture_output=True, text=True, cwd=BASE_DIR,
    )
    ok = (proc.returncode == 0)
    log(f"Pipeline finished with returncode={proc.returncode}")
    if not ok:
        log("---STDERR tail---")
        log(proc.stderr[-2000:])
    return ok


def evaluate_holdout() -> dict:
    """
    Walk-forward validation: score the trained quality classifier on the last
    OOS_HOLDOUT_MONTHS of trades and compute mean win-rate at the veto
    threshold.
    """
    qclf = joblib.load(os.path.join(MODEL_DIR, "setup_quality_classifier.pkl"))
    qsc = joblib.load(os.path.join(MODEL_DIR, "setup_quality_scaler.pkl"))
    feats = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))

    trades = pd.read_csv(
        os.path.join(REPORT_DIR, "all_trades.csv"), parse_dates=["entry_dt", "exit_dt"]
    )
    cutoff = trades["entry_dt"].max() - pd.DateOffset(months=OOS_HOLDOUT_MONTHS)
    oos = trades[trades["entry_dt"] >= cutoff].copy()
    log(f"OOS window: {cutoff.date()} → {trades['entry_dt'].max().date()}  ({len(oos):,} trades)")

    # Re-derive features for these trades using cached OHLCV + indicator lib.
    # We reuse the helper already used in ml_veto_analysis.py; import lazily.
    from run_pipeline import add_indicators, fetch_universe, CFG
    today = datetime.today().date()
    start = (today - timedelta(days=365 * CFG.daily_years + 30)).isoformat()
    syms = oos["symbol"].unique().tolist()
    data = fetch_universe(syms, start, today.isoformat(), "1d", min_rows=200, verbose=False)
    ind_cache = {s: add_indicators(df) for s, df in data.items()}

    rows = []
    for _, t in oos.iterrows():
        ind = ind_cache.get(t["symbol"])
        if ind is None:
            continue
        idx = ind.index.searchsorted(pd.Timestamp(t["entry_dt"])) - 1
        if idx < 0:
            continue
        snap = ind.iloc[idx]
        fv = np.array([snap.get(c, np.nan) for c in feats], dtype=float)
        if np.isnan(fv).any():
            continue
        rows.append({"strategy": t["strategy"], "win": int(t["pnl_net"] > 0), "fv": fv})

    if not rows:
        return {"mean_wr": 0.0, "strategies": {}, "n_trades": 0}

    X = np.array([r["fv"] for r in rows])
    y = np.array([r["win"] for r in rows])
    strats = np.array([r["strategy"] for r in rows])
    probs = qclf.predict_proba(qsc.transform(X))[:, 1]

    per = {}
    wrs = []
    for strat in sorted(pd.unique(strats)):
        mask = (strats == strat) & (probs >= ML_VETO_THRESHOLD)
        if mask.sum() >= 5:
            wr = float(y[mask].mean())
            per[strat] = {"wr": round(wr, 4), "n": int(mask.sum())}
            wrs.append(wr)
    mean_wr = float(np.mean(wrs)) if wrs else 0.0
    return {"mean_wr": round(mean_wr, 4), "strategies": per, "n_trades": int(len(rows))}


def promote_or_rollback(archive_path: str, eval_result: dict) -> bool:
    """If the new bundle under-performs, restore from archive."""
    mean_wr = eval_result["mean_wr"]
    promoted = mean_wr >= MIN_ACCEPTABLE_WR

    if promoted:
        log(f"✅ Promoted new bundle (OOS mean WR {mean_wr*100:.1f}% ≥ {MIN_ACCEPTABLE_WR*100:.0f}%)")
    else:
        log(f"⚠ Rolling back — OOS mean WR {mean_wr*100:.1f}% < {MIN_ACCEPTABLE_WR*100:.0f}%")
        for sub in ("models", "reports"):
            src = os.path.join(archive_path, sub)
            dst = os.path.join(BASE_DIR, sub)
            if os.path.exists(src):
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
        log("Rolled back to previous bundle.")
    return promoted


def append_history(promoted: bool, archive_path: str, eval_result: dict) -> None:
    record = {
        "retrained_at": datetime.now().isoformat(timespec="seconds"),
        "promoted": bool(promoted),
        "archive_path": archive_path,
        "oos_mean_win_rate": eval_result["mean_wr"],
        "oos_trades": eval_result["n_trades"],
        "oos_per_strategy": eval_result["strategies"],
        "ml_veto_threshold": ML_VETO_THRESHOLD,
    }
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    log(f"History appended → {HISTORY_FILE}")


def main() -> int:
    log("=" * 60)
    log("Walk-forward retraining started")
    log("=" * 60)

    archive_path = archive_current_bundle()

    if not run_pipeline():
        log("Pipeline failed — restoring archive")
        for sub in ("models", "reports"):
            src = os.path.join(archive_path, sub)
            dst = os.path.join(BASE_DIR, sub)
            if os.path.exists(src):
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(src, dst)
        return 1

    result = evaluate_holdout()
    log(f"OOS mean WR @ τ={ML_VETO_THRESHOLD}: {result['mean_wr']*100:.1f}% "
        f"across {len(result['strategies'])} strategies")

    promoted = promote_or_rollback(archive_path, result)
    append_history(promoted, archive_path, result)
    log("Done.")
    return 0 if promoted else 2


if __name__ == "__main__":
    sys.exit(main())
