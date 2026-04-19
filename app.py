"""
NSE Quant Dashboard — main page (🎯 Signal Inference).

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from dashboard_inference import load_bundle, infer_for_bar          # noqa: E402
from run_pipeline import add_indicators, ML_FEATS, fetch_ohlcv       # noqa: E402

# ─── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Quant Dashboard",
    page_icon="🇮🇳",
    layout="wide",
)

# ─── Bundle (cached across reruns) ─────────────────────────────────────
@st.cache_resource(show_spinner="Loading ML bundle ...")
def get_bundle():
    return load_bundle()


bundle = get_bundle()
THRESHOLD = bundle["manifest"]["ml_veto"]["recommended_threshold"]
EXPECTED_WR = bundle["manifest"]["ml_veto"]["expected_mean_win_rate_after_veto"]

# ─── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Model status")
    st.success(
        f"**Bundle v{bundle['manifest']['pipeline_version']}**\n\n"
        f"Trained: `{bundle['manifest']['built_at'][:10]}`\n\n"
        f"Recommended τ = **{THRESHOLD}**\n\n"
        f"Expected mean WR: **{EXPECTED_WR*100:.1f}%**"
    )
    st.markdown("---")
    st.markdown("### Navigation")
    st.caption(
        "• 🎯 Signal (this page)\n"
        "• 📊 Leaderboard\n"
        "• 🔬 ML Veto\n"
        "• 📡 Paper Trade Live\n"
        "• 🌊 Regime Monitor"
    )
    st.markdown("---")
    st.caption(
        f"Universe trained on: {len(bundle['manifest']['strategies'])} strategies"
    )

# ─── Header ────────────────────────────────────────────────────────────
st.title("🇮🇳 NSE Quant Dashboard")
st.caption(
    "Institutional-grade signal inference powered by a trained Setup-Quality "
    "classifier, Strategy Selector and HMM regime detector."
)

# ─── Helpers ───────────────────────────────────────────────────────────
@st.cache_data(ttl=900, show_spinner=False)
def features_for_symbol(symbol: str) -> dict | None:
    """Fetch 2y daily bars from yfinance and return the latest indicator row."""
    ticker = f"{symbol}.NS" if not symbol.startswith("^") else symbol
    df = yf.download(ticker, period="2y", interval="1d",
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    ind = add_indicators(df)
    if ind.empty:
        return None
    row = ind.iloc[-1].to_dict()
    row["_latest_close"] = float(df["close"].iloc[-1])
    row["_latest_date"] = df.index[-1].date().isoformat()
    return row

# ─── Market snapshot (Nifty regime) ────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def market_regime() -> tuple[str, float]:
    today = datetime.today().date()
    start = (today - timedelta(days=400)).isoformat()
    nifty = fetch_ohlcv("^NSEI", start, today.isoformat(), "1d")
    if nifty.empty:
        return "UNKNOWN", float("nan")
    ind = add_indicators(nifty).dropna()
    if ind.empty:
        return "UNKNOWN", float("nan")
    row = ind.iloc[-1]
    x = np.array([[row["ret_1"], row["vol_20"], row["adx"], row["atr_pct"]]])
    hmm = bundle["regime_hmm"]
    sc = bundle["regime_scaler"]
    names = bundle["regime_names"]
    state = int(hmm.predict(sc.transform(x))[0])
    return names.get(state, str(state)), float(row["close"])


col_a, col_b, col_c = st.columns(3)
try:
    regime, nifty_close = market_regime()
    col_a.metric("Nifty 50", f"₹{nifty_close:,.0f}" if not np.isnan(nifty_close) else "—")
    col_b.metric("Market regime", regime)
except Exception as e:
    col_b.metric("Market regime", "—")
    col_b.caption(f"({e})")
col_c.metric("ML-veto τ", f"{THRESHOLD}", f"+{EXPECTED_WR*100-50:.0f} pp WR")

st.markdown("---")

# ─── Signal inference UI ───────────────────────────────────────────────
st.subheader("🎯 Get a real-time recommendation")

with st.form("symbol_form"):
    c1, c2 = st.columns([3, 1])
    symbol = c1.text_input(
        "NSE symbol (without .NS suffix)",
        value="RELIANCE",
        help="Enter any Nifty-500 stock code — e.g., HDFCBANK, TCS, SBIN",
    )
    submit = c2.form_submit_button("🚀 Analyze", use_container_width=True)

if submit:
    with st.spinner(f"Fetching data & running inference for {symbol} ..."):
        feats = features_for_symbol(symbol.strip().upper())

    if feats is None:
        st.error(f"Could not fetch data for **{symbol}** — try another ticker.")
    else:
        out = infer_for_bar(feats, bundle)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Latest close", f"₹{feats['_latest_close']:,.2f}")
        m2.metric("As of", feats["_latest_date"])
        m3.metric("Win probability", f"{out['win_probability']*100:.1f}%",
                  delta=f"τ = {THRESHOLD}")
        m4.metric(
            "ML filter",
            "PASS ✅" if out["ml_filter_pass"] else "VETO ❌",
            delta="Trade" if out["ml_filter_pass"] else "Skip",
            delta_color="normal" if out["ml_filter_pass"] else "inverse",
        )

        if out["ml_filter_pass"]:
            st.success(
                f"**{symbol}** — the ML bundle approves this setup with "
                f"**{out['win_probability']*100:.1f} %** win probability. "
                f"Recommended strategies below."
            )
        else:
            st.warning(
                f"**{symbol}** — probability below threshold "
                f"({out['win_probability']*100:.1f} % < {THRESHOLD*100:.0f} %). "
                f"Listed strategies are indicative only."
            )

        st.subheader("Top 3 strategies for this setup")
        for name, prob in out["top_strategies"]:
            st.progress(prob, text=f"{name} — {prob*100:.1f}%")

        with st.expander("🔍 Feature snapshot (what the model saw)"):
            shown = {k: feats[k] for k in ML_FEATS if k in feats}
            st.json({k: round(float(v), 4) if isinstance(v, (int, float, np.floating))
                     else v for k, v in shown.items()})
else:
    st.info("👆 Enter a symbol above and click **Analyze** to see the ML verdict.")

# ─── Latest track record (public proof) ────────────────────────────────
TRACK = BASE_DIR / "reports" / "paper_track_record.csv"
if TRACK.exists():
    st.markdown("---")
    st.subheader("🏅 Paper-trade track record")
    tr = pd.read_csv(TRACK).tail(10)
    st.dataframe(tr, use_container_width=True, hide_index=True)
    st.caption("Grows every time the Paper Trade Live page is opened.")

st.markdown("---")
st.caption(
    "Disclaimer: For research and educational purposes only. Past performance "
    "does not guarantee future returns. Nothing here is financial advice."
)
