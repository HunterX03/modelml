"""
Dashboard inference helper — drop this module into your Streamlit project
(or just import from it) to use the trained bundle end-to-end.

It exposes:

    load_bundle()            -> dict with every loaded artefact
    infer_for_bar(features)  -> {'win_probability', 'top_strategies', 'regime'}
    apply_ml_veto(df_signals, threshold) -> filtered signals DataFrame

Example Streamlit usage:

    import streamlit as st
    from dashboard_inference import load_bundle, infer_for_bar

    bundle = load_bundle()
    st.subheader("Today's recommendation")
    row = get_today_features_for_symbol("AARTIDRUGS")   # <- your data
    out = infer_for_bar(row, bundle=bundle)
    st.metric("Win probability", f"{out['win_probability']*100:.1f}%")
    for name, prob in out['top_strategies']:
        st.write(f"• **{name}** — {prob*100:.1f}%")
"""
import json
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
BUNDLE_PATH = os.path.join(MODEL_DIR, "bundle.json")


@lru_cache(maxsize=1)
def load_bundle() -> dict:
    """Load every model artefact once and cache."""
    with open(BUNDLE_PATH) as f:
        manifest = json.load(f)

    bundle = {"manifest": manifest}
    bundle["quality_clf"] = joblib.load(os.path.join(BASE_DIR, manifest["quality_clf"]))
    bundle["quality_scaler"] = joblib.load(os.path.join(BASE_DIR, manifest["quality_scaler"]))
    bundle["features"] = joblib.load(os.path.join(BASE_DIR, manifest["features"]))
    if manifest.get("selector"):
        bundle["selector"] = joblib.load(os.path.join(BASE_DIR, manifest["selector"]))
        bundle["sel_scaler"] = joblib.load(os.path.join(BASE_DIR, manifest["sel_scaler"]))
        bundle["label_encoder"] = joblib.load(os.path.join(BASE_DIR, manifest["label_encoder"]))
    if manifest.get("regime_hmm"):
        bundle["regime_hmm"] = joblib.load(os.path.join(BASE_DIR, manifest["regime_hmm"]))
        bundle["regime_scaler"] = joblib.load(os.path.join(BASE_DIR, manifest["regime_scaler"]))
        bundle["regime_names"] = joblib.load(os.path.join(BASE_DIR, manifest["regime_names"]))
    return bundle


def infer_for_bar(feature_row: dict, bundle: dict | None = None) -> dict:
    """
    Run full inference on a single row of indicator features.

    `feature_row` must contain every column listed in bundle['features'].
    Missing columns are filled with 0 (you should avoid this).
    """
    if bundle is None:
        bundle = load_bundle()
    feats = bundle["features"]

    x = np.array([[float(feature_row.get(c, 0)) for c in feats]], dtype=float)

    # Win-probability
    p_win = float(bundle["quality_clf"].predict_proba(bundle["quality_scaler"].transform(x))[0, 1])

    top = []
    if "selector" in bundle:
        probs = bundle["selector"].predict_proba(bundle["sel_scaler"].transform(x))[0]
        order = np.argsort(probs)[::-1][:3]
        top = [(bundle["label_encoder"].inverse_transform([int(i)])[0], float(probs[i])) for i in order]

    return {
        "win_probability": round(p_win, 3),
        "top_strategies": top,
        "recommended_threshold": bundle["manifest"].get("ml_veto", {}).get("recommended_threshold", 0.55),
        "ml_filter_pass": bool(p_win >= bundle["manifest"].get("ml_veto", {}).get("recommended_threshold", 0.55)),
    }


def apply_ml_veto(signals_df: pd.DataFrame, feature_cols: list | None = None,
                  threshold: float | None = None, bundle: dict | None = None) -> pd.DataFrame:
    """
    Filter a DataFrame of candidate signals through the quality classifier.

    `signals_df` must contain the feature columns.
    Returns a copy with two added cols: `prob_win`, `ml_pass` and only
    the rows where `ml_pass == True`.
    """
    if bundle is None:
        bundle = load_bundle()
    feats = feature_cols or bundle["features"]
    if threshold is None:
        threshold = bundle["manifest"].get("ml_veto", {}).get("recommended_threshold", 0.55)

    X = signals_df[feats].values.astype(float)
    probs = bundle["quality_clf"].predict_proba(bundle["quality_scaler"].transform(X))[:, 1]
    out = signals_df.copy()
    out["prob_win"] = probs
    out["ml_pass"] = probs >= threshold
    return out[out["ml_pass"]].copy()


if __name__ == "__main__":
    # Tiny demo — pick a random row from all_trades.csv and show inference
    bundle = load_bundle()
    feats = bundle["features"]
    print(f"Bundle version: {bundle['manifest']['pipeline_version']}")
    print(f"Loaded {len(feats)} features, {len(bundle['manifest']['strategies'])} strategies")
    print(f"Recommended ML-veto threshold: {bundle['manifest'].get('ml_veto', {}).get('recommended_threshold')}")
    print(f"Expected mean WR after veto: "
          f"{bundle['manifest'].get('ml_veto', {}).get('expected_mean_win_rate_after_veto', 0)*100:.1f}%")

    # Fake-bar example: all zeros (just to show the API works)
    demo = {c: 0.0 for c in feats}
    demo["rsi14"] = 55.0
    demo["adx"] = 25.0
    demo["ret_20"] = 0.05
    print("\nDemo inference (RSI=55, ADX=25, 20d return +5%):")
    print(json.dumps(infer_for_bar(demo, bundle), indent=2))
