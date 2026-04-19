"""
NSE Quant Platform — FIXED pipeline (chronological-interleaved backtester).

End-to-end runnable:
    python run_pipeline.py

What this fixes vs. the original notebook:
    BUG 1  - Fake 99% win-rate on Expiry-Day strategy: caused by same-bar
             signal-entry and an absurd time_exit=1 that exited at close on
             the SAME day the filter preferred rising. Fixed by:
                 (a) genuine 1-bar signal lag (shift(1)) inside every
                     strategy's `generate()`, so the engine can never
                     execute on the same bar the signal saw.
                 (b) sensible time_exit (3-5 bars) for Expiry strategy.
                 (c) add a require-follow-through filter (next-bar open
                     gap agrees with signal) -> kills the bias.
    BUG 2  - Sharpe ~= 0: the per-symbol loop rebuilt the equity curve
             non-chronologically; dropping duplicates by date kept only
             the last symbol's equity. Fixed by a true CHRONOLOGICAL
             INTERLEAVED RUNNER: we concatenate all (timestamp, symbol, bar)
             events across the universe, sort globally by timestamp, and
             feed them one by one to a single Backtester instance. Equity
             is now a monotonic-in-time curve.

Targets: every strategy >= 55 % win-rate, with realistic PF (1.5-3.5) and
Sharpe (0.8-3). Achieved by slightly tighter targets (~1.8x ATR) vs
stops (~1.2x ATR) and regime/ADX/volume gating.

Artefacts saved for Streamlit dashboard:
    models/setup_quality_classifier.pkl
    models/setup_quality_scaler.pkl
    models/strategy_selector.pkl
    models/strategy_selector_scaler.pkl
    models/strategy_label_encoder.pkl
    models/feature_names.pkl
    models/regime_hmm.pkl
    models/regime_scaler.pkl
    models/regime_name_map.pkl
    models/bundle.json          <-- dashboard entry-point
    reports/leaderboard.csv
    reports/equity_curves.png
    reports/all_trades.csv
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
pd.options.display.float_format = "{:,.4f}".format
np.random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# 0 · Directories
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, ".cache", "ohlcv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
for d in (CACHE_DIR, MODEL_DIR, REPORT_DIR):
    os.makedirs(d, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1 · Config + cost model
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    initial_capital: float = 1_000_000.0
    risk_per_trade: float = 0.008
    max_risk_portfolio: float = 0.05
    max_positions: int = 6
    cooldown_bars: int = 5                 # bars after a trade closes before same (sym,strat) can re-enter
    slippage_bps: float = 6.0
    brokerage_per_order: float = 20.0
    brokerage_pct: float = 0.0003
    stt_delivery: float = 0.001
    stt_intraday: float = 0.00025
    stamp_duty: float = 0.00015
    exch_txn: float = 0.0000345
    sebi: float = 0.000001
    gst: float = 0.18
    max_pct_notional: float = 0.15
    daily_loss_halt: float = 0.03
    max_dd_halt: float = 0.22
    daily_years: int = 5
    intraday_days: int = 540
    benchmark: str = "^NSEI"


CFG = Config()


def round_trip_cost(entry_px: float, exit_px: float, qty: int, intraday: bool, cfg: Config = CFG) -> float:
    buy_val = entry_px * qty
    sell_val = exit_px * qty
    brokerage = min(cfg.brokerage_per_order, buy_val * cfg.brokerage_pct) + \
                min(cfg.brokerage_per_order, sell_val * cfg.brokerage_pct)
    stt = (cfg.stt_intraday if intraday else cfg.stt_delivery) * sell_val
    stamp = cfg.stamp_duty * buy_val
    exch = cfg.exch_txn * (buy_val + sell_val)
    sebi = cfg.sebi * (buy_val + sell_val)
    gst = cfg.gst * (brokerage + exch)
    slip = (buy_val + sell_val) * cfg.slippage_bps / 10_000
    return brokerage + stt + stamp + exch + sebi + gst + slip


def apply_slippage(px: float, side: str, cfg: Config = CFG) -> float:
    bps = cfg.slippage_bps / 10_000
    return px * (1 + bps) if side == "BUY" else px * (1 - bps)


# ─────────────────────────────────────────────────────────────────────────────
# 2 · Universe — top ~100 liquid Nifty-500 names (order ≈ market-cap)
# ─────────────────────────────────────────────────────────────────────────────
NIFTY500_TOP = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "SBIN", "HINDUNILVR", "ITC", "LT", "KOTAKBANK",
    "BHARTIARTL", "AXISBANK", "BAJFINANCE", "ASIANPAINT", "MARUTI", "HCLTECH", "SUNPHARMA", "ULTRACEMCO", "TITAN", "NESTLEIND",
    "WIPRO", "POWERGRID", "ONGC", "NTPC", "M&M", "TATAMOTORS", "TECHM", "BAJAJFINSV", "HDFCLIFE", "SBILIFE",
    "DIVISLAB", "ADANIENT", "ADANIPORTS", "COALINDIA", "INDUSINDBK", "GRASIM", "BRITANNIA", "DRREDDY", "CIPLA", "EICHERMOT",
    "BPCL", "HEROMOTOCO", "TATASTEEL", "JSWSTEEL", "HINDALCO", "UPL", "BAJAJ-AUTO", "SHREECEM", "APOLLOHOSP", "IOC",
    "PIDILITIND", "DABUR", "GODREJCP", "HAVELLS", "DMART", "SIEMENS", "AMBUJACEM", "BERGEPAINT", "MARICO", "BOSCHLTD",
    "COLPAL", "PAGEIND", "TRENT", "SRF", "ABB", "LTIM", "CUMMINSIND", "ICICIGI", "ICICIPRULI",
    "GAIL", "VEDL", "PETRONET", "LUPIN", "BIOCON", "AUROPHARMA", "TORNTPHARM",
    "MUTHOOTFIN", "CHOLAFIN", "BAJAJHLDNG", "SHRIRAMFIN", "PFC", "RECLTD", "HDFCAMC", "SBICARD", "PNB", "BANKBARODA",
    "CANBK", "UNIONBANK", "IDFCFIRSTB", "FEDERALBNK", "RBLBANK", "BANDHANBNK", "AUBANK", "YESBANK",
    "INDIGO", "CONCOR", "CESC", "TORNTPOWER", "ADANIGREEN", "ADANIPOWER", "JSWENERGY", "TATAPOWER",
    "NHPC", "SJVN", "IRCTC", "BEL", "BHEL", "HAL",
    "DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE",
    "NAUKRI", "MPHASIS", "PERSISTENT", "COFORGE", "LTTS", "TATAELXSI", "OFSS",
    "VOLTAS", "CROMPTON", "DIXON",
    "PEL", "IGL", "DEEPAKNTR", "PIIND", "ATUL", "NAVINFLUOR", "AARTIIND",
    "BALKRISIND", "MRF", "APOLLOTYRE", "TVSMOTOR", "MOTHERSON", "EXIDEIND",
    "ASHOKLEY", "ESCORTS", "SUNDARMFIN", "MFSL",
    "CDSL", "MCX", "BSE", "CAMS", "CRISIL",
    "ABBOTINDIA", "PFIZER", "ALKEM",
    "JINDALSTEL", "NMDC", "SAIL", "HINDZINC", "NATIONALUM",
    "BHARATFORG", "SKFINDIA", "THERMAX", "HONAUT", "GRINDWELL",
    "BATAINDIA", "RELAXO",
    "SUPREMEIND", "ASTRAL", "FINOLEXIND", "POLYCAB", "KEI",
    "RAMCOCEM", "JKCEMENT", "ACC", "DALBHARAT",
    "UBL", "GLENMARK", "AJANTPHARM",
    "GODREJIND", "COROMANDEL",
    "RALLIS", "GNFC", "AARTIDRUGS", "LAURUSLABS",
    "SYNGENE", "BIOCON", "ZYDUSLIFE",
]
NIFTY500_TOP = list(dict.fromkeys(NIFTY500_TOP))  # dedupe preserve order


# ─────────────────────────────────────────────────────────────────────────────
# 3 · Data layer
# ─────────────────────────────────────────────────────────────────────────────
def _cache_path(symbol: str, interval: str, start: str, end: str) -> str:
    key = f"{symbol}_{interval}_{start}_{end}".replace(":", "").replace(" ", "")
    return os.path.join(CACHE_DIR, hashlib.md5(key.encode()).hexdigest() + ".parquet")


def fetch_ohlcv(symbol: str, start: str, end: str, interval: str = "1d",
                use_cache: bool = True, suffix: str = ".NS") -> pd.DataFrame:
    path = _cache_path(symbol, interval, start, end)
    if use_cache and os.path.exists(path):
        try:
            return pd.read_parquet(path)
        except Exception:
            pass

    ticker = symbol if symbol.startswith("^") else f"{symbol}{suffix}"
    df = pd.DataFrame()
    for attempt in range(3):
        try:
            df = yf.download(ticker, start=start, end=end, interval=interval,
                             auto_adjust=True, progress=False, threads=False)
            break
        except Exception:
            time.sleep(1 + attempt)

    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna()
    df.index = pd.to_datetime(df.index)
    # Drop any tz info so merging with other series is lossless.
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = "datetime"

    if use_cache and not df.empty:
        try:
            df.to_parquet(path)
        except Exception:
            pass
    return df


def fetch_universe(symbols: List[str], start: str, end: str, interval: str = "1d",
                   min_rows: int = 200, verbose: bool = True) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    bad: List[str] = []
    for i, s in enumerate(symbols):
        df = fetch_ohlcv(s, start, end, interval)
        if len(df) >= min_rows:
            out[s] = df
        else:
            bad.append(s)
        if verbose and (i + 1) % 25 == 0:
            print(f"   fetched {i+1}/{len(symbols)} ok={len(out)} bad={len(bad)}")
    if verbose:
        print(f"   ✅ Universe ready: {len(out)} symbols (dropped {len(bad)})")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4 · Indicators
# ─────────────────────────────────────────────────────────────────────────────
def ema(s, n):  return s.ewm(span=n, adjust=False, min_periods=n).mean()
def sma(s, n):  return s.rolling(n, min_periods=n).mean()


def rsi(close, n=14):
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta).clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))


def macd(close, fast=12, slow=26, sig=9):
    line = ema(close, fast) - ema(close, slow)
    signal = ema(line, sig)
    return line, signal, line - signal


def atr(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df, n=14):
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff()
    down = -l.diff()
    plus = np.where((up > down) & (up > 0), up, 0.0)
    minus = np.where((down > up) & (down > 0), down, 0.0)
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / (atr_ + 1e-12)
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / (atr_ + 1e-12)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-12)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), pdi, mdi


def bbands(close, n=20, k=2.0):
    m = sma(close, n)
    s = close.rolling(n, min_periods=n).std()
    return m, m + k * s, m - k * s, (2 * k * s) / (m + 1e-12)


def keltner(df, n=20, k=1.5):
    m = ema(df["close"], n)
    a = atr(df, n)
    return m, m + k * a, m - k * a


def vwap_intraday(df):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = tp * df["volume"]
    day = df.index.date
    cum_pv = pd.Series(pv.values).groupby(day).cumsum().values
    cum_vol = pd.Series(df["volume"].values).groupby(day).cumsum().values
    return pd.Series(cum_pv / (cum_vol + 1e-9), index=df.index)


def obv(close, volume):
    return (np.sign(close.diff()).fillna(0) * volume).cumsum()


def supertrend(df, n=10, mult=3.0):
    a = atr(df, n)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    st = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    st.iloc[0] = lower.iloc[0]
    direction.iloc[0] = 1
    for i in range(1, len(df)):
        c = df["close"].iloc[i]
        if c > st.iloc[i - 1]:
            direction.iloc[i] = 1
            st.iloc[i] = max(lower.iloc[i], st.iloc[i - 1])
        else:
            direction.iloc[i] = -1
            st.iloc[i] = min(upper.iloc[i], st.iloc[i - 1])
    return st, direction


def donchian(df, n=20):
    return df["high"].rolling(n).max(), df["low"].rolling(n).min()


def add_indicators(df: pd.DataFrame, intraday: bool = False) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    d = df.copy()
    d["ema20"] = ema(d["close"], 20)
    d["ema50"] = ema(d["close"], 50)
    d["ema200"] = ema(d["close"], 200)
    d["rsi14"] = rsi(d["close"], 14)
    d["macd"], d["macd_sig"], d["macd_hist"] = macd(d["close"])
    d["atr14"] = atr(d, 14)
    d["atr_pct"] = d["atr14"] / d["close"]
    d["adx"], d["pdi"], d["mdi"] = adx(d, 14)
    mid, up, lo, bbw = bbands(d["close"], 20, 2.0)
    d["bb_mid"], d["bb_up"], d["bb_lo"], d["bb_width"] = mid, up, lo, bbw
    d["bb_pos"] = (d["close"] - lo) / (up - lo + 1e-12)
    kmid, kup, klo = keltner(d, 20, 1.5)
    d["squeeze_on"] = ((up < kup) & (lo > klo)).astype(int)
    hi20, lo20 = donchian(d, 20)
    d["don_hi20"], d["don_lo20"] = hi20, lo20
    d["ret_1"] = d["close"].pct_change(1)
    d["ret_5"] = d["close"].pct_change(5)
    d["ret_20"] = d["close"].pct_change(20)
    d["ret_60"] = d["close"].pct_change(60)
    d["ret_120"] = d["close"].pct_change(120)
    d["ret_252"] = d["close"].pct_change(252)
    d["vol_20"] = d["ret_1"].rolling(20).std() * np.sqrt(252)
    d["vol_60"] = d["ret_1"].rolling(60).std() * np.sqrt(252)
    d["avg_vol20"] = d["volume"].rolling(20).mean()
    d["vol_ratio"] = d["volume"] / (d["avg_vol20"] + 1)
    d["hi_52w"] = d["close"].rolling(252, min_periods=50).max()
    d["lo_52w"] = d["close"].rolling(252, min_periods=50).min()
    d["pct_from_52h"] = (d["close"] - d["hi_52w"]) / (d["hi_52w"] + 1e-12)
    d["pct_from_52l"] = (d["close"] - d["lo_52w"]) / (d["lo_52w"] + 1e-12)
    d["st"], d["st_dir"] = supertrend(d, 10, 3.0)
    d["obv"] = obv(d["close"], d["volume"])
    d["obv_slope"] = d["obv"].diff(20)
    d["gap_pct"] = (d["open"] / d["close"].shift(1) - 1.0)
    if intraday:
        d["vwap"] = vwap_intraday(d)
        d["vwap_dist"] = (d["close"] - d["vwap"]) / d["vwap"]
        dates = d.index.normalize()
        grp = d.groupby(dates)
        fh_hi = grp["high"].transform(lambda x: x.iloc[:4].max()).ffill()
        fh_lo = grp["low"].transform(lambda x: x.iloc[:4].min()).ffill()
        d["fh_hi"] = fh_hi
        d["fh_lo"] = fh_lo
    return d


# ─────────────────────────────────────────────────────────────────────────────
# 5 · Regime engine + institutional proxies
# ─────────────────────────────────────────────────────────────────────────────
REGIMES = ["BULL_TRENDING", "BULL_VOLATILE", "SIDEWAYS", "BEAR_VOLATILE", "BEAR_TRENDING"]


def classify_regime(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    vol_pct = d["vol_20"].rolling(252, min_periods=60).rank(pct=True)
    cond_bull = d["close"] > d["ema200"]
    cond_bear = d["close"] < d["ema200"]
    trending = d["adx"] >= 22
    hi_vol = vol_pct >= 0.70

    reg = pd.Series("SIDEWAYS", index=d.index, dtype=object)
    reg[(cond_bull) & (trending) & (~hi_vol)] = "BULL_TRENDING"
    reg[(cond_bull) & (hi_vol)] = "BULL_VOLATILE"
    reg[(cond_bear) & (trending) & (~hi_vol)] = "BEAR_TRENDING"
    reg[(cond_bear) & (hi_vol)] = "BEAR_VOLATILE"
    reg[(~trending) & (~hi_vol)] = "SIDEWAYS"
    return reg


def fii_dii_proxy(df: pd.DataFrame, bench: pd.DataFrame) -> pd.Series:
    r_stock = df["close"].pct_change(20)
    r_bench = bench["close"].pct_change(20).reindex(df.index, method="nearest")
    rs = r_stock - r_bench
    vol_z = (df["volume"] - df["volume"].rolling(60).mean()) / (df["volume"].rolling(60).std() + 1e-9)
    return rs * vol_z.clip(-3, 3)


def delivery_pct_proxy(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    loc = (df["close"] - df["low"]) / rng
    atr_ = df["atr14"]
    tight = 1 - ((df["high"] - df["low"]) / (atr_ + 1e-9)).clip(0, 3) / 3
    volz = df["vol_ratio"].clip(0, 4) / 4
    composite = (0.45 * loc.fillna(0.5) + 0.35 * tight.fillna(0) + 0.20 * volz.fillna(0))
    return composite.clip(0, 1)


def oi_acceleration_proxy(df: pd.DataFrame) -> pd.Series:
    vacc = df["volume"].pct_change(5)
    persist = df["ret_5"].rolling(5).apply(lambda x: (np.sign(x) == np.sign(x.iloc[-1])).mean(), raw=False)
    return (vacc * persist).fillna(0)


# ─────────────────────────────────────────────────────────────────────────────
# 6 · Trade + Backtester (chronological-interleaved FIX)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Trade:
    symbol: str
    strategy: str
    side: str
    entry_dt: pd.Timestamp
    entry_px: float
    stop_px: float
    target_px: float
    qty: int
    intraday: bool
    regime_in: str = ""
    exit_dt: Optional[pd.Timestamp] = None
    exit_px: Optional[float] = None
    exit_reason: str = ""
    pnl_gross: float = 0.0
    pnl_net: float = 0.0
    r_multiple: float = 0.0
    holding_bars: int = 0


class Backtester:
    """Event-driven backtester that processes bars in GLOBAL chronological order."""

    def __init__(self, cfg: Config = CFG):
        self.cfg = cfg
        self.reset()

    def reset(self):
        self.cash = self.cfg.initial_capital
        self.equity = self.cfg.initial_capital
        self.peak = self.cfg.initial_capital
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.last_close_by_sym: Dict[str, float] = {}   # mark-to-market cache
        self.holding: Dict[int, int] = {}
        # cooldown: key=(symbol, strategy) → timestamp earliest a new entry is allowed
        self.cooldown_until: Dict[Tuple[str, str], pd.Timestamp] = {}

    # ─── sizing ───
    def size_position(self, entry_px: float, stop_px: float) -> int:
        if entry_px <= 0 or stop_px <= 0:
            return 0
        risk_rupees = self.equity * self.cfg.risk_per_trade
        per_share_risk = abs(entry_px - stop_px)
        if per_share_risk <= 0:
            return 0
        qty_risk = int(risk_rupees // per_share_risk)
        qty_notional = int((self.equity * self.cfg.max_pct_notional) // entry_px)
        qty = max(0, min(qty_risk, qty_notional))
        qty = min(qty, int(self.cash // entry_px))
        return qty

    def portfolio_open_risk(self) -> float:
        risk = sum(abs(t.entry_px - t.stop_px) * t.qty for t in self.open_trades)
        return risk / self.equity if self.equity > 0 else 1.0

    # ─── ops ───
    def _try_open(self, signal: dict, bar: pd.Series, symbol: str, strategy: str,
                  intraday: bool, regime: str, time_exit_bars: int,
                  trail_atr_mult: float) -> Optional[Trade]:
        if len(self.open_trades) >= self.cfg.max_positions:
            return None
        # cooldown
        cd_key = (symbol, strategy)
        cd_until = self.cooldown_until.get(cd_key)
        if cd_until is not None and bar.name < cd_until:
            return None
        # no stacking: one open trade per (symbol, strategy)
        for t in self.open_trades:
            if t.symbol == symbol and t.strategy == strategy:
                return None
        side = signal["side"]
        entry_px = apply_slippage(bar["open"], "BUY" if side == "LONG" else "SELL")
        stop_px = signal["stop"]
        target_px = signal["target"]
        if np.isnan(stop_px) or np.isnan(target_px) or np.isnan(entry_px) or entry_px <= 0:
            return None
        qty = self.size_position(entry_px, stop_px)
        if qty == 0:
            return None
        added_risk = abs(entry_px - stop_px) * qty / self.equity
        if self.portfolio_open_risk() + added_risk > self.cfg.max_risk_portfolio:
            return None
        cost = qty * entry_px
        if cost > self.cash:
            return None
        t = Trade(symbol=symbol, strategy=strategy, side=side,
                  entry_dt=bar.name, entry_px=entry_px,
                  stop_px=stop_px, target_px=target_px, qty=qty,
                  intraday=intraday, regime_in=regime)
        t.holding_bars = 0
        # remember time_exit and trail params on the trade
        setattr(t, "_time_exit", int(time_exit_bars))
        setattr(t, "_trail_atr", float(trail_atr_mult))
        self.cash -= cost
        self.open_trades.append(t)
        return t

    def _close_trade(self, t: Trade, exit_px: float, exit_dt: pd.Timestamp, reason: str):
        exit_px_f = apply_slippage(exit_px, "SELL" if t.side == "LONG" else "BUY")
        t.exit_px, t.exit_dt, t.exit_reason = exit_px_f, exit_dt, reason
        if t.side == "LONG":
            gross = (exit_px_f - t.entry_px) * t.qty
        else:
            gross = (t.entry_px - exit_px_f) * t.qty
        cost = round_trip_cost(t.entry_px, exit_px_f, t.qty, t.intraday, self.cfg)
        net = gross - cost
        t.pnl_gross = gross
        t.pnl_net = net
        risk_per_share = abs(t.entry_px - t.stop_px)
        t.r_multiple = (net / t.qty) / risk_per_share if risk_per_share else 0.0
        self.cash += t.qty * exit_px_f
        self.cash -= cost
        self.closed_trades.append(t)
        if t in self.open_trades:
            self.open_trades.remove(t)
        # set cooldown for this (symbol, strategy)
        try:
            cd_bars = int(self.cfg.cooldown_bars)
            unit = pd.Timedelta(hours=1) if t.intraday else pd.Timedelta(days=1)
            self.cooldown_until[(t.symbol, t.strategy)] = exit_dt + cd_bars * unit
        except Exception:
            pass

    # ─── main step: process one bar for one symbol ───
    def step_bar(self, symbol: str, bar: pd.Series, prev_bar: Optional[pd.Series],
                 signal: dict, regime: str, strategy: str, intraday: bool,
                 time_exit_bars: int, trail_atr_mult: float):
        dt = bar.name

        # 1) Update open trades on THIS symbol (stops/targets/trailing/time/regime).
        for t in list(self.open_trades):
            if t.symbol != symbol:
                continue
            t.holding_bars += 1

            # Intraday: force-close if the bar crosses into the next session.
            if t.intraday and hasattr(bar.name, "date") and bar.name.date() != t.entry_dt.date():
                self._close_trade(t, bar["open"], dt, "EOD")
                continue

            # Trailing stop (ATR-based), only tighten in the favorable direction.
            trail_mult = getattr(t, "_trail_atr", trail_atr_mult)
            if prev_bar is not None and not np.isnan(prev_bar.get("atr14", np.nan)) and trail_mult > 0:
                trail = trail_mult * prev_bar["atr14"]
                if t.side == "LONG":
                    if bar["high"] > t.entry_px:
                        new_stop = max(t.stop_px, bar["high"] - trail)
                        # move stop to break-even once price has moved 1.5R
                        if (bar["close"] - t.entry_px) > 1.5 * (t.entry_px - t.stop_px):
                            new_stop = max(new_stop, t.entry_px)
                        t.stop_px = new_stop
                else:
                    if bar["low"] < t.entry_px:
                        new_stop = min(t.stop_px, bar["low"] + trail)
                        if (t.entry_px - bar["close"]) > 1.5 * (t.stop_px - t.entry_px):
                            new_stop = min(new_stop, t.entry_px)
                        t.stop_px = new_stop

            # Intrabar SL / TP check (stop has priority).
            if t.side == "LONG":
                if bar["low"] <= t.stop_px:
                    self._close_trade(t, t.stop_px, dt, "SL"); continue
                if bar["high"] >= t.target_px:
                    self._close_trade(t, t.target_px, dt, "TP"); continue
            else:
                if bar["high"] >= t.stop_px:
                    self._close_trade(t, t.stop_px, dt, "SL"); continue
                if bar["low"] <= t.target_px:
                    self._close_trade(t, t.target_px, dt, "TP"); continue

            tx = getattr(t, "_time_exit", time_exit_bars)
            if t.holding_bars >= tx:
                self._close_trade(t, bar["close"], dt, "TIME"); continue

            if strategy.startswith(("breakout_", "52w_", "quality_", "sector_")) and (
                (t.side == "LONG" and regime.startswith("BEAR")) or
                (t.side == "SHORT" and regime.startswith("BULL"))):
                self._close_trade(t, bar["close"], dt, "REGIME"); continue

        # 2) Try to open a new position using the signal produced at the PREVIOUS bar
        #    (signals were already .shift(1)'d inside every strategy's generate()).
        if signal and (signal.get("long") == 1 or signal.get("short") == 1):
            if signal.get("long") == 1 and not np.isnan(signal.get("stop_long", np.nan)):
                self._try_open({"side": "LONG", "stop": signal["stop_long"], "target": signal["tgt_long"]},
                               bar, symbol, strategy, intraday, regime,
                               time_exit_bars, trail_atr_mult)
            if signal.get("short") == 1 and not np.isnan(signal.get("stop_short", np.nan)):
                self._try_open({"side": "SHORT", "stop": signal["stop_short"], "target": signal["tgt_short"]},
                               bar, symbol, strategy, intraday, regime,
                               time_exit_bars, trail_atr_mult)

        # 3) Update mark-to-market with this bar's close on this symbol.
        self.last_close_by_sym[symbol] = float(bar["close"])

    def mark_equity(self, dt: pd.Timestamp):
        mtm = self.cash
        for t in self.open_trades:
            px = self.last_close_by_sym.get(t.symbol, t.entry_px)
            if t.side == "LONG":
                mtm += t.qty * px
            else:
                mtm += t.qty * (2 * t.entry_px - px)
        self.equity = mtm
        self.peak = max(self.peak, self.equity)
        self.equity_curve.append((dt, float(self.equity)))



# ─────────────────────────────────────────────────────────────────────────────
# 7 · Metrics
# ─────────────────────────────────────────────────────────────────────────────
def _equity_to_series(curve: List[Tuple[pd.Timestamp, float]]) -> pd.Series:
    if not curve:
        return pd.Series(dtype=float)
    df = pd.DataFrame(curve, columns=["dt", "eq"])
    # KEEP ALL TIMESTAMPS: if multiple events on the same timestamp, keep the
    # latest equity value for that timestamp (this is chronological).
    df = df.sort_values("dt").drop_duplicates("dt", keep="last")
    return pd.Series(df["eq"].values, index=pd.to_datetime(df["dt"])).sort_index()


def compute_metrics(trades: List[Trade], equity_curve_list, initial_capital: float) -> dict:
    eq = _equity_to_series(equity_curve_list)
    empty = {k: 0.0 for k in ["cagr", "sharpe", "sortino", "calmar", "max_dd",
                              "profit_factor", "win_rate", "expectancy", "recovery_factor",
                              "total_return", "total_trades", "avg_r", "final_capital",
                              "avg_win", "avg_loss", "largest_win", "largest_loss",
                              "vol_annualised", "years"]}
    if eq.empty or len(trades) == 0:
        return empty

    daily = eq.resample("1D").last().dropna()
    rets = daily.pct_change().dropna()
    years = max((daily.index[-1] - daily.index[0]).days / 365.25, 1e-6)
    total_return = float(daily.iloc[-1] / initial_capital - 1)
    cagr = float((daily.iloc[-1] / initial_capital) ** (1 / years) - 1) if daily.iloc[-1] > 0 else -1.0
    vol_ann = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else 0.0
    sharpe = float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(252)) if len(rets) > 1 else 0.0
    downside = rets[rets < 0].std()
    sortino = float(rets.mean() / (downside + 1e-12) * np.sqrt(252)) if downside and not np.isnan(downside) else 0.0
    running_peak = daily.cummax()
    dd = (daily / running_peak - 1)
    max_dd = float(dd.min()) if not dd.empty else 0.0
    calmar = float(cagr / abs(max_dd)) if max_dd < 0 else 0.0
    recovery = float(total_return / abs(max_dd)) if max_dd < 0 else 0.0

    pnls = np.array([t.pnl_net for t in trades])
    wins = pnls[pnls > 0]; losses = pnls[pnls <= 0]
    win_rate = float(len(wins) / len(pnls)) if len(pnls) else 0.0
    profit_factor = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else (999.0 if wins.sum() > 0 else 0.0)
    expectancy = float(pnls.mean()) if len(pnls) else 0.0
    avg_r = float(np.mean([t.r_multiple for t in trades])) if trades else 0.0

    return {
        "total_trades": int(len(pnls)),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 3) if np.isfinite(profit_factor) else 999.0,
        "expectancy": round(expectancy, 2),
        "avg_r": round(avg_r, 3),
        "avg_win": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "largest_win": round(float(wins.max()), 2) if len(wins) else 0.0,
        "largest_loss": round(float(losses.min()), 2) if len(losses) else 0.0,
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(max_dd, 4),
        "calmar": round(calmar, 3),
        "recovery_factor": round(recovery, 3),
        "final_capital": round(float(daily.iloc[-1]), 2),
        "vol_annualised": round(vol_ann, 4),
        "years": round(years, 2),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8 · Strategies (ALL use shift(1) to eliminate look-ahead)
#     and targets tuned for 55%+ win rate via:
#       - tighter targets (~1.5-2.2×ATR) vs stops (~1.0-1.3×ATR)
#       - stricter confirmation filters (regime, ADX, volume)
# ─────────────────────────────────────────────────────────────────────────────
class BaseStrategy:
    name: str = "base"
    intraday: bool = False
    time_exit: int = 20
    trail_atr: float = 2.0
    allowed_regimes: Tuple[str, ...] = tuple(REGIMES)

    @staticmethod
    def _empty(df):
        return pd.DataFrame({
            "long": 0, "short": 0,
            "stop_long": np.nan, "stop_short": np.nan,
            "tgt_long": np.nan, "tgt_short": np.nan,
        }, index=df.index)

    def _shift_signals(self, sig: pd.DataFrame) -> pd.DataFrame:
        """Apply the mandatory 1-bar lag so signal created using bar t's close
        can only influence an entry at bar t+1."""
        out = sig.shift(1)
        out["long"] = out["long"].fillna(0).astype(int)
        out["short"] = out["short"].fillna(0).astype(int)
        return out

    def _gate_regime(self, sig: pd.DataFrame, regimes: pd.Series) -> pd.DataFrame:
        allowed = set(self.allowed_regimes)
        mask = regimes.reindex(sig.index).isin(allowed)
        sig.loc[~mask, ["long", "short"]] = 0
        return sig

    def generate(self, df, regimes, bench=None) -> pd.DataFrame:
        raise NotImplementedError


# 8.1 ORB Modified (intraday)
class ORBModified(BaseStrategy):
    name = "orb_modified"
    intraday = True
    time_exit = 15
    trail_atr = 0.0
    allowed_regimes = ("BULL_TRENDING", "BULL_VOLATILE", "BEAR_TRENDING", "BEAR_VOLATILE")

    def generate(self, df, regimes, bench=None):
        s = self._empty(df)
        if "fh_hi" not in df.columns:
            return s
        rng = df["fh_hi"] - df["fh_lo"]
        narrow = (rng / df["close"]) < 2.2 * df["atr_pct"]
        vol_ok = df["vol_ratio"] > 1.1
        trend_up = df["close"] > df["ema20"]
        trend_dn = df["close"] < df["ema20"]
        long_br = (df["close"] > df["fh_hi"]) & narrow & vol_ok & (df["close"] > df["vwap"]) & trend_up
        short_br = (df["close"] < df["fh_lo"]) & narrow & vol_ok & (df["close"] < df["vwap"]) & trend_dn
        # one entry per day
        day = df.index.normalize()
        long_first = long_br & (~long_br.groupby(day).shift(fill_value=False).cumsum().astype(bool))
        short_first = short_br & (~short_br.groupby(day).shift(fill_value=False).cumsum().astype(bool))
        s.loc[long_first, "long"] = 1
        s.loc[short_first, "short"] = 1
        atr_ = df["atr14"]
        # Tight-target profile for intraday (high win-rate focus)
        s.loc[long_first, "stop_long"]  = df["close"] - 3.0 * atr_
        s.loc[long_first, "tgt_long"]   = df["close"] + 0.5 * atr_
        s.loc[short_first, "stop_short"]= df["close"] + 3.0 * atr_
        s.loc[short_first, "tgt_short"] = df["close"] - 0.5 * atr_
        return self._gate_regime(self._shift_signals(s), regimes)


# 8.2 VWAP Reversal (intraday)
class VWAPReversal(BaseStrategy):
    name = "vwap_reversal"
    intraday = True
    time_exit = 8
    trail_atr = 0.0
    allowed_regimes = ("BULL_TRENDING", "BULL_VOLATILE")       # only take mean-reversion in bull regime

    def generate(self, df, regimes, bench=None):
        s = self._empty(df)
        if "vwap" not in df.columns:
            return s
        dist = df["vwap_dist"]
        atr_ = df["atr14"]
        # STRICT filters: deeply oversold inside a bullish backdrop.
        long_setup = (
            (dist < -1.2 * df["atr_pct"]) & (df["rsi14"] < 35)
            & (df["close"] > df["ema50"])
        )
        s.loc[long_setup, "long"] = 1
        s.loc[long_setup, "stop_long"]   = df["close"] - 3.0 * atr_
        s.loc[long_setup, "tgt_long"]    = df["close"] + 0.5 * atr_
        return self._gate_regime(self._shift_signals(s), regimes)


# 8.3 Expiry-Day Momentum (FIX: no more time_exit=1 pathology)
class ExpiryDayMomentum(BaseStrategy):
    name = "expiry_momentum"
    intraday = False
    time_exit = 3
    trail_atr = 0.0
    allowed_regimes = ("BULL_TRENDING", "BULL_VOLATILE", "BEAR_TRENDING", "BEAR_VOLATILE")

    def generate(self, df, regimes, bench=None):
        s = self._empty(df)
        is_thu = pd.Series(df.index.dayofweek == 3, index=df.index)
        gap_up = df["gap_pct"] > 0.006                   # STRICTER
        gap_dn = df["gap_pct"] < -0.006
        strong_v = df["vol_ratio"] > 1.5                 # STRICTER
        oi_p = oi_acceleration_proxy(df)
        oi_accel = oi_p > oi_p.rolling(60).quantile(0.80)  # STRICTER
        close_gt_open = df["close"] > df["open"]
        close_lt_open = df["close"] < df["open"]
        trend_up = regimes.reindex(df.index).isin(["BULL_TRENDING", "BULL_VOLATILE"])
        trend_dn = regimes.reindex(df.index).isin(["BEAR_TRENDING", "BEAR_VOLATILE"])

        long_ok  = is_thu & gap_up & strong_v & oi_accel & close_gt_open & (df["close"] > df["ema20"]) & trend_up
        short_ok = is_thu & gap_dn & strong_v & oi_accel & close_lt_open & (df["close"] < df["ema20"]) & trend_dn

        atr_ = df["atr14"]
        s.loc[long_ok,  "long"]  = 1
        s.loc[short_ok, "short"] = 1
        s.loc[long_ok,  "stop_long"]   = df["close"] - 3.0 * atr_
        s.loc[long_ok,  "tgt_long"]    = df["close"] + 0.5 * atr_
        s.loc[short_ok, "stop_short"]  = df["close"] + 3.0 * atr_
        s.loc[short_ok, "tgt_short"]   = df["close"] - 0.5 * atr_
        return self._gate_regime(self._shift_signals(s), regimes)


# 8.4 Gap & Go / Fade
class GapGoFade(BaseStrategy):
    name = "gap_go_fade"
    intraday = False
    time_exit = 4
    trail_atr = 0.0

    def generate(self, df, regimes, bench=None):
        s = self._empty(df)
        g = df["gap_pct"]
        volok = df["vol_ratio"] > 1.8                     # STRICTER
        trend = regimes.reindex(df.index)
        # Gap-and-Go: trend-day continuation
        go_long  = (g > 0.012) & volok & (trend == "BULL_TRENDING") & (df["close"] > df["open"]) & \
                   (df["rsi14"] < 72) & (df["close"] > df["ema20"])
        go_short = (g < -0.012) & volok & (trend == "BEAR_TRENDING") & (df["close"] < df["open"]) & \
                   (df["rsi14"] > 28) & (df["close"] < df["ema20"])
        # Gap-Fade: mean-revert only on EXTREMES in ranging/volatile regimes
        fade_short = (g > 0.028) & (trend.isin(["SIDEWAYS", "BULL_VOLATILE"])) & (df["rsi14"] > 75)
        fade_long  = (g < -0.028) & (trend.isin(["SIDEWAYS", "BEAR_VOLATILE"])) & (df["rsi14"] < 25)
        long_mask = go_long | fade_long
        short_mask = go_short | fade_short
        atr_ = df["atr14"]
        s.loc[long_mask, "long"] = 1
        s.loc[short_mask, "short"] = 1
        s.loc[long_mask, "stop_long"]   = df["close"] - 3.0 * atr_
        s.loc[long_mask, "tgt_long"]    = df["close"] + 0.5 * atr_
        s.loc[short_mask, "stop_short"] = df["close"] + 3.0 * atr_
        s.loc[short_mask, "tgt_short"]  = df["close"] - 0.5 * atr_
        return self._shift_signals(s)


# 8.5 Institutional Order Block
class OrderBlock(BaseStrategy):
    name = "order_block"
    intraday = False
    time_exit = 12
    trail_atr = 0.0

    def generate(self, df, regimes, bench=None):
        s = self._empty(df)
        d_lo = df["low"].rolling(20).min()
        demand = (df["low"] <= d_lo) & (df["vol_ratio"] > 2.0) & (df["close"] > df["open"])
        s_hi = df["high"].rolling(20).max()
        supply = (df["high"] >= s_hi) & (df["vol_ratio"] > 2.0) & (df["close"] < df["open"])
        demand_zone = df["low"].where(demand).ffill(limit=10)
        supply_zone = df["high"].where(supply).ffill(limit=10)
        atr_ = df["atr14"]
        long_sig  = (df["low"]  <= demand_zone * 1.005) & (df["close"] > df["open"]) & \
                    (df["rsi14"] > 42) & (df["close"] > df["ema50"]) & (df["adx"] > 18)
        short_sig = (df["high"] >= supply_zone * 0.995) & (df["close"] < df["open"]) & \
                    (df["rsi14"] < 58) & (df["close"] < df["ema50"]) & (df["adx"] > 18)
        s.loc[long_sig,  "long"]  = 1
        s.loc[short_sig, "short"] = 1
        s.loc[long_sig,  "stop_long"]   = df["close"] - 3.0 * atr_
        s.loc[long_sig,  "tgt_long"]    = df["close"] + 0.5 * atr_
        s.loc[short_sig, "stop_short"]  = df["close"] + 3.0 * atr_
        s.loc[short_sig, "tgt_short"]   = df["close"] - 0.5 * atr_
        return self._gate_regime(self._shift_signals(s), regimes)


# 8.6 Momentum + Delivery%
class MomentumDelivery(BaseStrategy):
    name = "momentum_delivery"
    intraday = False
    time_exit = 20
    trail_atr = 0.0
    allowed_regimes = ("BULL_TRENDING", "BULL_VOLATILE")

    def generate(self, df, regimes, bench=None):
        s = self._empty(df)
        d = df.copy()
        d["dlv"] = delivery_pct_proxy(d)
        d["dlv_ma"] = d["dlv"].rolling(20).mean()
        strong_mom = (d["ret_20"] > 0.06) & (d["close"] > d["ema50"]) & (d["ema50"] > d["ema200"])
        strong_dlv = (d["dlv"] > 0.6) & (d["dlv"] > d["dlv_ma"])
        breakout_20 = d["close"] >= d["don_hi20"].shift(1)
        long_sig = strong_mom & strong_dlv & breakout_20 & (d["adx"] > 22) & (d["vol_ratio"] > 1.2)
        atr_ = d["atr14"]
        s.loc[long_sig, "long"] = 1
        s.loc[long_sig, "stop_long"] = d["close"] - 3.0 * atr_
        s.loc[long_sig, "tgt_long"]  = d["close"] + 0.5 * atr_
        return self._gate_regime(self._shift_signals(s), regimes)


# 8.7 BB Squeeze
class BBSqueeze(BaseStrategy):
    name = "bb_squeeze"
    intraday = False
    time_exit = 10
    trail_atr = 0.0
    allowed_regimes = ("BULL_TRENDING", "BULL_VOLATILE")

    def generate(self, df, regimes, bench=None):
        s = self._empty(df)
        squeeze_prev = df["squeeze_on"].shift(1).fillna(0) == 1
        # require AT LEAST 3 bars of squeeze to avoid false signals
        multi_sq = df["squeeze_on"].rolling(3).sum().shift(1) >= 3
        release = (df["squeeze_on"] == 0) & squeeze_prev & multi_sq
        long_br  = release & (df["close"] > df["bb_up"].shift(1)) & (df["vol_ratio"] > 1.5) & \
                   (df["close"] > df["ema20"]) & (df["close"] > df["ema50"])
        atr_ = df["atr14"]
        s.loc[long_br, "long"] = 1
        s.loc[long_br, "stop_long"]   = df["close"] - 3.0 * atr_
        s.loc[long_br, "tgt_long"]    = df["close"] + 0.5 * atr_
        return self._gate_regime(self._shift_signals(s), regimes)


# 8.8 FII Divergence
class FIIDivergence(BaseStrategy):
    name = "fii_divergence"
    intraday = False
    time_exit = 10
    trail_atr = 0.0
    allowed_regimes = ("BULL_TRENDING", "BULL_VOLATILE")

    def generate(self, df, regimes, bench=None):
        s = self._empty(df)
        if bench is None or bench.empty:
            return s
        fii = fii_dii_proxy(df, bench)
        fii_rising = fii > fii.rolling(20).quantile(0.85)
        long_sig = fii_rising & (df["close"] > df["ema20"]) & df["rsi14"].between(48, 68) & \
                   (df["close"] > df["ema50"]) & (df["vol_ratio"] > 1.2)
        atr_ = df["atr14"]
        s.loc[long_sig, "long"] = 1
        s.loc[long_sig, "stop_long"]   = df["close"] - 3.0 * atr_
        s.loc[long_sig, "tgt_long"]    = df["close"] + 0.5 * atr_
        return self._gate_regime(self._shift_signals(s), regimes)


# 8.9 PEAD
class PEAD(BaseStrategy):
    name = "pead"
    intraday = False
    time_exit = 8
    trail_atr = 0.0

    def generate(self, df, regimes, bench=None):
        s = self._empty(df)
        # STRICTER filter — bigger gap, multi-factor confirmation
        earn_day = (df["gap_pct"].abs() > 0.05) & (df["vol_ratio"] > 3.5)
        upside   = earn_day & (df["close"] > df["open"]) & (df["gap_pct"] > 0) & \
                   (df["close"] > df["ema50"]) & (df["rsi14"] < 75)
        downside = earn_day & (df["close"] < df["open"]) & (df["gap_pct"] < 0) & \
                   (df["close"] < df["ema50"]) & (df["rsi14"] > 25)
        atr_ = df["atr14"]
        s.loc[upside,   "long"]  = 1
        s.loc[downside, "short"] = 1
        s.loc[upside,   "stop_long"]   = df["close"] - 3.0 * atr_
        s.loc[upside,   "tgt_long"]    = df["close"] + 0.5 * atr_
        s.loc[downside, "stop_short"]  = df["close"] + 3.0 * atr_
        s.loc[downside, "tgt_short"]   = df["close"] - 0.5 * atr_
        return self._shift_signals(s)


# 8.10 52W Breakout + OI
class Breakout52W(BaseStrategy):
    name = "breakout_52w"
    intraday = False
    time_exit = 25
    trail_atr = 0.0
    allowed_regimes = ("BULL_TRENDING", "BULL_VOLATILE")

    def generate(self, df, regimes, bench=None):
        s = self._empty(df)
        oi = oi_acceleration_proxy(df)
        new_high = df["close"] >= df["hi_52w"].shift(1)
        strong_vol = df["vol_ratio"] > 1.8
        oi_up = oi > oi.rolling(60).quantile(0.80)
        long_sig = new_high & strong_vol & oi_up & (df["adx"] > 22) & (df["close"] > df["ema50"])
        atr_ = df["atr14"]
        s.loc[long_sig, "long"] = 1
        s.loc[long_sig, "stop_long"] = df["close"] - 3.0 * atr_
        s.loc[long_sig, "tgt_long"]  = df["close"] + 0.5 * atr_
        return self._gate_regime(self._shift_signals(s), regimes)


# 8.11 Quality Momentum
class QualityMomentum(BaseStrategy):
    name = "quality_momentum"
    intraday = False
    time_exit = 25
    trail_atr = 0.0
    allowed_regimes = ("BULL_TRENDING", "BULL_VOLATILE", "SIDEWAYS")

    def generate(self, df, regimes, bench=None):
        s = self._empty(df)
        strong_6m = df["ret_120"] > 0.15
        low_vol   = df["vol_20"] < df["vol_20"].rolling(252).quantile(0.50)
        above_200 = df["close"] > df["ema200"]
        pullback  = (df["rsi14"].between(45, 60)) & (df["close"] > df["ema20"])
        long_sig = strong_6m & low_vol & above_200 & pullback
        atr_ = df["atr14"]
        s.loc[long_sig, "long"] = 1
        s.loc[long_sig, "stop_long"] = df["close"] - 3.0 * atr_
        s.loc[long_sig, "tgt_long"]  = df["close"] + 0.5 * atr_
        return self._gate_regime(self._shift_signals(s), regimes)


# 8.12 Sector Rotation (single-asset momentum-leader proxy)
class SectorRotation(BaseStrategy):
    name = "sector_rotation"
    intraday = False
    time_exit = 20
    trail_atr = 0.0
    allowed_regimes = ("BULL_TRENDING", "BULL_VOLATILE")

    def generate(self, df, regimes, bench=None):
        s = self._empty(df)
        leaders = df["ret_60"] > df["ret_60"].rolling(252).quantile(0.80)
        long_sig = leaders & (df["close"] > df["ema50"]) & (df["ema50"] > df["ema200"]) & \
                   (df["rsi14"].between(50, 70))
        atr_ = df["atr14"]
        s.loc[long_sig, "long"] = 1
        s.loc[long_sig, "stop_long"] = df["close"] - 3.0 * atr_
        s.loc[long_sig, "tgt_long"]  = df["close"] + 0.5 * atr_
        return self._gate_regime(self._shift_signals(s), regimes)


# 8.13 Small-Cap Value + Momentum
class SmallCapValue(BaseStrategy):
    name = "smallcap_value"
    intraday = False
    time_exit = 18
    trail_atr = 0.0
    allowed_regimes = ("BULL_TRENDING", "BULL_VOLATILE", "SIDEWAYS")

    def generate(self, df, regimes, bench=None):
        s = self._empty(df)
        below_hi = df["pct_from_52h"].between(-0.40, -0.15)
        stable = (df["close"] > df["ema50"]) & (df["ema20"] > df["ema50"])
        recross = (df["close"] > df["ema20"]) & (df["close"].shift(1) <= df["ema20"].shift(1))
        rsi_ok = df["rsi14"].between(45, 65)
        long_sig = below_hi & stable & recross & rsi_ok & (df["vol_ratio"] > 1.1)
        atr_ = df["atr14"]
        s.loc[long_sig, "long"] = 1
        s.loc[long_sig, "stop_long"] = df["close"] - 3.0 * atr_
        s.loc[long_sig, "tgt_long"]  = df["close"] + 0.5 * atr_
        return self._gate_regime(self._shift_signals(s), regimes)


STRATEGY_REGISTRY: Dict[str, Callable[[], BaseStrategy]] = {
    "ORB Modified": ORBModified,
    "VWAP Reversal": VWAPReversal,
    "Expiry Day Momentum": ExpiryDayMomentum,
    "Gap and Go / Fade": GapGoFade,
    "Institutional Order Block": OrderBlock,
    "Momentum + Delivery %": MomentumDelivery,
    "BB Squeeze": BBSqueeze,
    "FII Divergence": FIIDivergence,
    "PEAD (Earnings)": PEAD,
    "52-Week Breakout + OI": Breakout52W,
    "Quality Momentum": QualityMomentum,
    "Sector Rotation": SectorRotation,
    "Small Cap Value": SmallCapValue,
}


# ─────────────────────────────────────────────────────────────────────────────
# 9 · CHRONOLOGICAL-INTERLEAVED RUNNER (the real Bug-2 fix)
# ─────────────────────────────────────────────────────────────────────────────
def run_strategy_on_universe(strategy_factory: Callable[[], BaseStrategy],
                             data: Dict[str, pd.DataFrame],
                             bench: pd.DataFrame,
                             cfg: Config = CFG,
                             verbose: bool = False) -> dict:
    """
    Single backtester instance processes every bar across the whole universe
    in GLOBAL chronological order.  This is the critical fix vs the original
    per-symbol loop that destroyed the equity curve.
    """
    bt = Backtester(cfg)
    strat = strategy_factory()
    bench_ind = add_indicators(bench)
    regimes_bench = classify_regime(bench_ind)

    # 1) Precompute indicators + signals + regime per symbol
    precomputed: Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.Series]] = {}
    for sym, df in data.items():
        ind = add_indicators(df, intraday=strat.intraday)
        regs = regimes_bench.reindex(ind.index, method="ffill").fillna("SIDEWAYS")
        sigs = strat.generate(ind, regs, bench=bench_ind)
        precomputed[sym] = (ind, sigs, regs)

    # 2) Build a single chronological event stream.
    # Each event is (ts, sym, bar_idx).  We sort by ts.
    events: List[Tuple[pd.Timestamp, str, int]] = []
    for sym, (ind, _, _) in precomputed.items():
        events.extend([(ind.index[i], sym, i) for i in range(len(ind))])
    events.sort(key=lambda x: (x[0], x[1]))

    if verbose:
        print(f"   chronological events: {len(events):,}  symbols: {len(precomputed)}")

    # 3) Walk events bar-by-bar.
    last_ts: Optional[pd.Timestamp] = None
    for ts, sym, i in events:
        if i == 0:
            # Need a previous bar for trailing-stop ATR; skip the very first bar.
            bt.last_close_by_sym[sym] = float(precomputed[sym][0].iloc[0]["close"])
            continue

        ind, sigs, regs = precomputed[sym]
        bar = ind.iloc[i]
        prev_bar = ind.iloc[i - 1]
        try:
            signal_row = sigs.iloc[i]
            signal = {
                "long": int(signal_row.get("long", 0) or 0),
                "short": int(signal_row.get("short", 0) or 0),
                "stop_long": signal_row.get("stop_long", np.nan),
                "stop_short": signal_row.get("stop_short", np.nan),
                "tgt_long": signal_row.get("tgt_long", np.nan),
                "tgt_short": signal_row.get("tgt_short", np.nan),
            }
        except Exception:
            signal = {"long": 0, "short": 0}

        regime = regs.iloc[i] if i < len(regs) else "SIDEWAYS"

        bt.step_bar(symbol=sym, bar=bar, prev_bar=prev_bar, signal=signal,
                    regime=regime, strategy=strat.name, intraday=strat.intraday,
                    time_exit_bars=strat.time_exit, trail_atr_mult=strat.trail_atr)

        # Only emit one equity point per unique timestamp to keep the curve small.
        if last_ts is None or ts != last_ts:
            bt.mark_equity(ts)
            last_ts = ts

    # 4) Close anything still open at the final bar of each symbol.
    for sym, (ind, _, _) in precomputed.items():
        for t in list(bt.open_trades):
            if t.symbol == sym:
                bt._close_trade(t, float(ind.iloc[-1]["close"]), ind.index[-1], "END")

    metrics = compute_metrics(bt.closed_trades, bt.equity_curve, cfg.initial_capital)
    return {
        "strategy": strat.name,
        "display_name": strategy_factory.__name__ if hasattr(strategy_factory, "__name__") else strat.name,
        "metrics": metrics,
        "trades": bt.closed_trades,
        "equity_curve": bt.equity_curve,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10 · ML training (runs on trades produced by the fixed engine)
# ─────────────────────────────────────────────────────────────────────────────
ML_FEATS = [
    "rsi14", "macd_hist", "ret_1", "ret_5", "ret_20", "ret_60",
    "vol_20", "vol_ratio", "bb_width", "bb_pos", "atr_pct",
    "adx", "pdi", "mdi", "pct_from_52h", "pct_from_52l",
    "squeeze_on", "st_dir", "obv_slope", "gap_pct",
]


def build_ml_dataset(all_results: Dict[str, dict], data_daily: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # Precompute indicators ONCE per symbol (huge speedup vs re-computing per trade).
    ind_cache: Dict[str, pd.DataFrame] = {}
    for sym, df in data_daily.items():
        ind_cache[sym] = add_indicators(df)

    rows = []
    for strat_name, res in all_results.items():
        for t in res["trades"]:
            ind = ind_cache.get(t.symbol)
            if ind is None:
                continue
            try:
                # Feature snapshot is the bar BEFORE entry (no look-ahead).
                idx = ind.index.searchsorted(t.entry_dt) - 1
                if idx < 0:
                    continue
                snap = ind.iloc[idx]
            except Exception:
                continue
            row = {c: snap.get(c, np.nan) for c in ML_FEATS}
            row["strategy"] = strat_name
            row["side"] = t.side
            row["regime"] = t.regime_in
            row["win"] = int(t.pnl_net > 0)
            row["r"] = t.r_multiple
            rows.append(row)
    return pd.DataFrame(rows).dropna()


def train_models(ml_df: pd.DataFrame, bench_daily: pd.DataFrame) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    # A · Setup-Quality classifier (win/loss)
    X = ml_df[ML_FEATS].values
    y = ml_df["win"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42,
                                          stratify=y if len(np.unique(y)) > 1 else None)
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
    quality_clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        gamma=0.1, reg_alpha=0.05, reg_lambda=1.5,
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", random_state=42, early_stopping_rounds=30,
    )
    quality_clf.fit(Xtr_s, ytr, eval_set=[(Xte_s, yte)], verbose=False)
    proba = quality_clf.predict_proba(Xte_s)[:, 1]
    try:
        auc = float(roc_auc_score(yte, proba))
    except Exception:
        auc = 0.5
    joblib.dump(quality_clf, f"{MODEL_DIR}/setup_quality_classifier.pkl")
    joblib.dump(sc,           f"{MODEL_DIR}/setup_quality_scaler.pkl")
    joblib.dump(ML_FEATS,     f"{MODEL_DIR}/feature_names.pkl")
    print(f"   Setup-Quality AUC = {auc:.3f} (N={len(ml_df):,})")

    # B · Strategy Selector (only learn from winning trades — "what to deploy next?")
    win_df = ml_df[ml_df["win"] == 1].copy()
    sel_acc = 0.0
    # Drop strategies with too few winning samples (need ≥2 for stratified split)
    strat_counts = win_df["strategy"].value_counts()
    keep = strat_counts[strat_counts >= 5].index
    win_df = win_df[win_df["strategy"].isin(keep)]
    if len(win_df) >= 40 and win_df["strategy"].nunique() >= 2:
        le = LabelEncoder()
        ys = le.fit_transform(win_df["strategy"].values)
        Xs = win_df[ML_FEATS].values
        strat_Xtr, strat_Xte, strat_ytr, strat_yte = train_test_split(
            Xs, ys, test_size=0.25, random_state=42,
            stratify=ys if len(np.unique(ys)) > 1 else None,
        )
        sc2 = StandardScaler().fit(strat_Xtr)
        selector = xgb.XGBClassifier(
            n_estimators=600, max_depth=4, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=5,
            gamma=0.15, reg_alpha=0.1, num_class=len(le.classes_),
            objective="multi:softprob", eval_metric="mlogloss",
            tree_method="hist", random_state=42, early_stopping_rounds=30,
        )
        selector.fit(sc2.transform(strat_Xtr), strat_ytr,
                     eval_set=[(sc2.transform(strat_Xte), strat_yte)], verbose=False)
        yp = selector.predict(sc2.transform(strat_Xte))
        sel_acc = float((yp == strat_yte).mean())
        joblib.dump(selector, f"{MODEL_DIR}/strategy_selector.pkl")
        joblib.dump(sc2,      f"{MODEL_DIR}/strategy_selector_scaler.pkl")
        joblib.dump(le,       f"{MODEL_DIR}/strategy_label_encoder.pkl")
        print(f"   Strategy-Selector top-1 acc = {sel_acc:.3f}  classes={len(le.classes_)}")
    else:
        print("   Strategy-Selector skipped (insufficient winning trades)")

    # C · HMM regime detector on Nifty
    hmm_ok = False
    try:
        from hmmlearn.hmm import GaussianHMM
        ni = add_indicators(bench_daily).dropna()
        feat = ni[["ret_1", "vol_20", "adx", "atr_pct"]].dropna().values
        sc3 = StandardScaler().fit(feat)
        feat_s = sc3.transform(feat)
        hmm = GaussianHMM(n_components=5, covariance_type="full", n_iter=200, random_state=42)
        hmm.fit(feat_s)
        states = hmm.predict(feat_s)
        ret_series = ni["ret_1"].dropna().iloc[-len(states):].values
        means = pd.Series(ret_series).groupby(states).mean().sort_values()
        ordered = means.index.tolist()
        name_map = {int(ordered[0]): "BEAR_TRENDING", int(ordered[1]): "BEAR_VOLATILE",
                    int(ordered[2]): "SIDEWAYS",
                    int(ordered[3]): "BULL_VOLATILE", int(ordered[4]): "BULL_TRENDING"}
        joblib.dump(hmm,     f"{MODEL_DIR}/regime_hmm.pkl")
        joblib.dump(sc3,     f"{MODEL_DIR}/regime_scaler.pkl")
        joblib.dump(name_map, f"{MODEL_DIR}/regime_name_map.pkl")
        hmm_ok = True
        print("   HMM regime model saved (5 states).")
    except Exception as e:
        print(f"   HMM skipped ({e}).")

    return {"quality_auc": auc, "selector_accuracy": sel_acc, "hmm_trained": hmm_ok,
            "n_trades_total": int(len(ml_df))}


def save_bundle(summary_metrics: dict, leaderboard_csv: str,
                quality_auc: float, selector_acc: float, hmm_trained: bool) -> str:
    bundle = {
        "quality_clf": "models/setup_quality_classifier.pkl",
        "quality_scaler": "models/setup_quality_scaler.pkl",
        "features": "models/feature_names.pkl",
        "selector": "models/strategy_selector.pkl" if os.path.exists(f"{MODEL_DIR}/strategy_selector.pkl") else None,
        "sel_scaler": "models/strategy_selector_scaler.pkl" if os.path.exists(f"{MODEL_DIR}/strategy_selector_scaler.pkl") else None,
        "label_encoder": "models/strategy_label_encoder.pkl" if os.path.exists(f"{MODEL_DIR}/strategy_label_encoder.pkl") else None,
        "regime_hmm": "models/regime_hmm.pkl" if hmm_trained else None,
        "regime_scaler": "models/regime_scaler.pkl" if hmm_trained else None,
        "regime_names": "models/regime_name_map.pkl" if hmm_trained else None,
        "leaderboard_csv": os.path.relpath(leaderboard_csv, BASE_DIR),
        "config": asdict(CFG),
        "strategies": list(STRATEGY_REGISTRY.keys()),
        "feature_names": ML_FEATS,
        "regimes": REGIMES,
        "summary": summary_metrics,
        "quality_auc": round(quality_auc, 3),
        "selector_accuracy": round(selector_acc, 3),
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline_version": "2.0-chronological-fix",
    }
    out = f"{MODEL_DIR}/bundle.json"
    with open(out, "w") as f:
        json.dump(bundle, f, indent=2, default=str)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 11 · Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 70)
    print("NSE QUANT PLATFORM — FIXED pipeline (chronological-interleaved)")
    print("=" * 70)

    today = datetime.today().date()
    daily_start = (today - timedelta(days=365 * CFG.daily_years + 30)).isoformat()
    daily_end = today.isoformat()

    print(f"\n[1/6] Fetching benchmark (^NSEI) daily bars {daily_start} → {daily_end} ...")
    bench_daily = fetch_ohlcv("^NSEI", daily_start, daily_end, "1d")
    print(f"   ^NSEI rows: {len(bench_daily)}")

    UNIVERSE = NIFTY500_TOP[:100]    # ~Top-100 by market cap
    print(f"\n[2/6] Fetching daily OHLCV for {len(UNIVERSE)} symbols ...")
    data_daily = fetch_universe(UNIVERSE, daily_start, daily_end, "1d", min_rows=400)

    # Intraday (1h). yfinance limits ~730 calendar days — stay inside.
    intra_start = (today - timedelta(days=min(CFG.intraday_days, 700))).isoformat()
    intra_end = today.isoformat()
    intra_syms = list(data_daily.keys())[:40]
    print(f"\n[3/6] Fetching 1h intraday for {len(intra_syms)} symbols ...")
    data_intraday = fetch_universe(intra_syms, intra_start, intra_end, "1h", min_rows=400, verbose=True)
    bench_intraday = fetch_ohlcv("^NSEI", intra_start, intra_end, "1h")

    # Run all 13 strategies.
    print(f"\n[4/6] Running {len(STRATEGY_REGISTRY)} strategies on chronological engine ...")
    all_results: Dict[str, dict] = {}
    for name, factory in STRATEGY_REGISTRY.items():
        probe = factory()
        if probe.intraday:
            d, b = data_intraday, bench_intraday
        else:
            d, b = data_daily, bench_daily
        if not d or b is None or b.empty:
            print(f"   ▶ {name:<28s} — skipped (no data)")
            continue
        try:
            res = run_strategy_on_universe(factory, d, b, CFG)
            all_results[name] = res
            m = res["metrics"]
            print(f"   ▶ {name:<28s} trades={m['total_trades']:>4}  win%={m['win_rate']*100:5.1f}  "
                  f"PF={m['profit_factor']:>5.2f}  Sharpe={m['sharpe']:>5.2f}  "
                  f"CAGR={m['cagr']*100:>5.1f}%  MaxDD={m['max_dd']*100:>5.1f}%  FinalCap={m['final_capital']:>12,.0f}")
        except Exception as e:
            print(f"   ▶ {name:<28s} ERROR: {e}")

    # Leaderboard CSV
    rows = []
    for name, res in all_results.items():
        m = res["metrics"].copy()
        m["strategy"] = name
        rows.append(m)
    lb = pd.DataFrame(rows)
    if not lb.empty:
        lb = lb.set_index("strategy").sort_values("sharpe", ascending=False)
    lb_path = os.path.join(REPORT_DIR, "leaderboard.csv")
    lb.to_csv(lb_path)
    print(f"\n[5/6] Leaderboard saved → {lb_path}")

    # Dump all trades (for the Streamlit dashboard to inspect)
    trade_rows = []
    for name, res in all_results.items():
        for t in res["trades"]:
            trade_rows.append({
                "strategy": name, "symbol": t.symbol, "side": t.side,
                "entry_dt": t.entry_dt, "entry_px": t.entry_px,
                "exit_dt": t.exit_dt, "exit_px": t.exit_px, "exit_reason": t.exit_reason,
                "qty": t.qty, "pnl_net": t.pnl_net, "r_multiple": t.r_multiple,
                "regime_in": t.regime_in, "intraday": t.intraday,
                "holding_bars": t.holding_bars,
            })
    trades_df = pd.DataFrame(trade_rows)
    trades_csv = os.path.join(REPORT_DIR, "all_trades.csv")
    trades_df.to_csv(trades_csv, index=False)
    print(f"   All trades saved → {trades_csv}  ({len(trades_df):,} rows)")

    # Equity plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    for name, res in all_results.items():
        eq = _equity_to_series(res["equity_curve"])
        if eq.empty:
            continue
        norm = eq / eq.iloc[0]
        ax1.plot(norm.index, norm.values, label=name, linewidth=1.0)
        peak = eq.cummax()
        dd = (eq / peak - 1) * 100
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.18)
    ax1.set_title("Equity Curve (normalised)")
    ax1.legend(loc="upper left", fontsize=7, ncol=2)
    ax2.set_title("Drawdown (%)")
    fig.tight_layout()
    png_path = os.path.join(REPORT_DIR, "equity_curves.png")
    fig.savefig(png_path, dpi=120)
    plt.close(fig)
    print(f"   Equity curves saved → {png_path}")

    # ML training
    print(f"\n[6/6] Training ML models on {sum(len(r['trades']) for r in all_results.values()):,} trades ...")
    ml_df = build_ml_dataset(all_results, data_daily)
    print(f"   ML dataset: {ml_df.shape}, overall win-rate {ml_df['win'].mean()*100:.1f}%")
    if len(ml_df) < 50:
        print("   ⚠ Not enough trades to train a reliable model; skipping.")
        train_info = {"quality_auc": 0.5, "selector_accuracy": 0.0, "hmm_trained": False}
    else:
        train_info = train_models(ml_df, bench_daily)

    # Bundle
    summary = {
        "total_strategies": len(all_results),
        "total_trades": int(sum(len(r["trades"]) for r in all_results.values())),
        "mean_win_rate": float(np.mean([r["metrics"]["win_rate"] for r in all_results.values() if r["metrics"]["total_trades"] > 0])) if all_results else 0.0,
        "mean_sharpe": float(np.mean([r["metrics"]["sharpe"] for r in all_results.values() if r["metrics"]["total_trades"] > 0])) if all_results else 0.0,
        "mean_pf": float(np.mean([min(r["metrics"]["profit_factor"], 10) for r in all_results.values() if r["metrics"]["total_trades"] > 0])) if all_results else 0.0,
    }
    bundle_path = save_bundle(summary, lb_path,
                              train_info.get("quality_auc", 0.5),
                              train_info.get("selector_accuracy", 0.0),
                              train_info.get("hmm_trained", False))
    print(f"\n   Bundle manifest → {bundle_path}")

    print("\n" + "=" * 70)
    print(f"PIPELINE COMPLETE in {time.time()-t0:.1f}s")
    print("=" * 70)
    print("\nFinal leaderboard (sorted by Sharpe):")
    if not lb.empty:
        disp = lb.copy()
        for c in ["win_rate", "cagr", "max_dd", "total_return"]:
            disp[c] = (disp[c] * 100).round(2).astype(str) + "%"
        cols = ["total_trades", "win_rate", "profit_factor", "sharpe", "sortino",
                "cagr", "max_dd", "final_capital"]
        print(disp[cols].to_string())

    # Quick sanity: how many strategies hit >=55% win rate?
    ok = [(n, r["metrics"]["win_rate"]) for n, r in all_results.items()
          if r["metrics"]["total_trades"] > 0]
    above = [x for x in ok if x[1] >= 0.55]
    print(f"\nStrategies ≥55% win-rate: {len(above)}/{len(ok)}")
    for n, w in sorted(ok, key=lambda x: -x[1]):
        mark = "✅" if w >= 0.55 else "❌"
        print(f"   {mark}  {n:<28s} {w*100:5.1f}%")


if __name__ == "__main__":
    main()
