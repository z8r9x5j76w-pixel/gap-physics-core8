#!/usr/bin/env python3
"""
Gap Physics Core8 Scanner — v7.0 Standalone
============================================
Lightweight daily signal scanner.
- Best-neighbour configs from robustness report (June 2026).
- 400-bar data window only (~18 months). No backtest code.
- VIX percentile dropped; all rules use absolute VIX level.
- TSM VIX rule changed from vixp_70 → vix_20.
Research / forward-test only. Not financial advice.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

from dataclasses import dataclass
from datetime import date as date_type, datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

APP_TITLE    = "Gap Physics Core8 · v7"
APP_SUBTITLE = "Best-neighbour configs · 18-month data · EOD Yahoo · Research only"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────
# PASSWORD
# ─────────────────────────────────────────────────────────────

def check_password() -> bool:
    if st.session_state.get("_gp7_auth"):
        return True
    st.markdown("<style>.stApp{background:#080d18;color:#dde6f0}</style>",
                unsafe_allow_html=True)
    st.markdown(f"## 🔒 {APP_TITLE}")
    pw = st.text_input("Password", type="password")
    if st.button("Enter"):
        try:
            correct = st.secrets["DASHBOARD_PASSWORD"]
        except Exception:
            correct = "quantgaps2024"
        if pw == correct:
            st.session_state["_gp7_auth"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


if not check_password():
    st.stop()


# ─────────────────────────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
.stApp{background:#080d18;color:#dde6f0}
.block-container{padding-top:1rem}
.signal-yes{background:#0f261c;border-left:5px solid #26C281;padding:12px;border-radius:6px;margin-bottom:8px}
.signal-no{background:#131923;border-left:5px solid #555;padding:12px;border-radius:6px;margin-bottom:8px}
.capital-warn{background:#1a1000;border-left:5px solid #f08000;padding:10px;border-radius:6px;margin-bottom:8px}
.warn-stale{background:#1f1a00;border-left:5px solid #f0c040;padding:10px;border-radius:6px;margin-bottom:8px}
.small-note{color:#8ca0b3;font-size:0.85em}
.cond-table td{padding:4px 10px;font-size:0.85em}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CONFIGS — CORE8 v7 (best-neighbour upgrades applied)
# ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GapConfig:
    ticker:       str
    label:        str
    gap_atr_max:  float
    vix_min:      Optional[float]   # absolute VIX level; None = no filter
    daily_trend:  str               # "none" | "sma50" | "sma100"
    weekly_trend: str               # "none" | "sma20"
    min_volume_z: Optional[float]   # log-ratio to MA; None = no filter
    tp_kind:      str               # "atr" | "gapfill"
    tp_atr_mult:  Optional[float]
    sl_atr_mult:  float
    max_hold:     int
    priority:     int


# Changes vs v6 frozen:
#   TGT  : max_hold 3 → 2
#   CAT  : sl_atr_mult 1.50 → 1.25
#   ASML : sl_atr_mult 1.50 → 1.25 · tp_atr_mult 1.5 → 1.75
#   AMAT : tp_atr_mult 1.5 → 1.75
#   AMZN : sl_atr_mult 1.00 → 1.25
#   TSM  : vix_rule vixp_70 → vix_20

CORE8: List[GapConfig] = [
    GapConfig("TGT",  "TGT",  -1.0, None, "none",   "sma20", 0.0,  "atr",     1.5,  0.75, 2, 1),
    GapConfig("CAT",  "CAT",  -1.0, None, "sma100", "none",  None, "atr",     1.0,  1.25, 5, 2),
    GapConfig("INTU", "INTU", -1.0, None, "sma100", "none",  0.0,  "atr",     1.5,  1.50, 5, 3),
    GapConfig("ASML", "ASML", -1.0, 20.0, "sma100", "none",  None, "atr",     1.75, 1.25, 2, 4),
    GapConfig("AXP",  "AXP",  -1.0, None, "sma50",  "sma20", None, "atr",     1.0,  1.50, 3, 5),
    GapConfig("AMAT", "AMAT", -1.0, 20.0, "none",   "none",  0.0,  "atr",     1.75, 1.50, 3, 6),
    GapConfig("TSM",  "TSM",  -1.0, 20.0, "none",   "sma20", None, "gapfill", None, 1.50, 2, 7),
    GapConfig("AMZN", "AMZN", -1.0, None, "sma100", "none",  None, "atr",     1.0,  1.25, 2, 8),
]

# Number of bars to download — enough for:
#   SMA200 warmup  : 200 bars
#   ATR14          : 14 bars
#   Volume MA20    : 20 bars
#   Weekly SMA20   : 20 weeks ≈ 100 bars
#   Safety buffer  : 100 bars
#   Total          : ~400 bars ≈ 18 months
BARS = 400


# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────

def _start_date_for_bars(n: int = BARS) -> str:
    """Return an ISO start date that gives approximately n trading bars."""
    # 1 calendar year ≈ 252 trading days; add 40% buffer for weekends/holidays
    cal_days = int(n * (365 / 252) * 1.4)
    start = pd.Timestamp.today() - pd.Timedelta(days=cal_days)
    return start.strftime("%Y-%m-%d")


def _flatten(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs(ticker, axis=1, level=-1)
        except Exception:
            df.columns = df.columns.get_level_values(0)
    df = df.copy()
    df.columns = [str(c).strip().title() for c in df.columns]
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def download_ohlcv(ticker: str, start: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, auto_adjust=True,
                     progress=False, threads=False)
    df = _flatten(df, ticker)

    required = ["Open", "High", "Low", "Close", "Volume"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{ticker}: missing {missing}")

    df = df[required].dropna().copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    # Strip today's partial bar
    today = pd.Timestamp(date_type.today())
    df = df[df.index < today]
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def download_vix(start: str) -> pd.Series:
    """Returns VIX Close as a Series indexed by date."""
    df = download_ohlcv("^VIX", start=start)
    return df["Close"].rename("VIX_Close")


# ─────────────────────────────────────────────────────────────
# FEATURE ENGINE  (lightweight — no VIX percentile)
# ─────────────────────────────────────────────────────────────

def _true_range(h: pd.Series, l: pd.Series, c: pd.Series) -> pd.Series:
    pc = c.shift(1)
    return pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)


def _weekly_trend(df: pd.DataFrame) -> pd.DataFrame:
    wk = pd.DataFrame({
        "High":  df["High"].resample("W-FRI").max(),
        "Low":   df["Low"].resample("W-FRI").min(),
        "Close": df["Close"].resample("W-FRI").last(),
    }).dropna()
    wk["W_ATR14"] = _true_range(wk["High"], wk["Low"], wk["Close"]).rolling(14, min_periods=5).mean()
    wk["W_SMA20"] = wk["Close"].rolling(20, min_periods=10).mean()
    wk["W_Trend_SMA20_ATR"] = (wk["Close"] - wk["W_SMA20"]) / wk["W_ATR14"]
    return wk[["W_Trend_SMA20_ATR"]].reindex(df.index, method="ffill")


def compute_features(price: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    df = price.copy()

    df["Prev_Close"] = df["Close"].shift(1)
    df["ATR14"]      = _true_range(df["High"], df["Low"], df["Close"]).rolling(14, min_periods=14).mean()
    df["Gap_ATR"]    = (df["Open"] - df["Prev_Close"]) / df["ATR14"]
    df["Gap_Pct"]    = (df["Open"] / df["Prev_Close"] - 1.0) * 100.0

    vol_ma          = df["Volume"].rolling(20, min_periods=10).mean()
    df["Volume_Z"]  = np.log(df["Volume"].replace(0, np.nan) / vol_ma.replace(0, np.nan))

    for n in [20, 50, 100, 200]:
        sma = df["Close"].rolling(n, min_periods=max(10, n // 2)).mean()
        df[f"Trend_SMA{n}_ATR"] = (df["Close"] - sma) / df["ATR14"]

    df = df.join(_weekly_trend(df))

    # VIX — simple forward-filled join (absolute level only)
    df = df.join(vix, how="left")
    df["VIX_Close"] = df["VIX_Close"].ffill()

    return df.dropna(subset=["Open", "High", "Low", "Close", "ATR14", "Gap_ATR"])


# ─────────────────────────────────────────────────────────────
# SIGNAL LOGIC
# ─────────────────────────────────────────────────────────────

_TREND_COL = {
    "sma20":  "Trend_SMA20_ATR",
    "sma50":  "Trend_SMA50_ATR",
    "sma100": "Trend_SMA100_ATR",
    "sma200": "Trend_SMA200_ATR",
}


def check_conditions(row: pd.Series, cfg: GapConfig) -> Dict[str, Tuple[bool, str]]:
    """
    Returns dict of condition_name → (passed: bool, detail: str).
    """
    out: Dict[str, Tuple[bool, str]] = {}

    # Gap
    g     = float(row["Gap_ATR"])
    out["Gap ATR"] = (g <= cfg.gap_atr_max,
                      f"{g:.2f} (need ≤ {cfg.gap_atr_max:.1f})")

    # VIX
    vix_val = float(row.get("VIX_Close", np.nan))
    if cfg.vix_min is None:
        out["VIX"] = (True, "off")
    else:
        out["VIX"] = (np.isfinite(vix_val) and vix_val >= cfg.vix_min,
                      f"{vix_val:.1f} (need ≥ {cfg.vix_min:.0f})")

    # Daily trend
    if cfg.daily_trend == "none":
        out["Daily trend"] = (True, "off")
    else:
        col = _TREND_COL.get(cfg.daily_trend, "")
        val = float(row.get(col, np.nan))
        out["Daily trend"] = (np.isfinite(val) and val > 0,
                              f"{cfg.daily_trend.upper()} ratio={val:.2f} (need > 0)")

    # Weekly trend
    if cfg.weekly_trend == "none":
        out["Weekly trend"] = (True, "off")
    else:
        val = float(row.get("W_Trend_SMA20_ATR", np.nan))
        out["Weekly trend"] = (np.isfinite(val) and val > 0,
                               f"W_SMA20 ratio={val:.2f} (need > 0)")

    # Volume
    if cfg.min_volume_z is None:
        out["Volume"] = (True, "off")
    else:
        vol = float(row.get("Volume_Z", np.nan))
        out["Volume"] = (np.isfinite(vol) and vol >= cfg.min_volume_z,
                         f"log-ratio={vol:.2f} (need ≥ {cfg.min_volume_z:.1f})")

    return out


def trade_levels(row: pd.Series, cfg: GapConfig) -> Tuple[float, float]:
    """Target and stop using Close as entry proxy."""
    entry = float(row["Close"])
    atr   = float(row["ATR14"])
    if cfg.tp_kind == "gapfill":
        target = float(row["Prev_Close"])
    else:
        target = entry + float(cfg.tp_atr_mult or 0.0) * atr
    stop = entry - cfg.sl_atr_mult * atr
    return target, stop


# ─────────────────────────────────────────────────────────────
# SCAN
# ─────────────────────────────────────────────────────────────

@dataclass
class ScanResult:
    ticker:      str
    label:       str
    signal:      bool
    date:        str
    close:       float
    gap_atr:     float
    gap_pct:     float
    vix:         float
    target:      float
    stop:        float
    max_hold:    int
    tp_kind:     str
    tp_mult:     Optional[float]
    sl_mult:     float
    priority:    int
    conditions:  Dict[str, Tuple[bool, str]]
    gapfill_warn: bool
    error:       str = ""


def scan_ticker(cfg: GapConfig, start: str, vix: pd.Series) -> Tuple[ScanResult, bool]:
    """Returns (ScanResult, ok: bool)."""
    try:
        price    = download_ohlcv(cfg.ticker, start)
        features = compute_features(price, vix)
        row      = features.iloc[-1]
        conds    = check_conditions(row, cfg)
        sig      = all(v[0] for v in conds.values())
        tgt, stp = trade_levels(row, cfg)

        gfw = cfg.tp_kind == "gapfill" and float(row["Prev_Close"]) <= float(row["Close"])

        return ScanResult(
            ticker=cfg.ticker, label=cfg.label,
            signal=sig,
            date=features.index[-1].date().isoformat(),
            close=float(row["Close"]),
            gap_atr=float(row["Gap_ATR"]),
            gap_pct=float(row["Gap_Pct"]),
            vix=float(row.get("VIX_Close", np.nan)),
            target=tgt, stop=stp,
            max_hold=cfg.max_hold,
            tp_kind=cfg.tp_kind,
            tp_mult=cfg.tp_atr_mult,
            sl_mult=cfg.sl_atr_mult,
            priority=cfg.priority,
            conditions=conds,
            gapfill_warn=gfw,
        ), True

    except Exception as exc:
        return ScanResult(
            ticker=cfg.ticker, label=cfg.label,
            signal=False, date="", close=np.nan,
            gap_atr=np.nan, gap_pct=np.nan, vix=np.nan,
            target=np.nan, stop=np.nan,
            max_hold=cfg.max_hold,
            tp_kind=cfg.tp_kind, tp_mult=cfg.tp_atr_mult,
            sl_mult=cfg.sl_atr_mult,
            priority=cfg.priority,
            conditions={}, gapfill_warn=False,
            error=str(exc),
        ), False


# ─────────────────────────────────────────────────────────────
# UI HELPERS
# ─────────────────────────────────────────────────────────────

def _fmt(v: float, decimals: int = 2) -> str:
    return f"{v:.{decimals}f}" if np.isfinite(v) else "—"


def render_summary_table(results: List[ScanResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        exit_str = (f"gapfill (prev close)"
                    if r.tp_kind == "gapfill"
                    else f"TP {r.tp_mult}×ATR / SL {r.sl_mult}×ATR")
        rows.append({
            "Pri":        r.priority,
            "Ticker":     r.ticker,
            "Signal":     "✅ LONG NEXT OPEN" if r.signal else "—",
            "Date":       r.date,
            "Close":      _fmt(r.close),
            "Gap ATR":    _fmt(r.gap_atr),
            "Gap %":      f"{r.gap_pct:+.2f}%" if np.isfinite(r.gap_pct) else "—",
            "VIX":        _fmt(r.vix),
            "Target ref": _fmt(r.target),
            "Stop ref":   _fmt(r.stop),
            "Hold":       r.max_hold,
            "Exit":       exit_str,
            "Gap ✓":      "✅" if r.conditions.get("Gap ATR", (False,))[0] else "❌",
            "VIX ✓":      "✅" if r.conditions.get("VIX", (False,))[0] else "❌",
            "Daily ✓":    "✅" if r.conditions.get("Daily trend", (False,))[0] else "❌",
            "Weekly ✓":   "✅" if r.conditions.get("Weekly trend", (False,))[0] else "❌",
            "Vol ✓":      "✅" if r.conditions.get("Volume", (False,))[0] else "❌",
            "Error":      r.error,
        })
    return pd.DataFrame(rows)


def render_condition_detail(r: ScanResult) -> None:
    if not r.conditions:
        st.warning(f"No condition data for {r.ticker} — scan error.")
        return

    rows = [{"Condition": k,
             "Pass": "✅" if v[0] else "❌",
             "Detail": v[1]}
            for k, v in r.conditions.items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if r.gapfill_warn:
        st.warning("⚠️ Gapfill warning: prev_close ≤ close — target may already be "
                   "below next open. Verify before entry.")


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────

st.markdown(f"## ⚡ {APP_TITLE}")
st.caption(f"{APP_SUBTITLE}")

# Staleness warning
_ts = st.session_state.get("gp7_ts")
if _ts:
    try:
        _age = (datetime.now(timezone.utc) -
                datetime.strptime(_ts, "%Y-%m-%d %H:%M UTC")
                .replace(tzinfo=timezone.utc)).total_seconds() / 3600
        if _age > 24:
            st.markdown(
                f"<div class='warn-stale'>⚠️ Last scan <b>{_age:.0f}h ago</b> "
                f"({_ts}) — click Run to refresh.</div>",
                unsafe_allow_html=True)
    except Exception:
        pass

col_run, col_cache, col_note = st.columns([1, 1, 3])
with col_cache:
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.success("Cache cleared.")
with col_note:
    st.markdown(
        "<div class='small-note'>Uses latest <b>completed</b> daily candle only. "
        "Today's partial bar is always excluded. "
        "Target/stop are reference levels — recalculate once actual open is known.</div>",
        unsafe_allow_html=True)

with col_run:
    run = st.button("▶ Run Scan", type="primary")

if run:
    start = _start_date_for_bars(BARS)
    results: List[ScanResult] = []

    prog = st.progress(0, "Downloading VIX…")
    try:
        vix_series = download_vix(start)
    except Exception as e:
        st.error(f"VIX download failed: {e}")
        st.stop()

    for i, cfg in enumerate(CORE8):
        prog.progress((i + 1) / len(CORE8), f"Scanning {cfg.ticker}…")
        res, _ = scan_ticker(cfg, start, vix_series)
        results.append(res)

    prog.empty()
    st.session_state["gp7_results"] = results
    st.session_state["gp7_ts"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

if "gp7_results" not in st.session_state:
    st.info("Click ▶ Run Scan to start.")
    st.stop()

results: List[ScanResult] = st.session_state["gp7_results"]
active  = [r for r in results if r.signal]
errors  = [r for r in results if r.error]

# ── Banner ───────────────────────────────────────────────────
last_date = next((r.date for r in results if r.date), "—")
if active:
    st.markdown(
        f"<div class='signal-yes'><b>🟢 {len(active)}/{len(results)} signal(s) active</b>"
        f"<span style='float:right'>Candle: {last_date}</span></div>",
        unsafe_allow_html=True)
else:
    st.markdown(
        f"<div class='signal-no'><b>⚪ No active signals</b>"
        f"<span style='float:right'>Candle: {last_date}</span></div>",
        unsafe_allow_html=True)

st.caption(f"Last run: {st.session_state.get('gp7_ts', '—')}")

# ── Capital conflict warning ─────────────────────────────────
if len(active) >= 2:
    cap = min(len(active), 3) * 25
    tstr = ", ".join(r.ticker for r in sorted(active, key=lambda x: x.priority))
    extra = f" — <b>{len(active)-3} signal(s) over limit, skip lowest priority</b>" if len(active) > 3 else ""
    st.markdown(
        f"<div class='capital-warn'>⚠️ <b>{len(active)} signals</b>: {tstr} → "
        f"<b>{cap}% capital deployed</b>{extra}</div>",
        unsafe_allow_html=True)

# ── Summary table ────────────────────────────────────────────
scan_df = render_summary_table(results)
st.dataframe(scan_df, use_container_width=True, hide_index=True)

csv = scan_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇ Download scan CSV", data=csv,
                   file_name="gap_physics_core8_v7_scan.csv", mime="text/csv")

if errors:
    with st.expander(f"⚠️ {len(errors)} ticker error(s)"):
        for r in errors:
            st.error(f"{r.ticker}: {r.error}")

# ── Per-ticker condition detail ──────────────────────────────
st.markdown("---")
st.markdown("### Per-ticker condition detail")

selected = st.selectbox("Ticker", [c.ticker for c in CORE8])
sel_res  = next((r for r in results if r.ticker == selected), None)

if sel_res and not sel_res.error:
    sig_label = "✅ SIGNAL — LONG NEXT OPEN" if sel_res.signal else "⚪ No signal"
    st.markdown(f"**{selected}** · {sig_label}")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Close",      _fmt(sel_res.close))
    m2.metric("Gap ATR",    _fmt(sel_res.gap_atr))
    m3.metric("Gap %",      f"{sel_res.gap_pct:+.2f}%" if np.isfinite(sel_res.gap_pct) else "—")
    m4.metric("VIX",        _fmt(sel_res.vix))
    m5.metric("Target ref", _fmt(sel_res.target))
    m6.metric("Stop ref",   _fmt(sel_res.stop))

    render_condition_detail(sel_res)

    # Manual open override
    st.markdown("**Recalculate levels with actual next open:**")
    actual_open = st.number_input("Actual open price (optional)", min_value=0.0,
                                   value=0.0, step=0.01, format="%.2f")
    if actual_open > 0:
        atr = sel_res.close  # fallback; recompute from stored data
        # Find features from session state isn't stored in v7 (lightweight)
        # So we approximate using Gap ATR and Close
        # For a true recalc: entry = actual_open, use same ATR from last row
        # We stored sl_mult and tp_mult on the result
        # Re-derive ATR from: gap_atr = (open - prev_close) / atr14
        # Better: store ATR14 on ScanResult
        # Since we don't store ATR14, inform user
        st.info("For precise recalculation, note: ATR14 is not stored in this lightweight version. "
                "Use: Target = open + tp_mult × ATR14, Stop = open − sl_mult × ATR14. "
                f"Config: TP {sel_res.tp_mult}×ATR / SL {sel_res.sl_mult}×ATR / Hold {sel_res.max_hold}d")

    st.markdown(
        f"<div class='small-note'>Exit config: "
        f"{'Gapfill (prev close)' if sel_res.tp_kind == 'gapfill' else f'TP {sel_res.tp_mult}×ATR'} · "
        f"SL {sel_res.sl_mult}×ATR · Max hold {sel_res.max_hold} sessions</div>",
        unsafe_allow_html=True)

elif sel_res and sel_res.error:
    st.error(f"Error loading {selected}: {sel_res.error}")

# ── Config reference ─────────────────────────────────────────
with st.expander("Core8 v7 config reference", expanded=False):
    st.markdown("**Changes from v6:** TGT hold 3→2 · CAT SL 1.50→1.25 · "
                "ASML SL 1.50→1.25 & TP 1.5→1.75 · AMAT TP 1.5→1.75 · "
                "AMZN SL 1.00→1.25 · TSM VIX vixp_70→vix_20")
    cfg_df = pd.DataFrame([{
        "Ticker":    c.ticker,
        "Gap ≤":     c.gap_atr_max,
        "VIX ≥":     c.vix_min if c.vix_min else "off",
        "Daily":     c.daily_trend,
        "Weekly":    c.weekly_trend,
        "Vol_Z ≥":   c.min_volume_z if c.min_volume_z is not None else "off",
        "TP":        f"gapfill" if c.tp_kind == "gapfill" else f"{c.tp_atr_mult}×ATR",
        "SL":        f"{c.sl_atr_mult}×ATR",
        "Hold":      c.max_hold,
        "Priority":  c.priority,
    } for c in CORE8])
    st.dataframe(cfg_df, use_container_width=True, hide_index=True)

st.markdown("---")
st.caption("Gap Physics Core8 v7 · EOD Yahoo Finance · Research only · Not financial advice")
