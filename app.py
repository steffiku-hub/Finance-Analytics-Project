# ================================================================
# app.py  —  Stock Investment Decision Support
# Features:
#   - Confidence % via predict_proba
#   - Sidebar period comparison
#   - Plain-English signal explanation
#   - Per-feature plain-English explanation cards
#   - Light theme, all English UI
# ================================================================

import pickle
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Investment Decision Support",
    page_icon="📈",
    layout="wide",
)

# ── Global CSS (light-theme) ─────────────────────────────────────
st.markdown("""
<style>
.signal-card {
    background: #f0f4ff;
    border-radius: 8px;
    padding: 12px 16px;
    margin: 8px 0;
}
.signal-card .card-title {
    font-weight: 700;
    font-size: 15px;
    color: #1a1a1a;
    margin-bottom: 4px;
}
.signal-card .card-detail {
    font-size: 13px;
    color: #444;
    line-height: 1.5;
}
.info-box {
    background: #eff6ff;
    border-left: 4px solid #3b82f6;
    border-radius: 6px;
    padding: 12px 16px;
    margin: 8px 0 16px 0;
    color: #1e3a5f;
    font-size: 0.88rem;
}
.info-box b { color: #1d4ed8; }
.signal-plain-buy  { background:#f0fdf4; border-left:4px solid #22c55e; border-radius:6px; padding:14px 18px; margin:12px 0; color:#14532d; }
.signal-plain-hold { background:#fefce8; border-left:4px solid #eab308; border-radius:6px; padding:14px 18px; margin:12px 0; color:#713f12; }
.signal-plain-sell { background:#fef2f2; border-left:4px solid #ef4444; border-radius:6px; padding:14px 18px; margin:12px 0; color:#7f1d1d; }
.signal-plain-buy b, .signal-plain-hold b, .signal-plain-sell b { font-size:1rem; }
.signal-plain-buy span, .signal-plain-hold span, .signal-plain-sell span { font-size:0.9rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────
FEATURE_COLS = [
    "Return_1d", "Return_5d",
    "MA_5", "MA_20", "MA_ratio", "MA_diff", "Price_vs_MA20",
    "Volatility_5", "Volume_MA5", "Volume_Ratio",
    "Momentum_3", "Momentum_10", "Momentum_20",
    "HL_Range", "Volatility_10", "Breakout_20", "Drawdown_20",
    "Volume_Spike",
]

FEATURE_DESCRIBE = {
    "Return_1d":     "1-day price return",
    "Return_5d":     "5-day price return",
    "MA_5":          "5-day moving average level",
    "MA_20":         "20-day moving average level",
    "MA_ratio":      "MA5/MA20 ratio (short vs long trend)",
    "MA_diff":       "MA5 minus MA20 gap",
    "Price_vs_MA20": "Price relative to 20-day MA",
    "Volatility_5":  "5-day return volatility",
    "Volume_MA5":    "5-day average volume",
    "Volume_Ratio":  "Today's volume vs 5-day avg",
    "Momentum_3":    "3-day price momentum",
    "Momentum_10":   "10-day price momentum",
    "Momentum_20":   "20-day price momentum",
    "HL_Range":      "Intraday High-Low range",
    "Volatility_10": "10-day return volatility",
    "Breakout_20":   "Price vs 20-day high (breakout)",
    "Drawdown_20":   "Price vs 20-day low (drawdown)",
    "Volume_Spike":  "Volume spike vs 20-day avg",
}

SIGNAL_LABEL    = { 1: "BUY",  0: "HOLD", -1: "SELL" }
SIGNAL_EMOJI    = { 1: "🟢",   0: "🟡",    -1: "🔴"  }
SIGNAL_COLOR    = { 1: "#22c55e", 0: "#eab308", -1: "#ef4444" }
SIGNAL_CSS_CLASS = { 1: "signal-plain-buy", 0: "signal-plain-hold", -1: "signal-plain-sell" }

SIGNAL_PLAIN = {
    1: (
        "📢 The model suggests this may be a good time to BUY.",
        "Based on technical indicators, this stock is showing strong recent momentum and "
        "an upward moving-average trend, suggesting it may be in the early-to-mid stage of "
        "an uptrend. This is not a guarantee of profit — always factor in your own risk tolerance."
    ),
    0: (
        "⏸️ The model suggests HOLDING — no clear action needed right now.",
        "Current technical indicators show no strong directional bias. "
        "If you already hold a position, it is reasonable to stay put. "
        "If you have not entered yet, consider waiting for a clearer signal."
    ),
    -1: (
        "⚠️ The model suggests considering a SELL or reducing exposure.",
        "Technical indicators show weakening momentum and possible bearish signals "
        "such as a death cross or support breakdown. Short-term downside risk appears elevated. "
        "If you hold a position, consider a stop-loss or trimming your size."
    ),
}

# Chart display periods — controls what is SHOWN, not what is fetched
DISPLAY_PERIODS = ["1wk", "1mo", "3mo", "6mo", "1y"]
DISPLAY_PERIOD_LABEL = {
    "1wk": "1 Week",
    "1mo": "1 Month",
    "3mo": "3 Months",
    "6mo": "6 Months",
    "1y":  "1 Year",
}
DISPLAY_PERIOD_DAYS = {
    "1wk": 7,
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y":  365,
}

# Comparison periods (Period Comparison tab)
PERIODS = ["1mo", "3mo", "6mo", "1y"]
PERIOD_LABEL = {"1mo": "1 Month", "3mo": "3 Months", "6mo": "6 Months", "1y": "1 Year"}

# Always fetch at least 6 months so MA_20 / Momentum_20 / Breakout_20 can be computed
MIN_FETCH_PERIOD = "6mo"

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "TSLA", "JPM", "JNJ", "XOM", "WMT",
    "META", "AMD", "BAC", "GS", "COST",
    "DIS", "CVX", "CAT", "BA", "PFE",
    "SPY", "QQQ",
]

# ── Load model ───────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

model = load_model()
HAS_PROBA      = model is not None and hasattr(model, "predict_proba")
HAS_IMPORTANCE = model is not None and hasattr(model, "feature_importances_")

# ── Feature engineering ──────────────────────────────────────────
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").copy()
    df["Return_1d"]     = df["Close"].pct_change()
    df["Return_5d"]     = df["Close"].pct_change(5)
    df["MA_5"]          = df["Close"].rolling(5).mean()
    df["MA_20"]         = df["Close"].rolling(20).mean()
    df["MA_ratio"]      = df["MA_5"] / df["MA_20"]
    df["MA_diff"]       = df["MA_5"] - df["MA_20"]
    df["Price_vs_MA20"] = df["Close"] / df["MA_20"]
    df["Volatility_5"]  = df["Return_1d"].rolling(5).std()
    df["Volatility_10"] = df["Return_1d"].rolling(10).std()
    df["Volume_MA5"]    = df["Volume"].rolling(5).mean()
    df["Volume_Ratio"]  = df["Volume"] / df["Volume_MA5"]
    df["Volume_MA20"]   = df["Volume"].rolling(20).mean()
    df["Volume_Spike"]  = df["Volume"] / df["Volume_MA20"]
    df["Momentum_3"]    = df["Close"] / df["Close"].shift(3) - 1
    df["Momentum_10"]   = df["Close"] / df["Close"].shift(10) - 1
    df["Momentum_20"]   = df["Close"] / df["Close"].shift(20) - 1
    df["HL_Range"]      = (df["High"] - df["Low"]) / df["Close"]
    df["High_20"]       = df["High"].rolling(20).max()
    df["Low_20"]        = df["Low"].rolling(20).min()
    df["Breakout_20"]   = df["Close"] / df["High_20"]
    df["Drawdown_20"]   = df["Close"] / df["Low_20"]
    return df

# ── Data download ────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_stock(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """
    Always fetches MIN_FETCH_PERIOD of data regardless of the display period,
    so rolling features like MA_20 always have enough history.
    Slicing to the display window happens later via slice_for_display().
    """
    fetch_period = period if period in ("1y", "2y") else MIN_FETCH_PERIOD
    df = yf.download(ticker, period=fetch_period, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    needed = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in needed):
        return pd.DataFrame()
    df = df[needed].reset_index()
    date_col = next((c for c in df.columns if c.lower() in {"date", "datetime", "timestamp"}), None)
    if date_col is None:
        return pd.DataFrame()
    if date_col != "Date":
        df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df.sort_values("Date").reset_index(drop=True)

def slice_for_display(df: pd.DataFrame, display_period: str) -> pd.DataFrame:
    """Trim a fully-featured DataFrame to only the requested display window."""
    days = DISPLAY_PERIOD_DAYS.get(display_period, 180)
    cutoff = df["Date"].max() - pd.Timedelta(days=days)
    return df[df["Date"] >= cutoff].copy()

# ── Confidence calculation ───────────────────────────────────────
def get_confidence(X: np.ndarray, signal: int) -> float:
    if not HAS_PROBA:
        return None
    proba   = model.predict_proba(X)[0]
    classes = list(model.classes_)
    if signal in classes:
        return round(proba[classes.index(signal)] * 100, 1)
    return None

# ── Per-feature plain-English explanation ────────────────────────
def explain_feature(feat: str, val: float) -> str:
    pct = val * 100

    if feat == "MA_ratio":
        direction = "above" if val > 1.0 else "below"
        sentiment = "bullish — short-term trend is rising. 📈" if val > 1.0 else "bearish — short-term trend is falling. 📉"
        return (f"We compare the 5-day average price to the 20-day average price. "
                f"A ratio of {val:.3f} means the short-term average is {direction} the long-term average — {sentiment}")
    if feat == "MA_diff":
        direction = "above" if val > 0 else "below"
        return (f"The 5-day moving average is ${abs(val):.2f} {direction} the 20-day moving average. "
                f"{'A positive gap suggests upward momentum.' if val > 0 else 'A negative gap suggests downward pressure.'}")
    if feat == "Price_vs_MA20":
        rel = "above" if val > 1 else "below"
        pct_diff = abs(val - 1) * 100
        return (f"Think of the 20-day average as the stock's one-month fair value. "
                f"The current price is {pct_diff:.1f}% {rel} this average. "
                f"{'Running above average — momentum is strong. 🔥' if val > 1 else 'Trading below average — may be weak or undervalued. 💡'}")
    if feat == "Return_1d":
        direction = "UP" if val > 0 else "DOWN"
        return (f"The stock moved {direction} {abs(pct):.2f}% yesterday. "
                f"{'Positive momentum from yesterday.' if val > 0 else 'Negative momentum from yesterday.'}")
    if feat == "Return_5d":
        direction = "UP" if val > 0 else "DOWN"
        return (f"Over the past 5 trading days (roughly 1 week), the stock is {direction} {abs(pct):.2f}%. "
                f"{'Short-term strength.' if val > 0 else 'Short-term weakness.'}")
    if feat == "Momentum_3":
        direction = "gained" if val > 0 else "lost"
        return (f"The stock has {direction} {abs(pct):.2f}% over the last 3 days. "
                f"{'Very recent buying pressure.' if val > 0 else 'Very recent selling pressure.'}")
    if feat == "Momentum_10":
        direction = "up" if val > 0 else "down"
        return (f"The stock is {direction} {abs(pct):.2f}% over the last 10 days (2 weeks). "
                f"{'The uptrend has some staying power.' if val > 0 else 'The downtrend has some staying power.'}")
    if feat == "Momentum_20":
        direction = "up" if val > 0 else "down"
        return (f"The stock is {direction} {abs(pct):.2f}% over the last 20 days (~1 month). "
                f"{'Sustained upward trend.' if val > 0 else 'Sustained downward trend.'}")
    if feat == "MA_5":
        return (f"The average closing price over the last 5 trading days is ${val:.2f}. "
                f"Used alongside MA_20 to detect trend direction.")
    if feat == "MA_20":
        return (f"The average closing price over the last 20 trading days (~1 month) is ${val:.2f}. "
                f"Acts as a key support or resistance reference level.")
    if feat == "Volatility_5":
        level = "High" if val > 0.02 else "Low"
        mood  = "moving a lot this week — higher risk, higher reward potential." if val > 0.02 else "relatively calm and stable this week."
        return f"{level} volatility ({abs(pct):.2f}%) — the stock has been {mood}"
    if feat == "Volatility_10":
        level = "High" if val > 0.02 else "Low"
        mood  = "moving a lot over 2 weeks — signals an uncertain market." if val > 0.02 else "stable over 2 weeks."
        return f"{level} volatility ({abs(pct):.2f}%) — the stock has been {mood}"
    if feat == "Volume_MA5":
        return f"The average number of shares traded per day over the last 5 days is {val:,.0f}. Used as a baseline for volume spikes."
    if feat == "Volume_Ratio":
        level = "higher" if val > 1 else "lower"
        return (f"Today's trading volume is {val:.2f}x the 5-day average — {level} than usual. "
                f"{'Elevated volume can confirm a price move.' if val > 1.2 else 'Volume is within normal range.'}")
    if feat == "Volume_Spike":
        return (f"Today's volume is {val:.2f}x the 20-day average. "
                f"{'A significant volume spike — strong market interest. 🔊' if val > 1.5 else 'Volume is within normal range.' if val > 0.8 else 'Low volume — weak market participation. 🔇'}")
    if feat == "HL_Range":
        width = "wide" if val > 0.02 else "narrow"
        return (f"Today's price moved {abs(pct):.2f}% from its low to its high — a {width} intraday range. "
                f"{'Wide range = active, volatile trading session.' if val > 0.02 else 'Narrow range = calm, steady trading today.'}")
    if feat == "Breakout_20":
        return (f"The current price is {val*100:.1f}% of the 20-day high. "
                f"{'Close to the 20-day high — potential breakout. 🚀' if val > 0.97 else 'Well below the 20-day high — momentum is weak.'}")
    if feat == "Drawdown_20":
        return (f"The current price is {val*100:.1f}% of the 20-day low. "
                f"{'Well above the recent low — good support cushion. ✅' if val > 1.05 else 'Close to the 20-day low — near support, watch carefully.'}")
    return f"Value: {val:.4f}"

FEATURE_TITLE = {
    "Return_1d":     lambda v: f"⚡ Yesterday's price change: {v*100:.2f}%",
    "Return_5d":     lambda v: f"📅 Last 5 days price change: {v*100:.2f}%",
    "MA_5":          lambda v: f"📐 5-day average price: ${v:.2f}",
    "MA_20":         lambda v: f"📐 20-day average price: ${v:.2f}",
    "MA_ratio":      lambda v: f"📊 Short vs Long-term Trend: {v:.3f}",
    "MA_diff":       lambda v: f"📏 MA gap (short minus long): ${v:.2f}",
    "Price_vs_MA20": lambda v: f"📍 Price vs 20-day average: {v:.3f}",
    "Volatility_5":  lambda v: f"🌊 5-day price swings: {v*100:.2f}%",
    "Volatility_10": lambda v: f"🌊 10-day price swings: {v*100:.2f}%",
    "Volume_MA5":    lambda v: f"📦 5-day avg volume: {v:,.0f}",
    "Volume_Ratio":  lambda v: f"📦 Volume ratio (today vs 5d avg): {v:.2f}x",
    "Volume_Spike":  lambda v: f"🔊 Volume spike (today vs 20d avg): {v:.2f}x",
    "Momentum_3":    lambda v: f"🚀 3-day momentum: {v*100:.2f}%",
    "Momentum_10":   lambda v: f"🏃 10-day momentum: {v*100:.2f}%",
    "Momentum_20":   lambda v: f"🗓️ 1-month momentum: {v*100:.2f}%",
    "HL_Range":      lambda v: f"📏 Today's High-Low range: {v*100:.2f}%",
    "Breakout_20":   lambda v: f"🚀 Price vs 20-day high: {v*100:.1f}%",
    "Drawdown_20":   lambda v: f"🛡️ Price vs 20-day low: {v*100:.1f}%",
}

# ── Top features by model importance ────────────────────────────
def get_top_features(n: int = 3):
    if not HAS_IMPORTANCE:
        return []
    imp     = model.feature_importances_
    indices = np.argsort(imp)[::-1][:n]
    return [(FEATURE_COLS[i], imp[i]) for i in indices]

# ── Predict ──────────────────────────────────────────────────────
def predict_ticker(ticker: str, period: str = "6mo"):
    raw = fetch_stock(ticker, period=period)
    if raw.empty or len(raw) < 25:
        return None, None, None, None
    df     = compute_features(raw)
    latest = df.dropna(subset=FEATURE_COLS).iloc[-1]
    X      = latest[FEATURE_COLS].values.reshape(1, -1)
    signal = int(model.predict(X)[0])
    conf   = get_confidence(X, signal)
    return signal, conf, latest, df

# ── Reusable UI helpers ──────────────────────────────────────────
def render_signal_plain(signal: int):
    title, body = SIGNAL_PLAIN[signal]
    css = SIGNAL_CSS_CLASS[signal]
    st.markdown(
        f"<div class='{css}'><b>{title}</b><br><span>{body}</span></div>",
        unsafe_allow_html=True,
    )

def render_confidence(confidence: float, color: str):
    if confidence is None:
        return
    if confidence >= 65:
        note, icon = "High confidence — the model is fairly certain about this signal.", "✅"
        note_color = "#166534"
    elif confidence >= 45:
        note, icon = "Moderate confidence — signal is plausible but not strong.", "🟡"
        note_color = "#854d0e"
    else:
        note, icon = "Low confidence — treat with caution — the model is not very sure about this signal.", "🔺"
        note_color = "#991b1b"
    st.markdown("**Model Confidence**")
    st.markdown(
        f"<div style='background:#e5e7eb;border-radius:8px;height:22px;width:100%;overflow:hidden;margin-bottom:4px'>"
        f"<div style='width:{confidence}%;height:22px;background:{color};border-radius:8px;"
        f"display:flex;align-items:center;padding-left:10px'>"
        f"<span style='color:white;font-weight:700;font-size:13px'>{confidence:.1f}%</span></div></div>"
        f"<div style='font-size:13px;color:{note_color};margin-top:2px'>{icon} {note}</div>",
        unsafe_allow_html=True,
    )

def render_top_factors(latest: pd.Series):
    top = get_top_features(3)
    if not top:
        return
    st.markdown("#### 📌 Why this signal? (Top 3 most important factors)")
    for feat_col, importance in top:
        val   = float(latest.get(feat_col, 0))
        title = FEATURE_TITLE.get(feat_col, lambda v: f"{feat_col}: {v:.4f}")(val)
        body  = explain_feature(feat_col, val)
        st.markdown(
            f"<div class='signal-card'>"
            f"<div class='card-title'>{title}</div>"
            f"<div class='card-detail'>{body}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

# ════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════
st.title("📈 Stock Investment Decision Support")
st.markdown("Pre-trained Random Forest → live market data → **Buy / Hold / Sell** signal with confidence & explanation.")

if model is None:
    st.error("⚠️ `model.pkl` not found. Please run `python train_model.py` first, then restart the app.")
    st.stop()
else:
    col_s1, col_s2 = st.columns([3, 1])
    with col_s1:
        st.success("✅ Model loaded from `model.pkl`")
    with col_s2:
        if HAS_PROBA:
            st.info("🎯 Confidence scoring enabled")

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.markdown("📅 **Chart Period**")
    chart_display_period = st.selectbox(
        "chart_period",
        options=DISPLAY_PERIODS,
        index=2,
        format_func=lambda p: DISPLAY_PERIOD_LABEL[p],
        label_visibility="collapsed",
    )

    custom_raw = st.text_input("Add custom tickers (comma-separated)", placeholder="e.g. NFLX, UBER")
    extra      = [t.strip().upper() for t in custom_raw.split(",") if t.strip()]
    all_tickers = list(dict.fromkeys(DEFAULT_TICKERS + extra))

    selected = st.multiselect(
        "Stocks to analyse",
        options=all_tickers,
        default=["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"],
    )

    st.markdown("---")
    st.markdown("📅 **Period Comparison**")
    st.caption("Compare signals across different time windows:")
    compare_periods = st.multiselect(
        "compare_periods",
        options=PERIODS,
        default=["1mo", "3mo", "6mo"],
        format_func=lambda p: PERIOD_LABEL[p],
        label_visibility="collapsed",
    )

    run_btn = st.button("🔍 Get Recommendations", type="primary")

# ── Tabs ─────────────────────────────────────────────────────────
tab_rec, tab_compare, tab_detail, tab_chart = st.tabs([
    "🔮 Recommendations",
    "📅 Period Comparison",
    "🔍 Single Stock Detail",
    "📉 Price Chart",
])

# ════════════════════════════════════════════════════════════════
# Tab 1 — Recommendations
# ════════════════════════════════════════════════════════════════
with tab_rec:
    if not selected:
        st.info("Select stocks in the sidebar and click **Get Recommendations**.")
    elif run_btn or "results" not in st.session_state:
        results  = []
        progress = st.progress(0, text="Fetching data…")
        for i, ticker in enumerate(selected):
            progress.progress((i + 1) / len(selected), text=f"Processing {ticker}…")
            signal, confidence, latest, _ = predict_ticker(ticker)
            if signal is None:
                results.append({"Ticker": ticker, "Close ($)": "—", "Signal": "⚠️ No data", "Confidence": "—"})
                continue
            results.append({
                "Ticker":     ticker,
                "Close ($)":  round(float(latest["Close"]), 2),
                "Signal":     f"{SIGNAL_EMOJI[signal]} {SIGNAL_LABEL[signal]}",
                "Confidence": f"{confidence:.1f}%" if confidence is not None else "N/A",
                "_sig":       signal,
                "_conf":      confidence or 0,
            })
        progress.empty()
        st.session_state["results"] = results

    if "results" in st.session_state:
        res   = st.session_state["results"]
        valid = [r for r in res if "_sig" in r]

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 BUY",  sum(1 for r in valid if r["_sig"] ==  1))
        c2.metric("🟡 HOLD", sum(1 for r in valid if r["_sig"] ==  0))
        c3.metric("🔴 SELL", sum(1 for r in valid if r["_sig"] == -1))
        avg_conf = np.mean([r["_conf"] for r in valid]) if valid else 0
        c4.metric("🎯 Avg Confidence", f"{avg_conf:.1f}%")

        st.markdown("---")
        display = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in res])
        st.dataframe(display, use_container_width=True, hide_index=True)

        if HAS_PROBA and valid:
            st.markdown("#### 🎯 Confidence by Stock")
            fig_c, ax_c = plt.subplots(figsize=(10, 2.5))
            tickers_ = [r["Ticker"] for r in valid]
            confs_   = [r["_conf"]  for r in valid]
            colors_  = [SIGNAL_COLOR[r["_sig"]] for r in valid]
            bars = ax_c.barh(tickers_, confs_, color=colors_, alpha=0.85)
            ax_c.set_xlim(0, 100)
            ax_c.set_xlabel("Confidence (%)")
            ax_c.axvline(50, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            for bar, v in zip(bars, confs_):
                ax_c.text(v + 1, bar.get_y() + bar.get_height() / 2, f"{v:.1f}%", va="center", fontsize=9)
            ax_c.grid(axis="x", alpha=0.2)
            fig_c.tight_layout()
            st.pyplot(fig_c)
            plt.close(fig_c)

# ════════════════════════════════════════════════════════════════
# Tab 2 — Period Comparison
# ════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown("### 📅 Compare Signals Across Different Time Periods")
    st.markdown("See how the model's recommendation changes depending on the data window used.")

    compare_ticker = st.selectbox(
        "Choose a stock to compare",
        options=selected if selected else DEFAULT_TICKERS[:5],
        key="compare_ticker_sel",
    )

    if not compare_periods:
        st.warning("Please select at least one period in the sidebar.")
    elif st.button("🔄 Run Period Comparison", key="compare_btn"):
        period_results = {}
        with st.spinner(f"Fetching {compare_ticker} across {len(compare_periods)} periods…"):
            for period in compare_periods:
                sig, conf, latest, _ = predict_ticker(compare_ticker, period=period)
                period_results[period] = {"signal": sig, "confidence": conf, "latest": latest}
        st.session_state["period_results"]  = period_results
        st.session_state["compare_ticker"]  = compare_ticker

    if "period_results" in st.session_state and st.session_state.get("compare_ticker") == compare_ticker:
        period_results = st.session_state["period_results"]
        cols = st.columns(len(period_results))
        for col, (period, data) in zip(cols, period_results.items()):
            sig  = data["signal"]
            conf = data["confidence"]
            with col:
                if sig is None:
                    st.error(f"**{PERIOD_LABEL[period]}**\n\nNo data")
                else:
                    color     = SIGNAL_COLOR[sig]
                    conf_line = (f"<div style='font-size:1rem;color:#555;margin-top:4px'>"
                                 f"Confidence: {conf:.1f}%</div>") if conf is not None else ""
                    st.markdown(
                        f"<div style='text-align:center;padding:16px;border-radius:10px;"
                        f"background:{color}12;border:2px solid {color};margin-bottom:8px'>"
                        f"<div style='font-size:1.1rem;font-weight:600;color:#555'>{PERIOD_LABEL[period]}</div>"
                        f"<div style='font-size:2.5rem;margin:4px 0'>{SIGNAL_EMOJI[sig]}</div>"
                        f"<div style='font-size:1.4rem;font-weight:700;color:{color}'>{SIGNAL_LABEL[sig]}</div>"
                        + conf_line + "</div>",
                        unsafe_allow_html=True,
                    )
                    if data["latest"] is not None:
                        st.caption(f"Close: ${float(data['latest']['Close']):.2f}")
                        st.caption(f"5d Return: {float(data['latest']['Return_5d'])*100:.2f}%")
                        st.caption(f"MA Ratio: {float(data['latest']['MA_ratio']):.4f}")

        st.markdown("---")
        sigs = [d["signal"] for d in period_results.values() if d["signal"] is not None]
        if sigs:
            if len(set(sigs)) == 1:
                s = sigs[0]
                st.success(f"✅ **Consistent signal**: All periods show {SIGNAL_EMOJI[s]} **{SIGNAL_LABEL[s]}** — this signal is more reliable.")
            else:
                st.warning("⚠️ **Conflicting signals**: Different periods show different signals. The stock may be at a turning point — wait for confirmation before acting.")
                for s in set(sigs):
                    periods_with = [PERIOD_LABEL[p] for p, d in period_results.items() if d["signal"] == s]
                    st.markdown(f"- {SIGNAL_EMOJI[s]} **{SIGNAL_LABEL[s]}**: {', '.join(periods_with)}")

# ════════════════════════════════════════════════════════════════
# Tab 3 — Single Stock Detail
# ════════════════════════════════════════════════════════════════
with tab_detail:
    pick = st.selectbox("Choose a stock", options=selected if selected else DEFAULT_TICKERS[:5])

    if st.button("Analyse", key="analyse_btn"):
        with st.spinner(f"Fetching {pick}…"):
            signal, confidence, latest, df_feat = predict_ticker(pick)

        if signal is None:
            st.error(f"Could not fetch data for **{pick}**. Check the ticker symbol.")
        else:
            color = SIGNAL_COLOR[signal]

            # Signal banner
            conf_text = (f"<div style='font-size:1.1rem;color:{color}99;margin-top:6px'>"
                         f"Confidence: {confidence:.1f}%</div>") if confidence is not None else ""
            st.markdown(
                f"<div style='text-align:center;padding:20px;border-radius:12px;"
                f"background:{color}15;border:2px solid {color}'>"
                f"<span style='font-size:3rem'>{SIGNAL_EMOJI[signal]}</span><br>"
                f"<span style='font-size:2rem;font-weight:700;color:{color}'>{pick}: {SIGNAL_LABEL[signal]}</span>"
                + conf_text + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")

            render_confidence(confidence, color)
            st.markdown("")
            render_signal_plain(signal)
            st.markdown("")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latest Close",    f"${float(latest['Close']):.2f}")
            m2.metric("1-Day Return",    f"{float(latest['Return_1d'])*100:.2f}%")
            m3.metric("5-Day Return",    f"{float(latest['Return_5d'])*100:.2f}%")
            m4.metric("MA Ratio (5/20)", f"{float(latest['MA_ratio']):.4f}")

            st.markdown("---")
            render_top_factors(latest)

            st.markdown("---")
            st.markdown("#### 📊 Feature Values Used for Prediction")
            feat_df = pd.DataFrame({
                "Feature":     FEATURE_COLS,
                "Value":       [round(float(latest[f]), 6) for f in FEATURE_COLS],
                "Description": [FEATURE_DESCRIBE.get(f, "") for f in FEATURE_COLS],
            })
            st.dataframe(feat_df, use_container_width=True, hide_index=True, height=320)

# ════════════════════════════════════════════════════════════════
# Tab 4 — Price Chart
# ════════════════════════════════════════════════════════════════
with tab_chart:
    chart_ticker = st.selectbox(
        "Select stock", options=selected if selected else DEFAULT_TICKERS[:5], key="chart_sel"
    )

    st.markdown(
        "<div class='info-box'>"
        "<b>ℹ️ Why doesn't the display period affect the prediction?</b><br><br>"
        "The model predicts using <b>today's technical indicators</b> (e.g. MA20, Momentum_20). "
        "These indicators need at least 20 trading days of history to compute correctly.<br>"
        "So the app always fetches <b>6 months of data</b> to ensure all indicators are accurate — "
        "but the prediction only looks at the <b>most recent day's values</b>.<br>"
        "The display period you select only controls <b>how much of the chart is shown</b>. "
        "It does not change the signal."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(f"Displaying: **{DISPLAY_PERIOD_LABEL[chart_display_period]}** (change period in the sidebar ←)")

    if st.button("Show Chart", key="chart_btn"):
        with st.spinner(f"Loading {chart_ticker}…"):
            sig, conf, latest, df_feat = predict_ticker(chart_ticker)

        if df_feat is None:
            st.error(f"Could not fetch data for **{chart_ticker}**.")
        else:
            df_plot = slice_for_display(
                df_feat.dropna(subset=["MA_5", "MA_20"]),
                chart_display_period,
            )

            if df_plot.empty:
                st.error("Not enough data for the selected display period. Try a longer window.")
            else:
                period_label = DISPLAY_PERIOD_LABEL[chart_display_period]
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(df_plot["Date"], df_plot["Close"], label="Close", color="#3b82f6", linewidth=1.8)
                ax.plot(df_plot["Date"], df_plot["MA_5"],  label="MA 5",  color="#f97316", linewidth=1.2, linestyle="--")
                ax.plot(df_plot["Date"], df_plot["MA_20"], label="MA 20", color="#a855f7", linewidth=1.2, linestyle="--")

                if chart_display_period == "1wk":
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %b %d"))
                    ax.xaxis.set_major_locator(mdates.DayLocator())
                elif chart_display_period == "1mo":
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
                else:
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
                    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))

                plt.xticks(rotation=30, ha="right")
                ax.set_title(f"{chart_ticker} — Close & Moving Averages ({period_label})")
                ax.set_ylabel("Price (USD)")
                ax.legend()
                ax.grid(alpha=0.25)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                if sig is not None:
                    color    = SIGNAL_COLOR[sig]
                    conf_str = f" &nbsp; Confidence: **{conf:.1f}%**" if conf is not None else ""
                    st.markdown(
                        f"**Signal:** "
                        f"<span style='color:{color};font-size:1.2rem;font-weight:bold'>"
                        f"{SIGNAL_EMOJI[sig]} {SIGNAL_LABEL[sig]}</span>" + conf_str,
                        unsafe_allow_html=True,
                    )
                    render_signal_plain(sig)
                    st.markdown("")
                    render_top_factors(latest)
