# ================================================================
# app.py  —  Stock Investment Decision Support (Enhanced)
# New features:
#   - Confidence % via predict_proba
#   - Sidebar period comparison (1mo / 3mo / 6mo / 1y)
#   - Decision explanation (key indicators driving the signal)
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

# ── Constants ────────────────────────────────────────────────────
FEATURE_COLS = [
    "Return_1d", "Return_5d",
    "MA_5", "MA_20", "MA_ratio", "MA_diff", "Price_vs_MA20",
    "Volatility_5", "Volume_MA5", "Volume_Ratio",
    "Momentum_3", "Momentum_10", "Momentum_20",
    "HL_Range", "Volatility_10", "Breakout_20", "Drawdown_20",
    "Volume_Spike",
]

# Human-readable feature names for explanations
FEATURE_NAMES = {
    "Return_1d":     "1日回報率",
    "Return_5d":     "5日回報率",
    "MA_5":          "5日移動平均",
    "MA_20":         "20日移動平均",
    "MA_ratio":      "短/長期均線比率 (5/20)",
    "MA_diff":       "均線差距 (MA5 - MA20)",
    "Price_vs_MA20": "股價 vs 20日均線",
    "Volatility_5":  "5日波動率",
    "Volume_MA5":    "5日成交量均值",
    "Volume_Ratio":  "成交量比率",
    "Momentum_3":    "3日動量",
    "Momentum_10":   "10日動量",
    "Momentum_20":   "20日動量",
    "HL_Range":      "當日高低幅度",
    "Volatility_10": "10日波動率",
    "Breakout_20":   "20日突破指標",
    "Drawdown_20":   "20日回撤指標",
    "Volume_Spike":  "成交量突破",
}

SIGNAL_LABEL = { 1: "BUY",  0: "HOLD", -1: "SELL" }
SIGNAL_EMOJI = { 1: "🟢",   0: "🟡",    -1: "🔴"  }
SIGNAL_COLOR = { 1: "#22c55e", 0: "#eab308", -1: "#ef4444" }

PERIODS = ["1mo", "3mo", "6mo", "1y"]
PERIOD_LABEL = {"1mo": "1 個月", "3mo": "3 個月", "6mo": "6 個月", "1y": "1 年"}

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

# Check if model supports predict_proba (Random Forest does)
HAS_PROBA = model is not None and hasattr(model, "predict_proba")
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
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df[["Open", "High", "Low", "Close", "Volume"]].reset_index()
    date_col = next((c for c in df.columns if c.lower() in {"date", "datetime"}), None)
    if date_col and date_col != "Date":
        df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    return df.sort_values("Date").reset_index(drop=True)

# ── Confidence calculation ───────────────────────────────────────
def get_confidence(X: np.ndarray, signal: int) -> float:
    """Return confidence % for the predicted signal using predict_proba."""
    if not HAS_PROBA:
        return None
    proba = model.predict_proba(X)[0]
    classes = list(model.classes_)
    if signal in classes:
        idx = classes.index(signal)
        return round(proba[idx] * 100, 1)
    return None

# ── Decision explanation ─────────────────────────────────────────
def generate_explanation(signal: int, latest: pd.Series) -> str:
    """Generate a human-readable explanation for the signal."""
    reasons = []

    # Momentum analysis
    m3  = latest.get("Momentum_3",  0)
    m10 = latest.get("Momentum_10", 0)
    m20 = latest.get("Momentum_20", 0)
    if signal == 1:
        if m10 > 0.03:
            reasons.append(f"📈 10日動量強勁（+{m10*100:.1f}%），顯示近期上升趨勢")
        if m20 > 0.05:
            reasons.append(f"📈 20日動量優秀（+{m20*100:.1f}%），中期走勢正面")
    elif signal == -1:
        if m10 < -0.03:
            reasons.append(f"📉 10日動量疲弱（{m10*100:.1f}%），近期下跌壓力")
        if m20 < -0.05:
            reasons.append(f"📉 20日動量惡化（{m20*100:.1f}%），中期趨勢向下")

    # Moving average analysis
    ma_ratio = latest.get("MA_ratio", 1.0)
    ma_diff  = latest.get("MA_diff",  0)
    if signal == 1 and ma_ratio > 1.01:
        reasons.append(f"📊 短期均線（MA5）高於長期均線（MA20）{(ma_ratio-1)*100:.2f}%，黃金交叉信號")
    elif signal == -1 and ma_ratio < 0.99:
        reasons.append(f"📊 短期均線（MA5）低於長期均線（MA20）{(1-ma_ratio)*100:.2f}%，死亡交叉信號")

    # Price vs MA20
    pma = latest.get("Price_vs_MA20", 1.0)
    if signal == 1 and pma > 1.02:
        reasons.append(f"💪 股價高於20日均線 {(pma-1)*100:.1f}%，強勢突破均線壓力")
    elif signal == -1 and pma < 0.98:
        reasons.append(f"⚠️ 股價低於20日均線 {(1-pma)*100:.1f}%，跌破均線支撐")

    # Breakout analysis
    breakout = latest.get("Breakout_20", 1.0)
    if signal == 1 and breakout > 0.97:
        reasons.append(f"🚀 股價接近20日高點（{breakout*100:.1f}%），突破形態良好")
    elif signal == -1 and breakout < 0.90:
        reasons.append(f"🔻 股價遠離20日高點（僅{breakout*100:.1f}%），技術形態偏弱")

    # Volume analysis
    vol_ratio = latest.get("Volume_Ratio",  1.0)
    vol_spike = latest.get("Volume_Spike",  1.0)
    if vol_spike > 1.5:
        reasons.append(f"🔊 成交量激增（{vol_spike:.1f}x 20日均量），資金活躍度高")
    elif vol_spike < 0.7:
        reasons.append(f"🔇 成交量萎縮（僅{vol_spike:.1f}x 20日均量），市場興趣不足")

    # Volatility
    vol5 = latest.get("Volatility_5", 0)
    if vol5 > 0.03:
        reasons.append(f"⚡ 5日波動率偏高（{vol5*100:.2f}%），市場波動較大，風險注意")

    # Return analysis
    r1 = latest.get("Return_1d", 0)
    r5 = latest.get("Return_5d", 0)
    if signal == 1:
        if r5 > 0.02:
            reasons.append(f"✅ 近5日累計回報 +{r5*100:.1f}%，短期表現亮眼")
    elif signal == -1:
        if r5 < -0.02:
            reasons.append(f"❌ 近5日累計回報 {r5*100:.1f}%，短期持續走弱")

    # Fallback
    if not reasons:
        if signal == 1:
            reasons.append("✅ 綜合技術指標顯示買入信號，但強度一般，建議謹慎操作")
        elif signal == -1:
            reasons.append("⚠️ 綜合技術指標顯示賣出信號，但強度一般，建議謹慎操作")
        else:
            reasons.append("⏸️ 各項指標無明顯偏向，建議觀望，等待更清晰信號")

    return reasons

# ── Feature importance top drivers ──────────────────────────────
def get_top_features(n=5):
    """Return top N feature names by model importance."""
    if not HAS_IMPORTANCE:
        return []
    imp = model.feature_importances_
    indices = np.argsort(imp)[::-1][:n]
    return [(FEATURE_COLS[i], FEATURE_NAMES.get(FEATURE_COLS[i], FEATURE_COLS[i]), imp[i]) for i in indices]

# ── Predict for one ticker ───────────────────────────────────────
def predict_ticker(ticker: str, period: str = "6mo"):
    """Returns (signal_int, confidence_pct, latest_row_series, full_df_with_features)"""
    raw = fetch_stock(ticker, period=period)
    if raw.empty or len(raw) < 25:
        return None, None, None, None
    df = compute_features(raw)
    latest = df.dropna(subset=FEATURE_COLS).iloc[-1]
    X = latest[FEATURE_COLS].values.reshape(1, -1)
    signal = int(model.predict(X)[0])
    confidence = get_confidence(X, signal)
    return signal, confidence, latest, df

# ════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════

st.title("📈 Stock Investment Decision Support")
st.markdown("Pre-trained Random Forest → live market data → **Buy / Hold / Sell** signal with confidence & explanation.")

# ── Model status banner ──────────────────────────────────────────
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

    custom_raw = st.text_input("Add custom tickers (comma-separated)", placeholder="e.g. NFLX, UBER")
    extra = [t.strip().upper() for t in custom_raw.split(",") if t.strip()]
    all_tickers = list(dict.fromkeys(DEFAULT_TICKERS + extra))

    selected = st.multiselect(
        "Stocks to analyse",
        options=all_tickers,
        default=["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"],
    )

    st.markdown("---")
    st.subheader("📅 Period Comparison")
    st.markdown("Compare signals across different time windows:")
    compare_periods = st.multiselect(
        "Select periods to compare",
        options=PERIODS,
        default=["1mo", "3mo", "6mo"],
        format_func=lambda p: PERIOD_LABEL[p],
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
# Tab 1 — Recommendations (batch)
# ════════════════════════════════════════════════════════════════
with tab_rec:
    if not selected:
        st.info("Select stocks in the sidebar and click **Get Recommendations**.")
    elif run_btn or "results" not in st.session_state:
        if not selected:
            st.stop()
        results = []
        progress = st.progress(0, text="Fetching data…")
        for i, ticker in enumerate(selected):
            progress.progress((i + 1) / len(selected), text=f"Processing {ticker}…")
            signal, confidence, latest, _ = predict_ticker(ticker, period="6mo")
            if signal is None:
                results.append({"Ticker": ticker, "Close ($)": "—", "Signal": "⚠️ No data", "Confidence": "—"})
                continue
            conf_str = f"{confidence:.1f}%" if confidence is not None else "N/A"
            results.append({
                "Ticker":       ticker,
                "Close ($)":    round(float(latest["Close"]), 2),
                "Signal":       f"{SIGNAL_EMOJI[signal]} {SIGNAL_LABEL[signal]}",
                "Confidence":   conf_str,
                "_sig":         signal,
                "_conf":        confidence or 0,
            })
        progress.empty()
        st.session_state["results"] = results
        st.session_state["selected"] = selected

    if "results" in st.session_state:
        res = st.session_state["results"]
        valid = [r for r in res if "_sig" in r]

        # Summary cards
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🟢 BUY",  sum(1 for r in valid if r["_sig"] ==  1))
        c2.metric("🟡 HOLD", sum(1 for r in valid if r["_sig"] ==  0))
        c3.metric("🔴 SELL", sum(1 for r in valid if r["_sig"] == -1))
        avg_conf = np.mean([r["_conf"] for r in valid]) if valid else 0
        c4.metric("🎯 Avg Confidence", f"{avg_conf:.1f}%")

        st.markdown("---")

        display = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in res])
        st.dataframe(display, use_container_width=True, hide_index=True)

        # Confidence distribution bar chart
        if HAS_PROBA and valid:
            st.markdown("#### 🎯 Confidence Distribution")
            fig_c, ax_c = plt.subplots(figsize=(10, 2.5))
            tickers_  = [r["Ticker"] for r in valid]
            confs_    = [r["_conf"] for r in valid]
            colors_   = [SIGNAL_COLOR[r["_sig"]] for r in valid]
            bars = ax_c.barh(tickers_, confs_, color=colors_, alpha=0.85)
            ax_c.set_xlim(0, 100)
            ax_c.set_xlabel("Confidence (%)")
            ax_c.axvline(50, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            for bar, val in zip(bars, confs_):
                ax_c.text(val + 1, bar.get_y() + bar.get_height()/2,
                          f"{val:.1f}%", va="center", fontsize=9)
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
                signal, confidence, latest, _ = predict_ticker(compare_ticker, period=period)
                period_results[period] = {
                    "signal": signal,
                    "confidence": confidence,
                    "latest": latest,
                }
        st.session_state["period_results"] = period_results
        st.session_state["compare_ticker"] = compare_ticker

    if "period_results" in st.session_state and st.session_state.get("compare_ticker") == compare_ticker:
        period_results = st.session_state["period_results"]

        cols = st.columns(len(period_results))
        for col, (period, data) in zip(cols, period_results.items()):
            sig = data["signal"]
            conf = data["confidence"]
            with col:
                if sig is None:
                    st.error(f"**{PERIOD_LABEL[period]}**\n\nNo data")
                else:
                    color = SIGNAL_COLOR[sig]
                    st.markdown(
                        f"<div style='text-align:center;padding:16px;border-radius:10px;"
                        f"background:{color}18;border:2px solid {color};margin-bottom:8px'>"
                        f"<div style='font-size:1.1rem;font-weight:600;color:#888'>{PERIOD_LABEL[period]}</div>"
                        f"<div style='font-size:2.5rem;margin:4px 0'>{SIGNAL_EMOJI[sig]}</div>"
                        f"<div style='font-size:1.4rem;font-weight:700;color:{color}'>{SIGNAL_LABEL[sig]}</div>"
                        + (f"<div style='font-size:1rem;color:#aaa;margin-top:4px'>信心度: {conf:.1f}%</div>" if conf is not None else "")
                        + "</div>",
                        unsafe_allow_html=True,
                    )
                    latest = data["latest"]
                    if latest is not None:
                        st.caption(f"Close: ${float(latest['Close']):.2f}")
                        st.caption(f"Return 5d: {float(latest['Return_5d'])*100:.2f}%")
                        st.caption(f"MA Ratio: {float(latest['MA_ratio']):.4f}")

        # Consistency analysis
        st.markdown("---")
        sigs = [d["signal"] for d in period_results.values() if d["signal"] is not None]
        if sigs:
            if len(set(sigs)) == 1:
                s = sigs[0]
                st.success(f"✅ **一致信號**: 所有時間段均顯示 {SIGNAL_EMOJI[s]} **{SIGNAL_LABEL[s]}**，信號較為可靠。")
            else:
                st.warning("⚠️ **信號不一致**: 不同時間段顯示不同信號，市場可能處於轉折點，建議謹慎操作並等待確認。")
                unique_sigs = set(sigs)
                for s in unique_sigs:
                    periods_with_sig = [PERIOD_LABEL[p] for p, d in period_results.items() if d["signal"] == s]
                    st.markdown(f"- {SIGNAL_EMOJI[s]} **{SIGNAL_LABEL[s]}**: {', '.join(periods_with_sig)}")

# ════════════════════════════════════════════════════════════════
# Tab 3 — Single Stock Detail
# ════════════════════════════════════════════════════════════════
with tab_detail:
    pick = st.selectbox("Choose a stock", options=selected if selected else DEFAULT_TICKERS[:5])
    detail_period = st.selectbox(
        "Analysis period",
        options=PERIODS,
        index=2,
        format_func=lambda p: PERIOD_LABEL[p],
        key="detail_period_sel",
    )

    if st.button("Analyse", key="analyse_btn"):
        with st.spinner(f"Fetching {pick}…"):
            signal, confidence, latest, df_feat = predict_ticker(pick, period=detail_period)

        if signal is None:
            st.error(f"Could not fetch data for **{pick}**. Check the ticker symbol.")
        else:
            color = SIGNAL_COLOR[signal]
            label = SIGNAL_LABEL[signal]
            emoji = SIGNAL_EMOJI[signal]

            # Signal + confidence card
            conf_text = f"<div style='font-size:1.1rem;color:{color}88;margin-top:6px'>信心度: {confidence:.1f}%</div>" if confidence is not None else ""
            st.markdown(
                f"<div style='text-align:center;padding:20px;border-radius:12px;"
                f"background:{color}22;border:2px solid {color}'>"
                f"<span style='font-size:3rem'>{emoji}</span><br>"
                f"<span style='font-size:2rem;font-weight:700;color:{color}'>{pick}: {label}</span>"
                + conf_text +
                f"</div>",
                unsafe_allow_html=True,
            )

            # Confidence meter
            if confidence is not None:
                st.markdown("")
                st.markdown(f"**🎯 信心度: {confidence:.1f}%**")
                st.progress(int(confidence))

            st.markdown("")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Latest Close",     f"${latest['Close']:.2f}")
            m2.metric("1-Day Return",     f"{latest['Return_1d']*100:.2f}%")
            m3.metric("5-Day Return",     f"{latest['Return_5d']*100:.2f}%")
            m4.metric("MA Ratio (5/20)",  f"{latest['MA_ratio']:.4f}")

            # ── Decision Explanation ─────────────────────────────────
            st.markdown("---")
            st.markdown("#### 💡 決策解釋")
            reasons = generate_explanation(signal, latest)
            for reason in reasons:
                st.markdown(f"- {reason}")

            # Top model features
            if HAS_IMPORTANCE:
                st.markdown("---")
                st.markdown("#### 🔬 模型最重要指標 (Top 5)")
                top_feats = get_top_features(5)
                for feat_col, feat_name, importance in top_feats:
                    val = float(latest.get(feat_col, 0))
                    st.markdown(
                        f"**{feat_name}** &nbsp; `{val:.4f}` &nbsp; "
                        f"<span style='color:#888'>（模型權重: {importance*100:.1f}%）</span>",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")
            st.markdown("#### 📊 All Feature Values")
            feat_df = pd.DataFrame({
                "Feature (EN)": FEATURE_COLS,
                "Feature (中文)": [FEATURE_NAMES.get(f, f) for f in FEATURE_COLS],
                "Value": [round(float(latest[f]), 6) for f in FEATURE_COLS],
            })
            st.dataframe(feat_df, use_container_width=True, hide_index=True, height=320)

# ════════════════════════════════════════════════════════════════
# Tab 4 — Price Chart
# ════════════════════════════════════════════════════════════════
with tab_chart:
    chart_ticker = st.selectbox(
        "Select stock", options=selected if selected else DEFAULT_TICKERS[:5], key="chart_sel"
    )
    chart_period = st.selectbox(
        "Data period",
        options=PERIODS,
        index=2,
        format_func=lambda p: PERIOD_LABEL[p],
        key="chart_period_sel",
    )
    days = st.slider("Days to display", 30, 365, 90)

    if st.button("Show Chart", key="chart_btn"):
        with st.spinner(f"Loading {chart_ticker}…"):
            sig, conf, latest, df_feat = predict_ticker(chart_ticker, period=chart_period)

        if df_feat is None:
            st.error(f"Could not fetch data for **{chart_ticker}**.")
        else:
            df_plot = df_feat.dropna(subset=["MA_5", "MA_20"]).tail(days)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df_plot["Date"], df_plot["Close"], label="Close", color="#3b82f6", linewidth=1.8)
            ax.plot(df_plot["Date"], df_plot["MA_5"],  label="MA 5",  color="#f97316", linewidth=1.2, linestyle="--")
            ax.plot(df_plot["Date"], df_plot["MA_20"], label="MA 20", color="#a855f7", linewidth=1.2, linestyle="--")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
            plt.xticks(rotation=30, ha="right")
            ax.set_title(f"{chart_ticker} — Close Price & Moving Averages (last {days} days)")
            ax.set_ylabel("Price (USD)")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            if sig is not None:
                color = SIGNAL_COLOR[sig]
                conf_str = f" &nbsp; 信心度: **{conf:.1f}%**" if conf is not None else ""
                st.markdown(
                    f"**Current signal:** "
                    f"<span style='color:{color};font-size:1.2rem;font-weight:bold'>"
                    f"{SIGNAL_EMOJI[sig]} {SIGNAL_LABEL[sig]}</span>"
                    + conf_str,
                    unsafe_allow_html=True,
                )
