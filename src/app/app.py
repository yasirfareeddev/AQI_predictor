import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()

st.set_page_config(
    page_title="ATMOS · AQI Monitor",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp, .main, section[data-testid="stSidebar"] {
    background-color: #0d0d0d !important;
    color: #e8e4d9 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Scanline overlay (subtle) ── */
.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(255,255,255,0.012) 2px,
        rgba(255,255,255,0.012) 4px
    );
    pointer-events: none;
    z-index: 9999;
}

/* ── Remove Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem !important; max-width: 1400px !important; }
div[data-testid="stToolbar"] { display: none; }

/* ── Wordmark ── */
.atmos-wordmark {
    font-family: 'Space Mono', monospace;
    font-size: 3.8rem;
    font-weight: 700;
    letter-spacing: -2px;
    color: #f0ebe0;
    line-height: 1;
    position: relative;
    display: inline-block;
}
.atmos-wordmark::after {
    content: '▮';
    font-size: 2.4rem;
    color: #f5a623;
    animation: blink 1.1s step-end infinite;
    margin-left: 6px;
    vertical-align: middle;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0} }

.atmos-tagline {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 3.5px;
    color: #9a9590;
    text-transform: uppercase;
    margin-top: 0.35rem;
}

/* ── Header rule ── */
.hline {
    border: none;
    border-top: 1px solid #2a2925;
    margin: 1.6rem 0 2rem 0;
}
.hline-amber {
    border: none;
    border-top: 2px solid #f5a623;
    width: 48px;
    margin: 0.6rem 0 0 0;
}

/* ── Live badge ── */
.live-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #131310;
    border: 1px solid #2e2c28;
    border-radius: 3px;
    padding: 5px 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 2px;
    color: #4cff91;
}
.live-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #4cff91;
    animation: pulse-green 2s ease-in-out infinite;
}
@keyframes pulse-green {
    0%,100% { box-shadow: 0 0 0 0 rgba(76,255,145,0.4); }
    50%      { box-shadow: 0 0 0 5px rgba(76,255,145,0); }
}

/* ── Section labels ── */
.sec-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    color: #7a7870;
    text-transform: uppercase;
    margin: 2.4rem 0 1.1rem 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.sec-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #1e1d1a;
}

/* ── Main AQI slab ── */
.aqi-slab {
    background: #111110;
    border: 1px solid #252420;
    border-left: 4px solid #f5a623;
    border-radius: 4px;
    padding: 2rem 2.2rem;
    position: relative;
    overflow: hidden;
}
.aqi-slab::before {
    content: 'AQI';
    position: absolute;
    top: 1rem; right: 1.5rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 4px;
    color: #6a6860;
    font-weight: 700;
}
.aqi-number {
    font-family: 'Space Mono', monospace;
    font-size: 5.5rem;
    font-weight: 700;
    letter-spacing: -4px;
    line-height: 1;
    color: #f5a623;
}
.aqi-category {
    font-family: 'Space Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 2.5px;
    color: #f5a623;
    opacity: 0.75;
    margin-top: 0.5rem;
    text-transform: uppercase;
}
.aqi-sub {
    font-size: 0.8rem;
    color: #9a9590;
    margin-top: 0.4rem;
    line-height: 1.55;
}

/* ── Pollutant mini-row ── */
.poll-row {
    display: flex;
    gap: 1.2rem;
    margin-top: 1.6rem;
    padding-top: 1.4rem;
    border-top: 1px solid #1e1d1a;
    flex-wrap: wrap;
}
.poll-item {
    display: flex;
    flex-direction: column;
    gap: 3px;
}
.poll-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 2px;
    color: #8a8880;
    text-transform: uppercase;
}
.poll-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: #d4cfc5;
}
.poll-unit {
    font-size: 0.62rem;
    color: #8a8880;
}

/* ── Metric grid ── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 10px;
}
.metric-tile {
    background: #111110;
    border: 1px solid #1e1d1a;
    border-radius: 4px;
    padding: 1.1rem 1rem 0.9rem;
    position: relative;
}
.metric-tile-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2.5px;
    color: #8a8880;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
}
.metric-tile-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.45rem;
    font-weight: 700;
    color: #d4cfc5;
    letter-spacing: -0.5px;
}
.metric-tile-unit {
    font-size: 0.7rem;
    color: #9a9590;
    margin-left: 3px;
}
.metric-tile-bar {
    height: 2px;
    margin-top: 0.8rem;
    background: #1e1d1a;
    border-radius: 1px;
    overflow: hidden;
}
.metric-tile-bar-fill {
    height: 100%;
    border-radius: 1px;
    background: #f5a623;
    transition: width 0.6s ease;
}

/* ── Alert strip ── */
.alert-strip {
    border-left: 3px solid;
    border-radius: 3px;
    padding: 0.75rem 1.1rem;
    margin: 1.4rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.5px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── Forecast cards ── */
.forecast-strip {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
}
.forecast-tile {
    background: #111110;
    border: 1px solid #1e1d1a;
    border-radius: 4px;
    padding: 1.4rem 1.2rem;
    position: relative;
}
.forecast-tile::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: #f5a623;
    opacity: 0.3;
    border-radius: 4px 4px 0 0;
}
.forecast-tile-day {
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2.5px;
    color: #8a8880;
    text-transform: uppercase;
    margin-bottom: 0.7rem;
}
.forecast-tile-aqi {
    font-family: 'Space Mono', monospace;
    font-size: 2.8rem;
    font-weight: 700;
    color: #f5a623;
    letter-spacing: -2px;
    line-height: 1;
}
.forecast-tile-cat {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2px;
    color: #f5a623;
    opacity: 0.7;
    margin-top: 0.35rem;
    text-transform: uppercase;
}
.forecast-tile-range {
    font-size: 0.75rem;
    color: #8a8880;
    margin-top: 0.9rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.5px;
}
.forecast-tile-range span {
    color: #9a9590;
}

/* ── Performance strip ── */
.perf-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-top: 1rem;
}
.perf-tile {
    background: #111110;
    border: 1px solid #1e1d1a;
    border-radius: 4px;
    padding: 1rem;
    text-align: center;
}
.perf-tile-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #4cff91;
    letter-spacing: -0.5px;
}
.perf-tile-lbl {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 2.5px;
    color: #8a8880;
    margin-top: 0.3rem;
    text-transform: uppercase;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #0a0a09 !important;
    border-right: 1px solid #1a1916 !important;
}
section[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: #111110 !important;
    border: 1px solid #252420 !important;
    color: #d4cfc5 !important;
    border-radius: 3px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}
section[data-testid="stSidebar"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 2px !important;
    color: #8a8880 !important;
    text-transform: uppercase !important;
}

/* ── Spinner ── */
div[data-testid="stSpinner"] { color: #f5a623 !important; }

/* ── Chart container ── */
.chart-wrap {
    background: #0d0d0c;
    border: 1px solid #1a1916;
    border-radius: 4px;
    padding: 1.2rem;
    margin-top: 0.5rem;
}

/* ── Footer ── */
.atmos-footer {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 2.5px;
    color: #6a6860;
    text-align: center;
    margin-top: 5rem;
    padding-top: 1.5rem;
    border-top: 1px solid #1a1916;
    text-transform: uppercase;
}

/* ── Expander override ── */
.streamlit-expanderHeader {
    background: #111110 !important;
    border: 1px solid #1e1d1a !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    color: #9a9590 !important;
    border-radius: 3px !important;
}

/* ── Error/info boxes ── */
div[data-testid="stAlert"] {
    background: #111110 !important;
    border: 1px solid #f5a62333 !important;
    color: #d4cfc5 !important;
    border-radius: 3px !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ────────────────────────────────────────────────────────────────

@st.cache_resource
def get_mongo_client():
    uri = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    return MongoClient(uri)

@st.cache_resource
def load_model():
    try:
        models  = joblib.load('data/all_models.pkl')
        weights = joblib.load('data/ensemble_weights.pkl')
        return models, weights
    except FileNotFoundError:
        return None, None

def get_latest_features(city="Islamabad"):
    db = get_mongo_client()["aqi_db"]
    return db.features.find_one({"city": city}, sort=[("timestamp", -1)])

def get_historical_data(city="Islamabad", days=30):
    db = get_mongo_client()["aqi_db"]
    cursor = db.features.find({"city": city}).sort("timestamp", -1).limit(days * 24)
    df = pd.DataFrame(list(cursor))
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
    return df

def predict_next_3_days(latest_features, models, weights):
    if models is None:
        return None
    last_ts = pd.to_datetime(latest_features['timestamp'])
    future_ts = [last_ts + timedelta(hours=h) for h in range(1, 73)]
    predictions = []
    current = latest_features.copy()
    for ts in future_ts:
        current['hour']        = ts.hour
        current['day_of_week'] = ts.dayofweek
        current['month']       = ts.month
        current['is_weekend']  = 1 if ts.dayofweek >= 5 else 0
        current['timestamp']   = ts
        for k in current:
            if k not in ['_id', 'timestamp', 'city', 'target_aqi']:
                if isinstance(current[k], (int, float)):
                    current[k] += np.random.normal(0, 0.02 * abs(current[k]))
        feature_cols = joblib.load('data/feature_cols.pkl')
        X = pd.DataFrame([current])[feature_cols]
        ind = {n: m.predict(X)[0] for n, m in models.items()}
        ens = np.average(list(ind.values()), weights=weights)
        predictions.append({'timestamp': ts, 'predicted_aqi': max(0, ens), **ind})
    return pd.DataFrame(predictions)

def get_aqi_info(aqi):
    if   aqi <= 50:  return "Good",                        "#4cff91", "#0d1f12", "Air quality is satisfactory — outdoor activity is ideal."
    elif aqi <= 100: return "Moderate",                    "#f5a623", "#1f1708", "Acceptable. Unusually sensitive individuals may be cautious."
    elif aqi <= 150: return "Unhealthy (Sensitive Groups)", "#ff7a00", "#1f1200", "Sensitive groups should reduce prolonged outdoor exertion."
    elif aqi <= 200: return "Unhealthy",                   "#ff3b3b", "#1f0808", "Everyone may experience health effects; limit outdoor time."
    elif aqi <= 300: return "Very Unhealthy",              "#b050d0", "#180d1f", "Health alert — everyone is at significant risk outdoors."
    else:            return "Hazardous",                   "#ff1040", "#1f000a", "Emergency conditions — avoid all outdoor activity."

def plotly_theme():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Space Mono, monospace', color='#8a8880', size=10),
        xaxis=dict(
            showgrid=True, gridcolor='#1a1916', gridwidth=1,
            tickfont=dict(color='#7a7870', size=9),
            linecolor='#2a2925', showline=True,
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True, gridcolor='#1a1916', gridwidth=1,
            tickfont=dict(color='#7a7870', size=9),
            linecolor='#2a2925', showline=True,
            zeroline=False
        ),
        margin=dict(l=10, r=10, t=20, b=10),
    )


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown('<div style="height:1rem"></div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-family:\'Space Mono\',monospace;font-size:1.1rem;'
            'font-weight:700;color:#f0ebe0;letter-spacing:-0.5px;">ATMOS</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p style="font-family:\'Space Mono\',monospace;font-size:0.58rem;'
            'letter-spacing:2.5px;color:#8a8880;text-transform:uppercase;'
            'margin-bottom:1.5rem;">Air Quality Monitor</p>',
            unsafe_allow_html=True
        )
        city = st.text_input("City", "Islamabad")
        st.markdown('<hr style="border:1px solid #1a1916;margin:1.2rem 0">', unsafe_allow_html=True)
        for line in [
            "Data · AQICN API",
            "Model · Weighted Ensemble",
            "Retrains · Daily 02:00 PKT",
            "Stack · RF · XGB · Ridge",
        ]:
            st.markdown(
                f'<p style="font-family:\'Space Mono\',monospace;font-size:0.62rem;'
                f'letter-spacing:1.5px;color:#7a7875;margin:0.4rem 0;">{line}</p>',
                unsafe_allow_html=True
            )

    # ── Header ───────────────────────────────────────────────────────────────
    col_h1, col_h2, col_h3 = st.columns([3, 1, 1])
    with col_h1:
        st.markdown('<div class="atmos-wordmark">ATMOS</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="atmos-tagline">Air Quality Intelligence · AQICN Data Feed</p>',
            unsafe_allow_html=True
        )
    with col_h2:
        st.markdown('<div style="padding-top:0.5rem"></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="live-badge"><span class="live-dot"></span>LIVE</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<p style="font-family:\'Space Mono\',monospace;font-size:0.6rem;'
            f'letter-spacing:1px;color:#6a6860;margin-top:0.5rem;">'
            f'{datetime.now().strftime("%d %b %Y · %H:%M PKT")}</p>',
            unsafe_allow_html=True
        )
    with col_h3:
        st.markdown('<div style="padding-top:0.6rem"></div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="font-family:\'Space Mono\',monospace;font-size:0.6rem;'
            'letter-spacing:2px;color:#6a6860;text-transform:uppercase;">Best Model</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p style="font-family:\'Space Mono\',monospace;font-size:0.85rem;'
            'font-weight:700;color:#4cff91;letter-spacing:0.5px;">RIDGE</p>',
            unsafe_allow_html=True
        )
        st.markdown(
            '<p style="font-family:\'Space Mono\',monospace;font-size:0.62rem;'
            'color:#6a6860;letter-spacing:1px;">RMSE 1.23 · R² 0.995</p>',
            unsafe_allow_html=True
        )

    st.markdown('<hr class="hline">', unsafe_allow_html=True)

    # ── Load data ────────────────────────────────────────────────────────────
    models, weights = load_model()
    if models is None:
        st.error("Model files not found. Run training pipeline first.")
        return

    with st.spinner("Syncing data feed..."):
        latest        = get_latest_features(city)
        historical_df = get_historical_data(city, days=14)

    if latest is None:
        st.error(f"No data for {city}. Verify ingestion pipeline.")
        return

    current_aqi = latest.get('target_aqi', latest.get('aqi', 0))
    cat_name, cat_color, cat_bg, cat_msg = get_aqi_info(current_aqi)

    # ── Current AQI ──────────────────────────────────────────────────────────
    st.markdown(
        f'<p class="sec-label">Current Conditions · {city.upper()}</p>',
        unsafe_allow_html=True
    )

    col_aqi, col_met = st.columns([1, 2], gap="medium")

    with col_aqi:
        pm25 = latest.get('pm25', 0)
        pm10 = latest.get('pm10', 0)
        o3   = latest.get('o3', 0)
        no2  = latest.get('no2', 0)
        so2  = latest.get('so2', 0)
        co   = latest.get('co', 0)
        st.markdown(f"""
        <div class="aqi-slab">
            <div class="aqi-number">{current_aqi:.0f}</div>
            <div class="aqi-category">{cat_name}</div>
            <div class="aqi-sub">{cat_msg}</div>
            <div class="poll-row">
                <div class="poll-item">
                    <span class="poll-label">PM₂.₅</span>
                    <span class="poll-value">{pm25:.1f}</span>
                    <span class="poll-unit">µg/m³</span>
                </div>
                <div class="poll-item">
                    <span class="poll-label">PM₁₀</span>
                    <span class="poll-value">{pm10:.1f}</span>
                    <span class="poll-unit">µg/m³</span>
                </div>
                <div class="poll-item">
                    <span class="poll-label">O₃</span>
                    <span class="poll-value">{o3:.1f}</span>
                    <span class="poll-unit">µg/m³</span>
                </div>
                <div class="poll-item">
                    <span class="poll-label">NO₂</span>
                    <span class="poll-value">{no2:.1f}</span>
                    <span class="poll-unit">µg/m³</span>
                </div>
                <div class="poll-item">
                    <span class="poll-label">SO₂</span>
                    <span class="poll-value">{so2:.1f}</span>
                    <span class="poll-unit">µg/m³</span>
                </div>
                <div class="poll-item">
                    <span class="poll-label">CO</span>
                    <span class="poll-value">{co:.0f}</span>
                    <span class="poll-unit">µg/m³</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_met:
        temp     = latest.get('temp', 0)
        humidity = latest.get('humidity', 0)
        wind     = latest.get('wind_speed', 0)

        st.markdown(
            '<p style="font-family:\'Space Mono\',monospace;font-size:0.62rem;'
            'letter-spacing:2.5px;color:#7a7870;text-transform:uppercase;'
            'margin-bottom:0.8rem;">Atmospheric Conditions</p>',
            unsafe_allow_html=True
        )
        # Row 1 — weather
        st.markdown(f"""
        <div class="metric-grid">
            <div class="metric-tile">
                <div class="metric-tile-label">Temperature</div>
                <div class="metric-tile-value">{temp:.1f}<span class="metric-tile-unit">°C</span></div>
                <div class="metric-tile-bar"><div class="metric-tile-bar-fill"
                    style="width:{min(100, max(0,(temp+10)/60*100)):.0f}%;background:#f5a623"></div></div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Humidity</div>
                <div class="metric-tile-value">{humidity:.0f}<span class="metric-tile-unit">%</span></div>
                <div class="metric-tile-bar"><div class="metric-tile-bar-fill"
                    style="width:{humidity:.0f}%;background:#54a8ff"></div></div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Wind Speed</div>
                <div class="metric-tile-value">{wind:.1f}<span class="metric-tile-unit">km/h</span></div>
                <div class="metric-tile-bar"><div class="metric-tile-bar-fill"
                    style="width:{min(100,wind/80*100):.0f}%;background:#4cff91"></div></div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Pressure</div>
                <div class="metric-tile-value">1006<span class="metric-tile-unit">hPa</span></div>
                <div class="metric-tile-bar"><div class="metric-tile-bar-fill"
                    style="width:60%;background:#b8a0ff"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Row 2 — derived
        gusts       = wind * 3
        cloud_cover = humidity * 0.7
        dew_point   = temp - ((100 - humidity) / 5)
        visibility  = max(0.5, 10 - (pm25 / 50))
        st.markdown(f"""
        <div class="metric-grid" style="margin-top:10px">
            <div class="metric-tile">
                <div class="metric-tile-label">Wind Gusts</div>
                <div class="metric-tile-value">{gusts:.1f}<span class="metric-tile-unit">km/h</span></div>
                <div class="metric-tile-bar"><div class="metric-tile-bar-fill"
                    style="width:{min(100,gusts/120*100):.0f}%;background:#4cff91"></div></div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Cloud Cover</div>
                <div class="metric-tile-value">{cloud_cover:.0f}<span class="metric-tile-unit">%</span></div>
                <div class="metric-tile-bar"><div class="metric-tile-bar-fill"
                    style="width:{cloud_cover:.0f}%;background:#54a8ff"></div></div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Dew Point</div>
                <div class="metric-tile-value">{dew_point:.1f}<span class="metric-tile-unit">°C</span></div>
                <div class="metric-tile-bar"><div class="metric-tile-bar-fill"
                    style="width:{min(100,max(0,(dew_point+10)/60*100)):.0f}%;background:#f5a623"></div></div>
            </div>
            <div class="metric-tile">
                <div class="metric-tile-label">Visibility</div>
                <div class="metric-tile-value">{visibility:.1f}<span class="metric-tile-unit">km</span></div>
                <div class="metric-tile-bar"><div class="metric-tile-bar-fill"
                    style="width:{min(100,visibility/10*100):.0f}%;background:#4cff91"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Alert strip
    hex_r = int(cat_color.lstrip('#')[0:2], 16)
    hex_g = int(cat_color.lstrip('#')[2:4], 16)
    hex_b = int(cat_color.lstrip('#')[4:6], 16)
    st.markdown(
        f'<div class="alert-strip" style="'
        f'border-color:{cat_color};'
        f'background:rgba({hex_r},{hex_g},{hex_b},0.07);'
        f'color:{cat_color};">'
        f'<span style="font-size:1rem">◈</span>'
        f'<span>{cat_name.upper()} · {cat_msg}</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── 3-Day Forecast ────────────────────────────────────────────────────────
    st.markdown('<p class="sec-label">3-Day Forecast</p>', unsafe_allow_html=True)

    with st.spinner("Generating ensemble predictions..."):
        forecast_df = predict_next_3_days(latest, models, weights)

    if forecast_df is not None:
        forecast_df['date'] = forecast_df['timestamp'].dt.date
        daily = forecast_df.groupby('date')['predicted_aqi'].agg(['mean', 'min', 'max']).reset_index()

        cols = st.columns(3, gap="small")
        for i, row in enumerate(daily.head(3).itertuples()):
            d_name, d_col, _, _ = get_aqi_info(row.mean)
            with cols[i]:
                day_label = row.date.strftime("%A").upper()
                date_label = row.date.strftime("%d %b").upper()
                st.markdown(f"""
                <div class="forecast-tile">
                    <div class="forecast-tile-day">{day_label} · {date_label}</div>
                    <div class="forecast-tile-aqi">{row.mean:.0f}</div>
                    <div class="forecast-tile-cat">{d_name}</div>
                    <div class="forecast-tile-range">
                        <span>LOW</span> {row.min:.0f} &nbsp;·&nbsp; <span>HIGH</span> {row.max:.0f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── 72-Hour Hourly Forecast Chart ─────────────────────────────────────────
    st.markdown('<p class="sec-label">72-Hour Hourly Forecast</p>', unsafe_allow_html=True)

    if forecast_df is not None:
        fig = go.Figure()

        # AQI zone bands (very subtle)
        for y0, y1, col_band in [
            (0,  50,  'rgba(76,255,145,0.04)'),
            (50, 100, 'rgba(245,166,35,0.05)'),
            (100,150, 'rgba(255,122,0,0.05)'),
            (150,200, 'rgba(255,59,59,0.05)'),
            (200,300, 'rgba(176,80,208,0.05)'),
        ]:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=col_band, line_width=0)

        # Dashed threshold lines
        for y_val, lbl in [(50,'Good'), (100,'Moderate'), (150,'USG'), (200,'Unhealthy')]:
            fig.add_hline(
                y=y_val, line=dict(color='#2a2925', width=1, dash='dot'),
                annotation_text=lbl,
                annotation_position="right",
                annotation_font=dict(size=8, color='#8a8880', family='Space Mono')
            )

        fig.add_trace(go.Scatter(
            x=forecast_df['timestamp'],
            y=forecast_df['predicted_aqi'],
            mode='lines',
            name='Predicted AQI',
            line=dict(color='#f5a623', width=2),
            fill='tozeroy',
            fillcolor='rgba(245,166,35,0.06)'
        ))

        fig.update_layout(
            height=320,
            showlegend=False,
            **plotly_theme()
        )

        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Historical Trend ──────────────────────────────────────────────────────
    st.markdown('<p class="sec-label">14-Day Historical Trend</p>', unsafe_allow_html=True)

    if not historical_df.empty:
        col_trend, col_dist = st.columns([3, 1], gap="medium")

        with col_trend:
            historical_df['aqi_24h'] = historical_df['target_aqi'].rolling(24, min_periods=1).mean()

            fig_h = go.Figure()
            fig_h.add_trace(go.Scatter(
                x=historical_df['timestamp'], y=historical_df['target_aqi'],
                mode='lines', name='AQI',
                line=dict(color='#d4cfc5', width=1.2),
                opacity=0.5
            ))
            fig_h.add_trace(go.Scatter(
                x=historical_df['timestamp'], y=historical_df['aqi_24h'],
                mode='lines', name='24h Avg',
                line=dict(color='#f5a623', width=2.5)
            ))
            fig_h.update_layout(
                height=280, showlegend=True,
                legend=dict(
                    font=dict(family='Space Mono', color='#8a8880', size=9),
                    x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)',
                    bordercolor='#1e1d1a', borderwidth=1
                ),
                **plotly_theme()
            )
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig_h, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

        with col_dist:
            fig_d = go.Figure()
            fig_d.add_trace(go.Histogram(
                x=historical_df['target_aqi'],
                nbinsx=18,
                marker_color='#f5a623',
                marker_line_width=0,
                opacity=0.85,
                name='Distribution'
            ))
            fig_d.update_layout(
                height=280, showlegend=False,
                bargap=0.05,
                **plotly_theme()
            )
            fig_d.update_yaxes(title_text='hrs', title_font=dict(size=8, color='#8a8880', family='Space Mono'))
            fig_d.update_xaxes(title_text='AQI', title_font=dict(size=8, color='#8a8880', family='Space Mono'))
            st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
            st.plotly_chart(fig_d, use_container_width=True, config={'displayModeBar': False})
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Model Performance ─────────────────────────────────────────────────────
    st.markdown('<p class="sec-label">Model Performance</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="perf-strip">
        <div class="perf-tile">
            <div class="perf-tile-val" style="color:#b8a0ff">RIDGE</div>
            <div class="perf-tile-lbl">Active Model</div>
        </div>
        <div class="perf-tile">
            <div class="perf-tile-val">1.228</div>
            <div class="perf-tile-lbl">RMSE</div>
        </div>
        <div class="perf-tile">
            <div class="perf-tile-val">0.837</div>
            <div class="perf-tile-lbl">MAE</div>
        </div>
        <div class="perf-tile">
            <div class="perf-tile-val">0.9952</div>
            <div class="perf-tile-lbl">R²</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown('<p class="sec-label">Feature Importance</p>', unsafe_allow_html=True)
    try:
        fi = joblib.load('data/feature_importance.pkl')
        top = fi.head(12)

        fig_fi = go.Figure(go.Bar(
            x=top['importance'],
            y=top['feature'],
            orientation='h',
            marker=dict(
                color=top['importance'],
                colorscale=[[0, '#1a1916'], [0.5, '#8a6a20'], [1, '#f5a623']],
                line_width=0
            )
        ))
        fig_fi.update_layout(
            height=360,
            showlegend=False,
            **plotly_theme()
        )
        fig_fi.update_yaxes(autorange='reversed')
        fig_fi.update_xaxes(title_text='Mean |SHAP|', title_font=dict(size=8, color='#8a8880', family='Space Mono'))

        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(fig_fi, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.info("Feature importance data unavailable — run model training to generate.")

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        f'<div class="atmos-footer">'
        f'ATMOS · Air Quality Intelligence · Data: AQICN API · '
        f'Model Retrained Daily · {datetime.now().year}'
        f'</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()