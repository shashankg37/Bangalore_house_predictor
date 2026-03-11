import streamlit as st
import pickle
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
}
.stApp {
    background: #0b0f1a !important;
}

/* ── hide streamlit chrome ── */
footer, #MainMenu, header { visibility: hidden; }

/* ── main container padding for mobile ── */
.block-container {
    padding: 1.5rem 1rem 2rem 1rem !important;
    max-width: 680px !important;
}

/* ── hero title ── */
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.75rem, 7vw, 2.8rem);
    font-weight: 800;
    color: #ffffff;
    line-height: 1.15;
    text-align: center;
    margin-bottom: 0.3rem;
}
.hero-title span { color: #f97316; }
.hero-sub {
    text-align: center;
    color: #64748b;
    font-size: clamp(0.78rem, 2.5vw, 0.92rem);
    margin-bottom: 1.8rem;
}

/* ── section label ── */
.section-label {
    color: #94a3b8;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.75rem;
    margin-top: 1.25rem;
}

/* ── streamlit widget text ── */
label, .stSelectbox label, .stNumberInput label {
    color: #cbd5e1 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}

/* ── selectbox ── */
.stSelectbox > div > div {
    background-color: #1e2433 !important;
    border: 1.5px solid #2d3748 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    min-height: 48px !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #f97316 !important;
    box-shadow: 0 0 0 2px rgba(249,115,22,0.2) !important;
}

/* ── number input ── */
.stNumberInput > div > div > input {
    background-color: #1e2433 !important;
    border: 1.5px solid #2d3748 !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-size: 1rem !important;
    min-height: 48px !important;
    padding: 0 1rem !important;
}
.stNumberInput > div > div > input:focus {
    border-color: #f97316 !important;
    box-shadow: 0 0 0 2px rgba(249,115,22,0.2) !important;
}
.stNumberInput button {
    background: #2d3748 !important;
    border: none !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── predict button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #f97316, #ea580c) !important;
    color: #ffffff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.85rem !important;
    min-height: 52px !important;
    letter-spacing: 0.04em !important;
    margin-top: 1.25rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
    box-shadow: 0 4px 20px rgba(249,115,22,0.3) !important;
}
.stButton > button:hover {
    opacity: 0.92 !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active {
    transform: translateY(0px) !important;
}

/* ── divider ── */
hr {
    border: none !important;
    border-top: 1px solid #1e2433 !important;
    margin: 1.5rem 0 !important;
}

/* ── result card ── */
.result-card {
    background: linear-gradient(135deg, #111827, #0f172a);
    border: 1.5px solid #f97316;
    border-radius: 18px;
    padding: clamp(1.25rem, 5vw, 2rem);
    text-align: center;
    margin-top: 1.5rem;
    box-shadow: 0 8px 32px rgba(249,115,22,0.12);
}
.result-label {
    font-size: 0.75rem;
    color: #64748b;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-price {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.2rem, 9vw, 3.5rem);
    font-weight: 800;
    color: #f97316;
    line-height: 1;
    margin-bottom: 0.4rem;
}
.result-sub {
    font-size: clamp(0.78rem, 2.5vw, 0.88rem);
    color: #475569;
    margin-bottom: 1.25rem;
    word-break: break-word;
}
.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.75rem;
}
.stat-box {
    background: #1e2433;
    border-radius: 12px;
    padding: 0.75rem 0.5rem;
}
.stat-val {
    font-family: 'Syne', sans-serif;
    font-size: clamp(0.95rem, 3.5vw, 1.2rem);
    font-weight: 700;
    color: #e2e8f0;
}
.stat-lbl {
    font-size: clamp(0.62rem, 2vw, 0.72rem);
    color: #475569;
    margin-top: 3px;
}

/* ── columns responsive on mobile ── */
@media screen and (max-width: 480px) {
    [data-testid="column"] {
        min-width: 100% !important;
        flex: 1 1 100% !important;
    }
    .stats-grid {
        grid-template-columns: repeat(3, 1fr);
        gap: 0.5rem;
    }
}

/* ── warning box ── */
.stAlert {
    background: #1e2433 !important;
    border-radius: 12px !important;
    border: 1px solid #f59e0b !important;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('banglore_home_prices_model.pickle', 'rb') as f:
        model = pickle.load(f)
    with open('columns.json', 'r') as f:
        data = json.load(f)
        columns = data if isinstance(data, list) else data['data_columns']
    locations = sorted([
        c.replace('location_', '')
        for c in columns
        if c.startswith('location_')
    ])
    return model, columns, locations

model, columns, locations = load_model()


# ── Predict ───────────────────────────────────────────────────────────────────
def predict_price(location, sqft, bath, balcony, bhk):
    x = np.zeros(len(columns))
    x[columns.index('total_sqft')] = sqft
    x[columns.index('bath')]       = bath
    x[columns.index('balcony')]    = balcony
    x[columns.index('bhk')]        = bhk
    loc_col = f'location_{location}'
    if loc_col in columns:
        x[columns.index(loc_col)] = 1
    return round(model.predict([x])[0], 2)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-title">🏠 Bengaluru <span>House Price</span> Predictor</div>
<div class="hero-sub">Linear Regression &nbsp;·&nbsp; 5,600+ listings &nbsp;·&nbsp; 254 locations &nbsp;·&nbsp; R² = 0.87</div>
""", unsafe_allow_html=True)


# ── Inputs ────────────────────────────────────────────────────────────────────
location = st.selectbox("📍 Location", ["Select a location..."] + locations)

sqft = st.number_input("📐 Total Sqft", min_value=300, max_value=10000, value=1200, step=50)

col1, col2 = st.columns(2)
with col1:
    bhk = st.selectbox("🛏️ BHK", [1, 2, 3, 4, 5, 6])
with col2:
    bath = st.selectbox("🚿 Bathrooms", [1, 2, 3, 4, 5, 6])

balcony = st.selectbox("🪟 Balconies", [0, 1, 2, 3])

predict_btn = st.button("🔍 Predict Price")


# ── Result ────────────────────────────────────────────────────────────────────
if predict_btn:
    if location == "Select a location...":
        st.warning("Please select a location first!")
    else:
        price = predict_price(location, sqft, bath, balcony, bhk)
        ppsf  = round((price * 100000) / sqft)

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Estimated Price</div>
            <div class="result-price">₹{price} L</div>
            <div class="result-sub">{location} &nbsp;·&nbsp; {bhk} BHK &nbsp;·&nbsp; {sqft} sqft</div>
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-val">₹{ppsf:,}</div>
                    <div class="stat-lbl">per sqft</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{bath}</div>
                    <div class="stat-lbl">bathrooms</div>
                </div>
                <div class="stat-box">
                    <div class="stat-val">{balcony}</div>
                    <div class="stat-lbl">balconies</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center;color:#334155;font-size:0.75rem;'>
    Built by Shashank &nbsp;·&nbsp; Bengaluru Housing Dataset &nbsp;·&nbsp; ML Beginner Project
</p>
""", unsafe_allow_html=True)
