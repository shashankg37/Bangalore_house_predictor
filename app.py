import streamlit as st
import pickle
import json
import numpy as np

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0b0f1a;
}

.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}

.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    margin-bottom: 0.4rem;
}

.hero h1 span {
    color: #f97316;
}

.hero p {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 300;
}

.card {
    background: #111827;
    border: 1px solid #1f2937;
    border-radius: 16px;
    padding: 2rem;
    margin: 1.5rem 0;
}

.result-box {
    background: linear-gradient(135deg, #1a1f2e 0%, #0f172a 100%);
    border: 1px solid #f97316;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1.5rem;
}

.result-box .label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #94a3b8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.result-box .price {
    font-family: 'Syne', sans-serif;
    font-size: 3.2rem;
    font-weight: 800;
    color: #f97316;
    line-height: 1;
}

.result-box .sub {
    font-size: 0.9rem;
    color: #64748b;
    margin-top: 0.5rem;
}

.stat-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
}

.stat {
    flex: 1;
    background: #1f2937;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}

.stat .s-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #e2e8f0;
}

.stat .s-label {
    font-size: 0.75rem;
    color: #64748b;
    margin-top: 2px;
}

.divider {
    border: none;
    border-top: 1px solid #1f2937;
    margin: 1.5rem 0;
}

/* Streamlit widget overrides */
label {
    color: #cbd5e1 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}

.stSelectbox > div > div {
    background: #1f2937 !important;
    border: 1px solid #374151 !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

.stSlider > div {
    color: #e2e8f0 !important;
}

.stButton > button {
    width: 100%;
    background: #f97316 !important;
    color: #ffffff !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.03em !important;
    transition: all 0.2s !important;
    margin-top: 0.5rem;
}

.stButton > button:hover {
    background: #ea6c0a !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(249,115,22,0.35) !important;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Load model + columns ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('banglore_home_prices_model.pickle', 'rb') as f:
        model = pickle.load(f)
    with open('columns.json', 'r') as f:
        data = json.load(f)
        # handle both formats
        if isinstance(data, list):
            columns = data
        else:
            columns = data['data_columns']
    locations = sorted([c.replace('location_', '') for c in columns if c.startswith('location_')])
    return model, columns, locations

model, columns, locations = load_model()
# ── Predict function ──────────────────────────────────────────────────────────
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


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>Bengaluru <span>House Price</span><br>Predictor</h1>
    <p>Linear Regression · 5,600+ listings · 254 locations</p>
</div>
""", unsafe_allow_html=True)


# ── Input form ────────────────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("#### 📍 Property Details")

location = st.selectbox("Location", ["Select a location..."] + locations)

col1, col2 = st.columns(2)
with col1:
    sqft = st.number_input("Total Sqft", min_value=300, max_value=10000, value=1200, step=50)
with col2:
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5, 6])

col3, col4 = st.columns(2)
with col3:
    bath = st.selectbox("Bathrooms", [1, 2, 3, 4, 5, 6])
with col4:
    balcony = st.selectbox("Balconies", [0, 1, 2, 3])

predict_btn = st.button("🔍 Predict Price")
st.markdown('</div>', unsafe_allow_html=True)


# ── Result ────────────────────────────────────────────────────────────────────
if predict_btn:
    if location == "Select a location...":
        st.warning("Please select a location first!")
    else:
        price = predict_price(location, sqft, bath, balcony, bhk)
        price_per_sqft = round((price * 100000) / sqft)

        st.markdown(f"""
        <div class="result-box">
            <div class="label">Estimated Price</div>
            <div class="price">₹{price} L</div>
            <div class="sub">{location} · {bhk} BHK · {sqft} sqft</div>
            <div class="stat-row">
                <div class="stat">
                    <div class="s-val">₹{price_per_sqft:,}</div>
                    <div class="s-label">per sqft</div>
                </div>
                <div class="stat">
                    <div class="s-val">{bath}</div>
                    <div class="s-label">bathrooms</div>
                </div>
                <div class="stat">
                    <div class="s-val">{balcony}</div>
                    <div class="s-label">balconies</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer info ───────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; color:#374151; font-size:0.78rem;'>
    Built with Linear Regression · Bengaluru Housing Dataset · Mean R² = 0.87
</p>
""", unsafe_allow_html=True)
