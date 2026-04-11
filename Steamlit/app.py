import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Helvetica', 'Helvetica Neue', sans-serif;
}

* {
    font-family: 'Helvetica', 'Helvetica Neue', sans-serif !important;
}

/* ===== MAIN BACKGROUND ===== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(
        135deg,
        #0f172a 0%,
        #1e293b 45%,
        #111827 100%
    );
    min-height: 100vh;
}

/* ===== GLOW EFFECT ===== */
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        radial-gradient(circle at 20% 20%, rgba(59,130,246,0.22), transparent 35%),
        radial-gradient(circle at 80% 30%, rgba(139,92,246,0.16), transparent 30%),
        radial-gradient(circle at 50% 80%, rgba(16,185,129,0.10), transparent 25%);
    z-index: 0;
    pointer-events: none;
}

/* ===== MAKE CONTENT ABOVE BACKGROUND ===== */
.main .block-container {
    position: relative;
    z-index: 1;
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 760px;
}

/* ===== REMOVE STREAMLIT HEADER BACKGROUND ===== */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

[data-testid="stToolbar"] {
    z-index: 2;
}

/* ===== TITLES ===== */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 8px;
    color: #f9fafb;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5e1;
    margin-bottom: 30px;
}

/* ===== INPUT CARD CONTAINER ===== */
.st-key-input_card {
    background: rgba(255,255,255,0.07);
    padding: 24px 28px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.10);
    box-shadow: 0 12px 40px rgba(0,0,0,0.18);
    backdrop-filter: blur(16px);
    margin-bottom: 18px;
}

.st-key-input_card > div {
    position: relative;
    z-index: 1;
}

/* ===== RESULT CARD ===== */
.result-card {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.15) 0%, rgba(139, 92, 246, 0.12) 50%, rgba(37, 99, 235, 0.10) 100%);
    padding: 42px 32px;
    border-radius: 28px;
    text-align: center;
    box-shadow: 0 20px 50px rgba(37, 99, 235, 0.25), inset 0 1px 0 rgba(255,255,255,0.15);
    margin-top: 24px;
    margin-bottom: 24px;
    border: 1.5px solid rgba(59, 130, 246, 0.35);
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
}

.result-card::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
    animation: subtle-glow 6s ease-in-out infinite;
}

@keyframes subtle-glow {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(10px, -10px); }
}
}

.result-label {
    color: #d1d5db;
    font-size: 18px;
    margin-bottom: 10px;
    position: relative;
    z-index: 1;
}

.result-price {
    color: #ffffff;
    font-size: 56px;
    font-weight: 900;
    line-height: 1.1;
    position: relative;
    z-index: 1;
    background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #1d4ed8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ===== SECTION TITLE ===== */
.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 18px;
    margin-bottom: 18px;
    color: #f9fafb;
}

/* ===== FEATURE BOX ===== */
.feature-box {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 16px;
    padding: 16px;
    text-align: center;
    margin-bottom: 12px;
    backdrop-filter: blur(10px);
}

.feature-label {
    font-size: 14px;
    color: #cbd5e1;
    margin-bottom: 8px;
}

.feature-value {
    font-size: 22px;
    font-weight: 700;
    color: #f9fafb;
}

/* ===== BUTTON ===== */
.stButton > button {
    width: 100%;
    border-radius: 14px;
    height: 54px;
    font-size: 16px;
    font-weight: 600;
    letter-spacing: 0.5px;
    border: 1px solid rgba(255,255,255,0.15);
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 50%, #1d4ed8 100%);
    color: #ffffff;
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 10px 28px rgba(37, 99, 235, 0.25), inset 0 1px 0 rgba(255,255,255,0.15);
    cursor: pointer;
    position: relative;
    overflow: hidden;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s ease;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #2563eb 100%);
    color: #ffffff;
    transform: translateY(-3px);
    box-shadow: 0 16px 40px rgba(37, 99, 235, 0.4), inset 0 1px 0 rgba(255,255,255,0.25);
}

.stButton > button:active {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(37, 99, 235, 0.3), inset 0 1px 0 rgba(255,255,255,0.15);
}

/* ===== INPUT SPACING ===== */
div[data-testid="stNumberInput"],
div[data-testid="stTextInput"],
div[data-testid="stSelectbox"] {
    margin-bottom: 10px;
}

/* ===== LABELS ===== */
label, .stMarkdown, .stTextInput, .stNumberInput, .stSelectbox {
    color: #f9fafb !important;
}

/* ===== EXPANDER ===== */
details {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px 8px;
    border: 1px solid rgba(255,255,255,0.08);
}

/* ===== DATAFRAME ===== */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 6px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# CONFIG
# =========================
MODEL_PATH = "car_price_pipeline_final.pkl"

num_cols = [
    "CAE", "log_mileage", "km_per_year", "engineSize", "mpg", "tax",
    "mileage_age_interaction", "high_mileage", "large_engine", "auto_large_engine"
]

cat_cols = ["brand", "model", "transmission", "fuelType"]
all_model_cols = num_cols + cat_cols


# =========================
# FEATURE ENGINEERING
# =========================
def fe_transform(df_raw, current_year=2020):
    df = df_raw.copy()

    for c in ["year", "mileage", "engineSize", "tax", "mpg", "price"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "model" in df.columns:
        df["model"] = df["model"].astype(str).str.lower().str.strip()

    if "brand" in df.columns:
        df["brand"] = df["brand"].astype(str).str.lower().str.strip()

    if "transmission" in df.columns:
        df["transmission"] = df["transmission"].astype(str).str.strip()

    if "fuelType" in df.columns:
        df["fuelType"] = df["fuelType"].astype(str).str.strip()

    if "year" in df.columns:
        df["vehicle_age"] = current_year - df["year"]
        df["CAE"] = np.log1p(df["vehicle_age"].clip(lower=0))

    if "mileage" in df.columns:
        df["log_mileage"] = np.log1p(df["mileage"].clip(lower=0))

    if "mileage" in df.columns and "vehicle_age" in df.columns:
        df["km_per_year"] = df["mileage"] / (df["vehicle_age"] + 1)

    if "log_mileage" in df.columns and "CAE" in df.columns:
        df["mileage_age_interaction"] = df["log_mileage"] * df["CAE"]

    if "mileage" in df.columns:
        df["high_mileage"] = (df["mileage"] > 100_000).astype(int)

    if "engineSize" in df.columns:
        df["large_engine"] = (df["engineSize"] >= 2.0).astype(int)

    if "transmission" in df.columns and "engineSize" in df.columns:
        df["auto_large_engine"] = (
            (df["transmission"] == "Automatic") & (df["engineSize"] >= 2.0)
        ).astype(int)

    return df


def select_model_columns(df):
    df = df.copy()
    for col in all_model_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[all_model_cols]


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

pipe = load_model()


# =========================
# GET PREPROCESSOR + MODEL
# =========================
@st.cache_resource
def get_pipeline_parts():
    step_names = list(pipe.named_steps.keys())

    preprocessor = None
    model = None

    for name in step_names:
        if "prep" in name.lower() or "transform" in name.lower() or "column" in name.lower():
            preprocessor = pipe.named_steps[name]
            break

    if preprocessor is None and "preprocessor" in pipe.named_steps:
        preprocessor = pipe.named_steps["preprocessor"]

    if preprocessor is None:
        preprocessor = pipe[:-1]

    model = pipe[-1]

    return preprocessor, model, step_names

preprocessor, model, step_names = get_pipeline_parts()


# =========================
# SHAP FUNCTIONS
# =========================
def transform_for_model(test_car: pd.DataFrame):
    test_car_fe = fe_transform(test_car)
    X = select_model_columns(test_car_fe)

    x_trans = preprocessor.transform(X)

    if hasattr(x_trans, "toarray"):
        x_trans = x_trans.toarray()

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(x_trans.shape[1])]

    return X, x_trans, feature_names


def get_shap_explanation(test_car: pd.DataFrame):
    X, x_trans, feature_names = transform_for_model(test_car)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_trans)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    base_value = explainer.expected_value
    if isinstance(base_value, np.ndarray):
        base_value = base_value[0]

    explanation = shap.Explanation(
        values=shap_values[0],
        base_values=base_value,
        data=x_trans[0],
        feature_names=feature_names
    )

    return explanation, x_trans, feature_names, shap_values, base_value


def plot_shap_waterfall(test_car: pd.DataFrame):
    explanation, _, _, _, _ = get_shap_explanation(test_car)

    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, max_display=15, show=False)
    st.pyplot(fig, clear_figure=True)


def show_shap_table(test_car: pd.DataFrame):
    explanation, _, _, _, _ = get_shap_explanation(test_car)

    df_imp = pd.DataFrame({
        "Feature": explanation.feature_names,
        "Value": explanation.data,
        "SHAP Impact": explanation.values
    })

    df_imp["Abs Impact"] = np.abs(df_imp["SHAP Impact"])
    df_imp = df_imp.sort_values("Abs Impact", ascending=False)
    df_imp = df_imp.drop(columns=["Abs Impact"])

    st.dataframe(df_imp, use_container_width=True, height=420)


# =========================
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "input"

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "car_data" not in st.session_state:
    st.session_state.car_data = None


# =========================
# PAGE 1: INPUT
# =========================
if st.session_state.page == "input":
    st.markdown('<div class="main-title">Used Car Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Enter the car details and get an estimated market price</div>', unsafe_allow_html=True)

    with st.container(key="input_card"):
        col1, col2 = st.columns(2)

        with col1:
            brand = st.text_input("Brand", value="BMW")
            year = st.number_input("Year", min_value=1990, max_value=2030, value=2018, step=1)
            transmission = st.selectbox("Transmission", ["Automatic", "Manual", "Semi-Auto"])
            engine_size = st.number_input("Engine Size", min_value=0.0, max_value=10.0, value=2.0, step=0.1)

        with col2:
            model_name = st.text_input("Model", value="3 Series")
            mileage = st.number_input("Mileage", min_value=0, max_value=500000, value=40000, step=1000)
            fuel_type = st.selectbox("Fuel Type", ["Diesel", "Petrol", "Hybrid", "Electric", "Other"])
            mpg = st.number_input("MPG", min_value=0.0, max_value=200.0, value=55.4, step=0.1)

        tax = st.number_input("Tax", min_value=0, max_value=2000, value=180, step=10)

    if st.button("Predict Price"):
        test_car = pd.DataFrame([{
            "brand": brand,
            "model": model_name,
            "year": year,
            "mileage": mileage,
            "transmission": transmission,
            "fuelType": fuel_type,
            "engineSize": engine_size,
            "mpg": mpg,
            "tax": tax
        }])

        try:
            pred_log = pipe.predict(test_car)[0]
            pred_price = float(np.expm1(pred_log))
        except Exception:
            try:
                test_car_fe = fe_transform(test_car)
                X = select_model_columns(test_car_fe)
                pred_log = pipe.predict(X)[0]
                pred_price = float(np.expm1(pred_log))
            except Exception as e:
                st.error("Prediction error")
                st.code(str(e))
                st.stop()

        st.session_state.prediction = pred_price
        st.session_state.car_data = test_car.iloc[0].to_dict()
        st.session_state.page = "result"
        st.rerun()


# =========================
# PAGE 2: RESULT
# =========================
elif st.session_state.page == "result":
    car = st.session_state.car_data
    predicted_price = st.session_state.prediction

    st.markdown('<div class="main-title">Prediction Result</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Estimated price based on the submitted vehicle features</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="result-card">
        <div class="result-label">Estimated Car Price</div>
        <div class="result-price">£{predicted_price:,.2f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Vehicle Details</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    features = [
        ("Brand", car["brand"]),
        ("Model", car["model"]),
        ("Year", car["year"]),
        ("Mileage", f'{car["mileage"]:,} miles'),
        ("Transmission", car["transmission"]),
        ("Fuel Type", car["fuelType"]),
        ("Engine Size", f'{car["engineSize"]} L'),
        ("MPG", car["mpg"]),
        ("Tax", f'£{car["tax"]}')
    ]

    for i, (label, value) in enumerate(features):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"""
            <div class="feature-box">
                <div class="feature-label">{label}</div>
                <div class="feature-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    test_car = pd.DataFrame([car])

    st.markdown('<div class="section-title">Why this price? (SHAP Waterfall)</div>', unsafe_allow_html=True)
    try:
        plot_shap_waterfall(test_car)
    except Exception as e:
        st.warning("SHAP waterfall could not be displayed.")
        st.code(str(e))

    st.markdown('<div class="section-title">Feature Impact Table</div>', unsafe_allow_html=True)
    try:
        show_shap_table(test_car)
    except Exception as e:
        st.warning("SHAP table could not be displayed.")
        st.code(str(e))

    st.write("")
    if st.button("Predict Another Car"):
        st.session_state.page = "input"
        st.rerun()