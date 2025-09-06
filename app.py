# app.py - AI Crop Recommendation (Bilingual)
import streamlit as st
from pathlib import Path
import pandas as pd
import altair as alt
from crop_predictor import FEATURES, load_model, recommend_topk, load_metadata

# -------------------------------
# Paths
# -------------------------------
EXPORT_DIR = Path("export_model")
MODEL_PATH = EXPORT_DIR / "crop_recommender_rf.joblib"
META_PATH = EXPORT_DIR / "model_metadata.json"

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="AI Crop Recommendation 🌱", layout="wide")

# -------------------------------
# Custom CSS
# -------------------------------
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)), 
                    url("https://images.unsplash.com/photo-1500937386664-56d1dfef3854?auto=format&fit=crop&w=1600&q=80");
        background-size: cover;
        background-position: center;
        color: white !important;
    }
    h1, h2, h3, h4, h5 {
        color: white !important;
    }
    div.stButton > button {
        background-color: #4caf50;
        color: white;
        border-radius: 25px;
        padding: 0.6em 1.5em;
        font-weight: bold;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #388e3c;
    }
    div[data-baseweb="select"] > div {
        background-color: white !important;
        color: black !important;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model
# -------------------------------
with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_PATH.as_posix())
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model = None

metadata = load_metadata(META_PATH.as_posix())

# -------------------------------
# Translation dictionary
# -------------------------------
trans = {
    "English": {
        "hero_title": "Earth-Friendly Agriculture for a Better Tomorrow 🌍",
        "hero_sub": "Unleash creativity in farming — discover innovative techniques to boost productivity and deepen your connection with the land.",
        "get_started": "Get Started Now",
        "learn_service": "Learn Service",
        "enter_conditions": "Enter Soil & Weather Conditions",
        "Nitrogen": "Nitrogen (N)",
        "Phosphorus": "Phosphorus (P)",
        "Potassium": "Potassium (K)",
        "pH": "Soil pH",
        "Temperature": "Temperature (°C)",
        "Humidity": "Humidity (%)",
        "Rainfall": "Rainfall (mm)",
        "top_k": "Number of top recommendations",
        "recommend": "Recommend Crops 🌾",
        "top_crops": "✅ Top Recommended Crops",
        "probabilities": "📊 Prediction Probabilities"
    },
    "हिंदी": {
        "hero_title": "एक बेहतर कल के लिए पर्यावरण-अनुकूल कृषि 🌍",
        "hero_sub": "खेती में रचनात्मकता को उजागर करें — उत्पादकता बढ़ाने और भूमि से गहरा संबंध बनाने के लिए नई तकनीकों की खोज करें।",
        "get_started": "अभी शुरू करें",
        "learn_service": "सेवा जानें",
        "enter_conditions": "मिट्टी और मौसम की जानकारी दर्ज करें",
        "Nitrogen": "नाइट्रोजन (N)",
        "Phosphorus": "फॉस्फोरस (P)",
        "Potassium": "पोटेशियम (K)",
        "pH": "मिट्टी का पीएच",
        "Temperature": "तापमान (°C)",
        "Humidity": "नमी (%)",
        "Rainfall": "वर्षा (मिमी)",
        "top_k": "शीर्ष सिफारिशों की संख्या",
        "recommend": "फसल सुझाएँ 🌾",
        "top_crops": "✅ शीर्ष सिफारिश की गई फसलें",
        "probabilities": "📊 भविष्यवाणी संभावनाएँ"
    }
}

# -------------------------------
# Crop name translations
# -------------------------------
crop_trans = {
    "rice": "चावल",
    "wheat": "गेहूँ",
    "maize": "मक्का",
    "chickpea": "चना",
    "kidneybeans": "राजमा",
    "pigeonpeas": "अरहर",
    "mothbeans": "मटकी",
    "mungbean": "मूँग",
    "blackgram": "उड़द",
    "lentil": "मसूर",
    "pomegranate": "अनार",
    "banana": "केला",
    "mango": "आम",
    "grapes": "अंगूर",
    "watermelon": "तरबूज",
    "muskmelon": "खरबूजा",
    "apple": "सेब",
    "orange": "संतरा",
    "papaya": "पपीता",
    "coconut": "नारियल",
    "cotton": "कपास",
    "jute": "जूट",
    "coffee": "कॉफ़ी"
}

# -------------------------------
# Language Selector + Hero Section
# -------------------------------
lang = st.selectbox("🌐 Select Language / भाषा चुनें", ["English", "हिंदी"])
t = trans[lang]

st.markdown(
    f"""
    <div style="padding: 60px 20px; text-align:center;">
        <h1 style="font-size: 3rem; font-weight: bold;">
            {t["hero_title"]}
        </h1>
        <p style="font-size: 1.2rem; max-width: 700px; margin:auto; color: #ddd;">
            {t["hero_sub"]}
        </p>
        <div style="margin-top: 20px;">
            <a href="#prediction" style="text-decoration: none;">
                <button style="background-color:#c6ff00; color:black; padding: 0.7em 1.5em; border-radius: 30px; border:none; font-weight:bold; margin-right:10px;">
                    {t["get_started"]}
                </button>
            </a>
            <a href="#service" style="text-decoration: none;">
                <button style="background-color:white; color:black; padding: 0.7em 1.5em; border-radius: 30px; border:none; font-weight:bold;">
                    {t["learn_service"]}
                </button>
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True
)

# -------------------------------
# Prediction Section
# -------------------------------
st.markdown("<div id='prediction'></div>", unsafe_allow_html=True)
st.header(t["enter_conditions"])

col1, col2 = st.columns(2)

with col1:
    N = st.number_input(t["Nitrogen"], min_value=0, max_value=200, value=50)
    P = st.number_input(t["Phosphorus"], min_value=0, max_value=200, value=50)
    K = st.number_input(t["Potassium"], min_value=0, max_value=200, value=50)
    ph = st.number_input(t["pH"], min_value=0.0, max_value=14.0, value=6.5)

with col2:
    temperature = st.number_input(t["Temperature"], min_value=-10.0, max_value=60.0, value=25.0)
    humidity = st.number_input(t["Humidity"], min_value=0.0, max_value=100.0, value=80.0)
    rainfall = st.number_input(t["Rainfall"], min_value=0.0, max_value=500.0, value=200.0)

k = st.slider(t["top_k"], min_value=1, max_value=10, value=5)

# -------------------------------
# Predict
# -------------------------------
if st.button(t["recommend"]):
    if model is None:
        st.error("Model is not loaded. Cannot make predictions.")
    else:
        topk, proba, labels = recommend_topk(
            model,
            N=N, P=P, K=K,
            temperature=temperature,
            humidity=humidity,
            ph=ph,
            rainfall=rainfall,
            k=k
        )

        st.subheader(t["top_crops"])
        for crop, score in topk:
            display_crop = crop_trans[crop] if lang == "हिंदी" and crop in crop_trans else crop
            st.write(f"- **{display_crop}** ({score:.2%})")

        st.subheader(t["probabilities"])
        df_proba = pd.DataFrame({"Crop": labels, "Probability": proba})
        df_top = df_proba.sort_values("Probability", ascending=False).head(k)

        if lang == "हिंदी":
            df_top["Crop"] = df_top["Crop"].apply(lambda c: crop_trans[c] if c in crop_trans else c)

        chart = (
            alt.Chart(df_top)
            .mark_bar(color="#c6ff00")
            .encode(
                x=alt.X("Probability:Q", axis=alt.Axis(format="%")),
                y=alt.Y("Crop:N", sort="-x"),
                tooltip=[alt.Tooltip("Crop:N"), alt.Tooltip("Probability:Q", format=".2%")]
            )
        )
        st.altair_chart(chart, use_container_width=True)
