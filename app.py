import streamlit as st
from pathlib import Path
import pandas as pd
import altair as alt

# Import helper functions
from crop_predictor import FEATURES, load_model, recommend_topk, load_metadata

# -------------------------------
# Paths
# -------------------------------
EXPORT_DIR = Path("export_model")
MODEL_PATH = EXPORT_DIR / "crop_recommender_rf.joblib"
META_PATH = EXPORT_DIR / "model_metadata.json"

# -------------------------------
# Load model and metadata
# -------------------------------
st.set_page_config(page_title="AI Crop Recommendation 🌱", layout="centered")

with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_PATH.as_posix())
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model = None

metadata = load_metadata(META_PATH.as_posix())

# -------------------------------
# Language Selection
# -------------------------------
lang = st.sidebar.selectbox("Language / भाषा", ["English", "हिंदी"])

# -------------------------------
# Translation dictionary
# -------------------------------
trans = {
    "English": {
        "title": "🌱 AI-based Crop Recommendation System",
        "model_info": "ℹ️ Model Info",
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
        "title": "🌱 एआई आधारित फसल सिफारिश प्रणाली",
        "model_info": "ℹ️ मॉडल जानकारी",
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

t = trans[lang]

st.title(t["title"])
st.sidebar.header(t["model_info"])
st.sidebar.json(metadata)

# -------------------------------
# User Inputs
# -------------------------------
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

        # If language is Hindi, translate crop names (simple mapping example)
        if lang == "हिंदी":
            crop_translation = {
                "rice": "चावल",
                "maize": "मकई",
                "chickpea": "चना",
                "kidneybeans": "राजमा",
                "pigeonpeas": "अरहर/तूर",
                "mothbeans": "मोठ",
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
            topk = [(crop_translation.get(crop.lower(), crop), score) for crop, score in topk]
            labels = [crop_translation.get(crop.lower(), crop) for crop in labels]

        st.subheader(t["top_crops"])
        for crop, score in topk:
            st.write(f"- **{crop}** ({score:.2%})")

        st.subheader(t["probabilities"])
        df_proba = pd.DataFrame({"Crop": labels, "Probability": proba})
        df_top = df_proba.sort_values("Probability", ascending=False).head(k)

        chart = (
            alt.Chart(df_top)
            .mark_bar(color="#4caf50")
            .encode(
                x=alt.X("Probability:Q", axis=alt.Axis(format="%")),
                y=alt.Y("Crop:N", sort="-x"),
                tooltip=[alt.Tooltip("Crop:N"), alt.Tooltip("Probability:Q", format=".2%")]
            )
        )
        st.altair_chart(chart, use_container_width=True)

        # hi
