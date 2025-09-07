import streamlit as st
import os
from diseases_prediction import predict_disease

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="🩺 Disease Prediction",
    layout="wide",
    page_icon="🌾",
)

# -------------------------------
# CSS for Transparent UI
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, rgba(0,0,0,0.6), rgba(0,0,0,0.6)), 
                url('https://images.unsplash.com/photo-1501004318641-b39e6451bec6?auto=format&fit=crop&w=1920&q=80') 
                no-repeat center center fixed;
    background-size: cover;
    color: #ffffff;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    overflow-x: hidden;
}
section, .stButton, .stNumberInput, .stTextInput, .stSelectbox, .stSlider, .stFileUploader {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(12px);
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
    color: #ffffff;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.15);
}
h1 {
    color: #aaff00;
    font-size: 3rem;
    text-align: center;
    text-shadow: 0 2px 8px rgba(0, 0, 0, 0.6);
    animation: fadeIn 1.5s ease-in-out;
}
h2 {
    color: #ffffff;
    font-size: 1.8rem;
    margin-top: 20px;
    text-shadow: 0 1px 3px rgba(0, 0, 0, 0.4);
}
.stButton>button {
    background: linear-gradient(90deg, #aaff00 0%, #76ff03 100%);
    color: #000;
    font-weight: 600;
    border-radius: 10px;
    padding: 0.7em 1.5em;
    font-size: 1rem;
    border: none;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: translateY(-3px);
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
}
div.stFileUploader>label {
    color: #aaff00;
    font-weight: 600;
    font-size: 1rem;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-15px); }
    to { opacity: 1; transform: translateY(0); }
}
@media (max-width: 768px) {
    h1 { font-size: 2.2rem; }
    h2 { font-size: 1.5rem; }
    .stButton>button { padding: 0.5em 1em; font-size: 0.9rem; }
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Translations
# -------------------------------
trans = {
    "English": {
        "hero_title": "🌱 Crop Disease Detection",
        "hero_sub": "Upload an image of a crop leaf to detect possible diseases.",
        "upload": "Upload Crop Image",
        "predict": "🔍 Predict Disease",
        "result": "🩺 Disease Prediction",
        "probabilities": "📊 Probabilities"
    },
    "हिंदी": {
        "hero_title": "🌱 फसल रोग पहचान",
        "hero_sub": "फसल पत्ते की छवि अपलोड करें और संभावित रोगों की पहचान करें।",
        "upload": "फसल छवि अपलोड करें",
        "predict": "🔍 रोग पहचानें",
        "result": "🩺 रोग की भविष्यवाणी",
        "probabilities": "📊 संभावनाएँ"
    }
}

# -------------------------------
# Language Selector
# -------------------------------
lang = st.selectbox("", ["English", "हिंदी"], key="language_selector")
t = trans[lang]

# -------------------------------
# Hero Section
# -------------------------------
st.markdown(f"""
<div style="text-align:center; margin-top:50px;">
    <h1>{t['hero_title']}</h1>
    <p style="font-size:1.2rem; color:#e0e0e0; margin-top:10px;">{t['hero_sub']}</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# File Upload
# -------------------------------
# -------------------------------
# File Upload OR Camera Capture
# -------------------------------
# -------------------------------
# Choose Input Method
# -------------------------------
option = st.radio(
    "Choose Image Source:",
    ["📤 Upload Photo", "📸 Take Photo with Camera"],
    horizontal=True
)

image_source = None

if option == "📤 Upload Photo":
    uploaded_file = st.file_uploader(t["upload"], type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_source = uploaded_file

elif option == "📸 Take Photo with Camera":
    camera_file = st.camera_input("📷 Take a Picture")
    if camera_file is not None:
        image_source = camera_file

# -------------------------------
# Save and Predict
# -------------------------------
if image_source is not None:
    os.makedirs("uploads", exist_ok=True)
    img_path = f"uploads/{image_source.name}"
    with open(img_path, "wb") as f:
        f.write(image_source.getbuffer())

    st.image(img_path, caption="Selected Image", use_column_width=True)

    if st.button(t["predict"]):
        with st.spinner("Analyzing image..."):
            predicted_class, prediction_probs = predict_disease(img_path)

        st.subheader(t["result"])
        st.success(f"*Prediction:* {predicted_class} ({prediction_probs:.2f}%)")

        st.subheader(t["probabilities"])
        st.write(prediction_probs)
