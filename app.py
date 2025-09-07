import streamlit as st
from pathlib import Path
import pandas as pd
import altair as alt
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import ConnectionError as RequestsConnectionError
from crop_predictor import FEATURES, load_model, recommend_topk, load_metadata
import time
import sys, socket
import streamlit.components.v1 as components

# -------------------------------
# Paths
# -------------------------------
EXPORT_DIR = Path("export_model")
MODEL_PATH = EXPORT_DIR / "crop_recommender_rf.joblib"
META_PATH = EXPORT_DIR / "model_metadata.json"
NPK_CSV_PATH = "state_npk.csv"

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="AI Crop Recommendation 🌱",
    layout="wide",
    page_icon="🌾",
)

# -------------------------------
# CSS for Transparent UI + Top-right Language
# -------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, rgba(0,0,0,0.6), rgba(0,0,0,0.6)), url('https://images.unsplash.com/photo-1501004318641-b39e6451bec6?auto=format&fit=crop&w=1920&q=80') no-repeat center center fixed;
    background-size: cover;
    color: #ffffff;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    overflow-x: hidden;
}
section, .stButton, .stNumberInput, .stTextInput, .stSelectbox, .stSlider {
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
div.stNumberInput>label, div.stTextInput>label, div.stSelectbox>label, div.stSlider>label {
    color: #aaff00;
    font-weight: 600;
    font-size: 1rem;
}
.stSlider .st-cx {
    background: #aaff00 !important;
}
.stSlider .st-cy {
    background: #ffffff !important;
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

/* Top-right language selector */
div[data-baseweb="select"] {
    position: absolute;
    top: 15px;
    right: 15px;
    width: 150px;
    z-index: 999;
}
</style>
""", unsafe_allow_html=True)



# -------------------------------
# Weather API
# -------------------------------
API_KEY = "bc1491c4cbf3fec53b1bed7c55c63482"
BASE_URL = "http://api.openweathermap.org/data/2.5/weather"

def get_weather_data(location):
    try:
        session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        res = session.get(BASE_URL, params={"q": location, "appid": API_KEY, "units": "metric"}, timeout=30)
        data = res.json()
        if res.status_code != 200:
            return None, f"❌ Error: {data.get('message', 'Unknown error')}"
        temp = data["main"]["temp"]
        hum = data["main"]["humidity"]
        rain = data.get("rain", {}).get("1h", 0) or data.get("rain", {}).get("3h", 0)
        return {"temperature": temp, "humidity": hum, "rainfall": rain, "coord": data["coord"]}, None
    except RequestsConnectionError as e:
        return None, f"Error fetching weather data: Connection aborted ({str(e)}). Please check your internet connection or try again later."
    except Exception as e:
        return None, f"Error fetching weather data: {str(e)}. Please check your internet connection or try again later."

# -------------------------------
# Soil API
# -------------------------------
SOIL_URL = "https://rest.isric.org/soilgrids/v2.0/properties/query"

def get_soil_data(lat, lon):
    default_soil = {"N": 50.0, "pH": 6.5, "P": None, "K": None}
    try:
        session = requests.Session()
        retries = Retry(total=10, backoff_factor=2,
                        status_forcelist=[429, 500, 502, 503, 504, 520, 521, 522, 524],
                        allowed_methods=["GET"])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        headers = {'User-Agent': 'Mozilla/5.0'}
        params = [
            ("lon", lon),
            ("lat", lat),
            ("property", "phh2o"),
            ("property", "nitrogen"),
            ("depth", "0-5cm"),
            ("value", "mean")
        ]
        for attempt in range(3):
            try:
                res = session.get(SOIL_URL, params=params, headers=headers, timeout=60)
                if res.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                elif res.status_code != 200:
                    return default_soil, f"Soil API error: {res.status_code}"
                else:
                    break
            except RequestsConnectionError:
                time.sleep(2 ** attempt)
                continue
        else:
            return default_soil, "Soil API unavailable"

        data = res.json()
        layers = data.get("properties", {}).get("layers", [])
        ph_val, n_val = None, None
        for layer in layers:
            name = layer.get("name", "")
            depth_vals = layer.get("depths", [{}])[0].get("values", {})
            mean_val = depth_vals.get("mean")
            if mean_val is not None:
                if name == "phh2o":
                    ph_val = mean_val / 10.0
                elif name == "nitrogen":
                    n_val = mean_val / 100.0
        soil_data = {
            "N": round(n_val * 100, 2) if n_val is not None else default_soil["N"],
            "pH": round(ph_val, 2) if ph_val is not None else default_soil["pH"],
            "P": default_soil["P"],
            "K": default_soil["K"]
        }
        return soil_data, None
    except Exception as e:
        return default_soil, str(e)

# -------------------------------
# Load Model & Metadata
# -------------------------------
with st.spinner("Loading model..."):
    try:
        model = load_model(MODEL_PATH.as_posix())
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        model = None

try:
    metadata = load_metadata(META_PATH.as_posix())
except Exception as e:
    st.error(f"Failed to load metadata: {e}")
    metadata = {}

# -------------------------------
# Load NPK CSV
# -------------------------------
try:
    df_npk = pd.read_csv(NPK_CSV_PATH)
    df_npk.columns = df_npk.columns.str.strip()
    df_npk = df_npk[df_npk["State/UT"] != "All India"]
    npks_column = "NPKS - Availability from 01-04-2024 to 26-03-2025"
    states = ["Select a State"] + sorted(df_npk["State/UT"].astype(str).str.strip().unique())
except Exception as e:
    st.error(f"Failed to load state-wise NPK data: {e}")
    df_npk = pd.DataFrame(columns=["State/UT"])
    states = ["Select a State"]

# -------------------------------
# Translations
# -------------------------------
trans = {
    "English": {
        "hero_title": "Smart Farming for a Sustainable Future 🌍",
        "hero_sub": "Harness AI to optimize crop selection and boost yields sustainably.",
        "enter_location": "Enter Location (City / District / Village)",
        "select_state": "Select State/UT for P and K (Average Values)",
        "fetch_weather": "Weather Data Fetched",
        "fetch_soil": "Soil Data Fetched",
        "enter_soil": "Enter Soil Nutrient Data",
        "Nitrogen": "Nitrogen (N)",
        "Phosphorus": "Phosphorus (P)",
        "Potassium": "Potassium (K)",
        "pH": "Soil pH",
        "Temperature": "Temperature (°C)",
        "Humidity": "Humidity (%)",
        "Rainfall": "Rainfall (mm)",
        "top_k": "Number of Top Recommendations",
        "recommend": "Get Crop Recommendations 🌾",
        "top_crops": "✅ Top Recommended Crops",
        "probabilities": "📊 Prediction Probabilities",
        "no_state": "Please select a valid state.",
        "model_not_loaded": "Model is not loaded. Cannot make predictions.",
        "fetch_soil_button": "🌱 Fetch Soil Data"
    },
    "हिंदी": {
        "hero_title": "स्थायी भविष्य के लिए स्मार्ट खेती 🌍",
        "hero_sub": "एआई का उपयोग करके फसल चयन को अनुकूलित करें और उपज बढ़ाएँ।",
        "enter_location": "स्थान दर्ज करें (शहर / जिला / गाँव)",
        "select_state": "P और K के लिए राज्य/UT चुनें (औसत मूल्य)",
        "fetch_weather": "मौसम डेटा प्राप्त हुआ",
        "fetch_soil": "मिट्टी डेटा प्राप्त हुआ",
        "enter_soil": "मिट्टी की पोषक जानकारी दर्ज करें",
        "Nitrogen": "नाइट्रोजन (N)",
        "Phosphorus": "फॉस्फोरस (P)",
        "Potassium": "पोटेशियम (K)",
        "pH": "मिट्टी का पीएच",
        "Temperature": "तापमान (°C)",
        "Humidity": "आर्द्रता (%)",
        "Rainfall": "वर्षा (mm)",
        "top_k": "शीर्ष सिफारिशों की संख्या",
        "recommend": "फसल सुझाव प्राप्त करें 🌾",
        "top_crops": "✅ शीर्ष सिफारिश की गई फसलें",
        "probabilities": "📊 भविष्यवाणी संभावनाएँ",
        "no_state": "कृपया एक मान्य राज्य चुनें।",
        "model_not_loaded": "मॉडल लोड नहीं हुआ। भविष्यवाणियाँ नहीं की जा सकतीं।",
        "fetch_soil_button": "🌱 मिट्टी डेटा प्राप्त करें"
    }
}

crop_trans = {
    "rice": "चावल", "wheat": "गेहूँ", "maize": "मक्का",
    "chickpea": "चना", "kidneybeans": "राजमा", "pigeonpeas": "अरहर",
    "mothbeans": "मटकी", "mungbean": "मूँग", "blackgram": "उड़द",
    "lentil": "मसूर", "pomegranate": "अनार", "banana": "केला",
    "mango": "आम", "grapes": "अंगूर", "watermelon": "तरबूज",
    "muskmelon": "खरबूजा", "apple": "सेब", "orange": "संतरा",
    "papaya": "पपीता", "coconut": "नारियल", "cotton": "कपास",
    "jute": "जूट", "coffee": "कॉफ़ी"
}

# -------------------------------
# Session State
# -------------------------------
if "weather_data" not in st.session_state:
    st.session_state.weather_data = None
if "soil_data" not in st.session_state:
    st.session_state.soil_data = None
if "last_soil_call" not in st.session_state:
    st.session_state.last_soil_call = 0

# -------------------------------
# Language Selector
# -------------------------------
lang = st.selectbox("", ["English", "हिंदी"], key="language_selector")
t = trans[lang]


# -------------------------------
# Hero Section (Top-Center)
# -------------------------------
st.markdown(f"""
<div style="text-align:center; margin-top:50px;">
    <h1>{t['hero_title']}</h1>
    <p style="font-size:1.2rem; color:#e0e0e0; margin-top:10px;">{t['hero_sub']}</p>
</div>
""", unsafe_allow_html=True)

# st.markdown(f"""
# <div style="text-align:center; margin-top:50px;">
#     <h1>{t['hero_title']}</h1>
#     <p style="font-size:1.2rem; color:#e0e0e0; margin-top:10px;">{t['hero_sub']}</p>
# </div>
# """, unsafe_allow_html=True)

# -------------------------------
# Hyperlink Button — Below Hero
# -------------------------------

st.markdown("""
<style>
/* Target page link container */
a[data-testid="stPageLink-NavLink"] {
    display: inline-block;
    background: linear-gradient(90deg, #ff6f00, #ffa000);
    color: white !important;
    padding: 12px 30px;
    font-weight: 600;
    border-radius: 50px;
    text-decoration: none !important;
    box-shadow: 0 4px 15px rgba(255, 105, 0, 0.3);
    transition: all 0.3s ease;
    font-size: 1rem;
}


/* Hover effect */
a[data-testid="stPageLink-NavLink"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4);
}
</style>
""", unsafe_allow_html=True)

st.page_link("pages/Chat.py", label="🌐 Chat bot")
st.page_link("pages/Diseases.py", label="🍂 Go to Disease Prediction")

# -------------------------------
# Weather and Soil Data Section
# -------------------------------
st.header(t["enter_location"])
col1, col2 = st.columns([1, 1], gap="medium")

with col1:
    location = st.text_input(t["enter_location"], key="location_input")
    state = st.selectbox(t["select_state"], states, key="state_select")

# Fetch P and K from selected state
P_default, K_default = 50.0, 50.0
if state != "Select a State" and not df_npk.empty:
    state_match = df_npk[df_npk["State/UT"].str.lower() == state.lower()]
    if not state_match.empty:
        npks_value = float(state_match[npks_column].values[0])
        P_default = npks_value * 0.4  # Estimate P as 40% of NPKS
        K_default = npks_value * 0.6  # Estimate K as 60% of NPKS
        st.warning(f"Note: Phosphorus (P) and Potassium (K) are estimated from NPKS availability ({npks_value}) as P={P_default:.2f}, K={K_default:.2f} for state {state}. For better accuracy, input specific P and K values manually.")
        st.write(f"Debug: Estimated P={P_default:.2f}, K={K_default:.2f} from NPKS for state {state}")
    else:
        st.warning(f"No data found for state: {state}. Using defaults (P=50, K=50).")
else:
    st.warning(t["no_state"])

# Fetch weather data
with col2:
    if location:
        with st.spinner("Fetching weather data..."):
            weather_data, error = get_weather_data(location)
            if error:
                st.error(error)
            else:
                st.session_state.weather_data = weather_data
                st.success(f"✅ {t['fetch_weather']} for {location}")
                st.json(weather_data)
                st.write("Debug: Weather data stored in session state")

                if st.button(t["fetch_soil_button"], key="fetch_soil_button"):
                    with st.spinner("Fetching soil data from SoilGrids..."):
                        # Respect rate limit (5 calls/minute ~ 12s delay)
                        now = time.time()
                        if now - st.session_state.last_soil_call < 12:
                            time.sleep(12 - (now - st.session_state.last_soil_call))
                        st.session_state.last_soil_call = time.time()
                        
                        lat, lon = weather_data["coord"]["lat"], weather_data["coord"]["lon"]
                        soil_data, s_err = get_soil_data(lat, lon)
                        if s_err:
                            st.error(s_err)
                        else:
                            st.session_state.soil_data = soil_data
                            st.success(f"✅ {t['fetch_soil']}")
                            st.json(soil_data)
                            st.write("Debug: Soil data stored in session state")

# Use session state for soil and weather data
weather_data = st.session_state.weather_data
soil_data = st.session_state.soil_data

# -------------------------------
# Soil and Environmental Inputs
# -------------------------------
st.markdown(f"### {t['enter_soil']}")
colN, colP, colK, colPH = st.columns(4)
with colN:
    N = st.number_input(
        f"{t['Nitrogen']} (mg/kg)",
        min_value=0.0,
        max_value=1000.0,
        value=soil_data["N"] if soil_data and soil_data["N"] is not None else 50.0,
        step=0.1,
        format="%.2f",
        key="nitrogen_input"
    )
with colP:
    P = st.number_input(
        f"{t['Phosphorus']} (mg/kg)",
        min_value=0.0,
        max_value=1000.0,
        value=P_default,
        step=0.1,
        format="%.2f",
        key="phosphorus_input"
    )
with colK:
    K = st.number_input(
        f"{t['Potassium']} (mg/kg)",
        min_value=0.0,
        max_value=1000.0,
        value=K_default,
        step=0.1,
        format="%.2f",
        key="potassium_input"
    )
with colPH:
    pH = st.number_input(
        f"{t['pH']} (unitless)",
        min_value=0.0,
        max_value=14.0,
        value=soil_data["pH"] if soil_data and soil_data["pH"] is not None else 6.5,
        step=0.01,
        format="%.2f",
        key="ph_input"
    )

colT, colH, colR = st.columns(3)
with colT:
    temperature = st.number_input(
        f"{t['Temperature']}",
        min_value=-10.0,
        max_value=100.0,
        value=weather_data["temperature"] if weather_data else 25.0,
        step=0.1,
        format="%.1f",
        key="temperature_input"
    )
with colH:
    humidity = st.number_input(
        f"{t['Humidity']}",
        min_value=0.0,
        max_value=100.0,
        value=float(weather_data["humidity"]) if weather_data else 70.0,
        step=0.1,
        format="%.1f",
        key="humidity_input"
    )
with colR:
    rainfall = st.number_input(
        f"{t['Rainfall']}",
        min_value=0.0,
        max_value=1000.0,
        value=float(weather_data["rainfall"]) if weather_data else 100.0,
        step=0.1,
        format="%.1f",
        key="rainfall_input"
    )

# -------------------------------
# Crop Recommendation
# -------------------------------
top_k = st.slider(t["top_k"], min_value=1, max_value=10, value=3, key="top_k_slider")
if st.button(t["recommend"], key="recommend_button"):
    if model is None:
        st.error(t["model_not_loaded"])
    elif state == "Select a State":
        st.error(t["no_state"])
    else:
        with st.spinner("Generating crop recommendations..."):
            try:
                topk, proba, labels = recommend_topk(
                    model,
                    N=N, P=P, K=K,
                    temperature=temperature,
                    humidity=humidity,
                    ph=pH,
                    rainfall=rainfall,
                    k=top_k
                )
                st.write("Debug: Recommendations generated successfully")

                st.subheader(t["top_crops"])
                for crop, score in topk:
                    display_crop = crop_trans[crop] if lang == "हिंदी" and crop in crop_trans else crop
                    st.markdown(f"- **{display_crop}** ({score:.2%})")

                st.subheader(t["probabilities"])
                df_proba = pd.DataFrame({"Crop": labels, "Probability": proba})
                df_top = df_proba.sort_values("Probability", ascending=False).head(top_k)
                if lang == "हिंदी":
                    df_top["Crop"] = df_top["Crop"].apply(lambda c: crop_trans[c] if c in crop_trans else c)

                chart = (
                    alt.Chart(df_top)
                    .mark_bar()
                    .encode(
                        x=alt.X("Probability:Q", title="Probability", axis=alt.Axis(format="%", titleColor="#ffffff")),
                        y=alt.Y("Crop:N", title="Crop", sort="-x"),
                        color=alt.ColorValue("#aaff00"),
                        tooltip=[
                            alt.Tooltip("Crop:N", title="Crop"),
                            alt.Tooltip("Probability:Q", title="Probability", format=".2%")
                        ]
                    )
                    .configure_axis(
                        labelColor="#ffffff",
                        titleColor="#ffffff",
                        gridColor="rgba(255, 255, 255, 0.2)"
                    )
                    .configure_view(stroke=None)
                )
                st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating recommendations: {str(e)}")
                st.write(f"Debug: Input values - N={N}, P={P}, K={K}, pH={pH}, Temp={temperature}, Humidity={humidity}, Rainfall={rainfall}")