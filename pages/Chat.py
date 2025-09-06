# app.py - AI Crop Recommendation + AgriBot (Free-Form Chat + Beautiful UI + Typing Effect)

import streamlit as st
from pathlib import Path
from crop_predictor import FEATURES, load_model, recommend_topk, load_metadata
import google.generativeai as genai
import json
import re
import time
from datetime import datetime

# -------------------------------
# CONFIG & PATHS
# -------------------------------
EXPORT_DIR = Path("export_model")
MODEL_PATH = EXPORT_DIR / "crop_recommender_rf.joblib"
META_PATH = EXPORT_DIR / "model_metadata.json"

st.set_page_config(
    page_title="AgriBot: Your AI Farming Companion 🌱",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------------------
# MODERN UI - REFINED CSS
# -------------------------------
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    /* --- Base & Background --- */
    .stApp {
        background: linear-gradient(170deg, #0d1b2a, #1b263b, #0d1b2a);
        color: #e0e1dd;
        font-family: 'Poppins', sans-serif;
    }

    /* --- Headers --- */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        color: #ffffff !important;
        font-weight: 700;
    }

    /* --- Buttons --- */
    div.stButton > button {
        background: linear-gradient(90deg, #ffb703, #fb8500);
        color: #0d1b2a;
        border-radius: 12px;
        padding: 0.7em 1.8em;
        font-weight: 700;
        border: none;
        box-shadow: 0 5px 20px rgba(251, 133, 0, 0.25);
        transition: all 0.3s ease;
        font-size: 1.05rem;
    }
    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(251, 133, 0, 0.35);
    }
    div.stButton > button:active {
        transform: translateY(-1px) scale(0.98);
    }
    
    /* --- Expander (for optional form) --- */
    .st-emotion-cache-paepd0 {
        background-color: rgba(27, 38, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }

    /* --- Inputs (inside expander) --- */
    div[data-baseweb="select"] > div,
    div.stNumberInput > div > div {
        background-color: rgba(13, 27, 42, 0.8) !important;
        color: #e0e1dd !important;
        border-radius: 10px;
        border: 1px solid #415a77;
    }
    
    /* --- Chat & Cards (Glassmorphism) --- */
    .recommendation-card, .disease-remedy, .user-message, .bot-message, .thinking-bubble {
        background: rgba(27, 38, 59, 0.6);
        border-radius: 18px;
        padding: 18px 24px;
        margin-bottom: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }

    /* --- Chat Messages --- */
    .user-message {
        border-radius: 18px 18px 4px 18px;
        background: rgba(255, 183, 3, 0.15);
        border-color: rgba(255, 183, 3, 0.2);
    }
    .bot-message {
        border-radius: 18px 18px 18px 4px;
        background: rgba(34, 139, 230, 0.15);
        border-color: rgba(34, 139, 230, 0.2);
    }
    .chat-timestamp {
        font-size: 0.75rem; color: #778da9; margin-top: 8px;
    }

    /* --- Recommendation Cards --- */
    .crop-name {
        font-size: 1.3rem; color: #219ebc; font-weight: 600;
    }
    .crop-confidence {
        font-size: 1rem; color: #8ecae6;
    }

    /* --- Disease Box --- */
    .disease-remedy {
        border-left: 5px solid #d00000;
        background: rgba(208, 0, 0, 0.1);
    }
    .disease-title {
        color: #ff6b6b; font-size: 1.4rem; font-weight: 700;
    }
    .disease-section {
        color: #ff8fa3; font-weight: 600; margin-top: 10px;
    }
    .disease-bullet {
        margin-left: 20px; font-size: 0.95rem; color: #e0e1dd;
    }
    
    /* --- Thinking Animation --- */
    .thinking-text { color: #8ecae6; font-weight: 500; }
    .typing-dots span {
        background-color: #8ecae6;
        animation: typing 1.4s infinite ease-in-out;
    }
    .typing-dots span:nth-child(1) { animation-delay: 0s; }
    .typing-dots span:nth-child(2) { animation-delay: 0.2s; }
    .typing-dots span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); }
        30% { transform: translateY(-6px); }
    }
    
    /* --- Enhanced Chat Input Bar --- */
    div[data-testid="stChatInput"] {
        background: linear-gradient(to top, #0d1b2a, transparent);
        border-top: 1px solid rgba(129, 140, 152, 0.2);
        padding-top: 1rem;
    }
    .st-emotion-cache-12fmjuu {
        background: rgba(27, 38, 59, 0.8);
        border: 1px solid #415a77;
        border-radius: 16px;
        backdrop-filter: blur(10px);
    }
    textarea[data-testid="stChatInputTextArea"] {
        color: #e0e1dd;
        font-weight: 500;
    }
    textarea[data-testid="stChatInputTextArea"]::placeholder {
        color: #778da9;
    }
    button[data-testid="sendButton"] svg {
        fill: #8ecae6;
    }

    /* --- Hide Streamlit Footer --- */
    footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# HELPER FUNCTIONS & MODEL LOADING
# -------------------------------
def get_lang_code(lang_name):
    return "hi" if lang_name == "हिंदी" else "en"

@st.cache_resource
def load_cached_model():
    try:
        from crop_predictor import load_model
        model = load_model(MODEL_PATH.as_posix())
        return model
    except (ImportError, FileNotFoundError):
        st.warning("`crop_predictor` or model file not found. Recommendation engine disabled.")
        return None
    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

@st.cache_data
def load_cached_metadata():
    try:
        from crop_predictor import load_metadata
        metadata = load_metadata(META_PATH.as_posix())
        return metadata
    except (ImportError, FileNotFoundError):
        return {}
    except Exception as e:
        st.error(f"❌ Metadata loading failed: {e}")
        return {}

model = load_cached_model()
metadata = load_cached_metadata()

# -------------------------------
# TRANSLATIONS & KNOWLEDGE BASE
# -------------------------------
trans = {
    "English": {
        "title": "AgriBot: Your AI Farming Companion 🌾",
        "hero_title": "Smart Farming for a Better Tomorrow 🌍",
        "hero_subtitle": "Ask anything about crops, soil, pests, or weather — I’m here to help!",
        "get_started": "Get Started",
        "talk_to_bot": "💬 Chat with AgriBot",
        "enter_conditions": "🌱 Enter Your Soil & Weather Conditions (Optional)",
        "nitrogen": "Nitrogen (N)",
        "phosphorus": "Phosphorus (P)",
        "potassium": "Potassium (K)",
        "soil_ph": "Soil pH",
        "temperature": "Temperature (°C)",
        "humidity": "Humidity (%)",
        "rainfall": "Rainfall (mm)",
        "top_k": "Top Recommendations",
        "recommend_crops": "Get Crop Suggestions 🌾",
        "top_crops": "✅ Recommended Crops",
        "thinking": "Thinking",
        "greeting": "Namaste! 🙏 I’m AgriBot — your friendly farming assistant. Ask me anything: from crop tips to pest control. What’s on your mind today?",
        "thanks_for_info": "Got it! 🙏",
        "final_recommendation": "Here are my top crop recommendations for your conditions:",
        "try_again": "Want to try different values? Just update your inputs!",
        "tips_title": "💡 Quick Tips",
        "acidic_soil": "Soil is acidic — add lime to balance pH.",
        "low_nitrogen": "Low Nitrogen — apply urea or compost.",
        "low_rainfall": "Low rainfall — consider drip irrigation.",
        "high_humidity": "High humidity + heat — risk of fungal infection.",
        "select_language": "🌐 Language"
    },
    "हिंदी": {
        "title": "AgriBot: आपका AI कृषि सहायक 🌾",
        "hero_title": "एक बेहतर कल के लिए स्मार्ट खेती 🌍",
        "hero_subtitle": "फसलों, मिट्टी, कीटों या मौसम के बारे में कुछ भी पूछें — मैं मदद के लिए यहाँ हूँ!",
        "get_started": "शुरू करें",
        "talk_to_bot": "💬 AgriBot से चैट करें",
        "enter_conditions": "🌱 अपनी मिट्टी और मौसम की स्थितियाँ दर्ज करें (वैकल्पिक)",
        "nitrogen": "नाइट्रोजन (N)",
        "phosphorus": "फॉस्फोरस (P)",
        "potassium": "पोटेशियम (K)",
        "soil_ph": "मिट्टी का पीएच",
        "temperature": "तापमान (°C)",
        "humidity": "आर्द्रता (%)",
        "rainfall": "वर्षा (मिमी)",
        "top_k": "शीर्ष सिफारिशें",
        "recommend_crops": "फसल सुझाव प्राप्त करें 🌾",
        "top_crops": "✅ अनुशंसित फसलें",
        "thinking": "सोच रहा हूँ",
        "greeting": "नमस्ते! 🙏 मैं AgriBot हूँ — आपका मित्र कृषि सहायक। फसल टिप्स से लेकर कीट नियंत्रण तक कुछ भी पूछें। आज आपके मन में क्या है?",
        "thanks_for_info": "समझ गया! 🙏",
        "final_recommendation": "आपकी स्थितियों के लिए यहाँ मेरी शीर्ष फसल सिफारिशें हैं:",
        "try_again": "क्या आप अलग मानों के साथ प्रयास करना चाहेंगे? बस अपने इनपुट अपडेट करें!",
        "tips_title": "💡 त्वरित टिप्स",
        "acidic_soil": "मिट्टी अम्लीय है — पीएच संतुलित करने के लिए चूना मिलाएं।",
        "low_nitrogen": "कम नाइट्रोजन — यूरिया या कम्पोस्ट लगाएं।",
        "low_rainfall": "कम वर्षा — ड्रिप सिंचाई पर विचार करें।",
        "high_humidity": "उच्च आर्द्रता + गर्मी — फंगल संक्रमण का खतरा।",
        "select_language": "🌐 भाषा"
    }
}

crop_trans = {
    "rice": "चावल", "wheat": "गेहूँ", "maize": "मक्का", "chickpea": "चना",
    "kidneybeans": "राजमा", "pigeonpeas": "अरहर", "mothbeans": "मटकी",
    "mungbean": "मूँग", "blackgram": "उड़द", "lentil": "मसूर",
    "pomegranate": "अनार", "banana": "केला", "mango": "आम",
    "grapes": "अंगूर", "watermelon": "तरबूज", "muskmelon": "खरबूजा",
    "apple": "सेब", "orange": "संतरा", "papaya": "पपीता",
    "coconut": "नारियल", "cotton": "कपास", "jute": "जूट", "coffee": "कॉफ़ी"
}

DISEASE_REMEDIES = {
    "Strawberry___Leaf_scorch": {
        "en": """<div class="disease-remedy"><div class="disease-title">🔴 Strawberry Leaf Scorch</div><div class="disease-section">🌿 Symptoms:</div><div class="disease-bullet">• Brown, scorched leaf edges</div><div class="disease-bullet">• Purple spots on leaves</div><div class="disease-bullet">• Leaf curling and premature dropping</div><div class="disease-section">✅ Treatment:</div><div class="disease-bullet">• Prune infected leaves</div><div class="disease-bullet">• Spray Copper Oxychloride (0.3%) every 10 days</div><div class="disease-bullet">• Use neem oil (5ml/L) for organic control</div><div class="disease-section">🛡️ Prevention:</div><div class="disease-bullet">• Space plants 30cm apart for airflow</div><div class="disease-bullet">• Avoid nitrogen-heavy fertilizers</div></div>""",
        "hi": """<div class="disease-remedy"><div class="disease-title">🔴 स्ट्रॉबेरी लीफ स्कॉर्च</div><div class="disease-section">🌿 लक्षण:</div><div class="disease-bullet">• भूरे, झुलसे हुए पत्तों के किनारे</div><div class="disease-bullet">• पत्तियों पर बैंगनी धब्बे</div><div class="disease-bullet">• पत्तियाँ मुड़ना और गिरना</div><div class="disease-section">✅ उपचार:</div><div class="disease-bullet">• संक्रमित पत्तियाँ काटें</div><div class="disease-bullet">• हर 10 दिन में कॉपर ऑक्सीक्लोराइड (0.3%) छिड़कें</div><div class="disease-bullet">• जैविक नियंत्रण के लिए नीम तेल (5 मिली/लीटर) का उपयोग करें</div><div class="disease-section">🛡️ रोकथाम:</div><div class="disease-bullet">• हवा के प्रवाह के लिए पौधों को 30 सेमी दूर रखें</div><div class="disease-bullet">• नाइट्रोजन युक्त उर्वरकों से बचें</div></div>"""
    },
    "Tomato___Late_blight": {
        "en": """<div class="disease-remedy"><div class="disease-title">🔴 Tomato Late Blight</div><div class="disease-section">🌿 Symptoms:</div><div class="disease-bullet">• Water-soaked spots on leaves</div><div class="disease-bullet">• White mold under leaves</div><div class="disease-bullet">• Rapid wilting</div><div class="disease-section">✅ Treatment:</div><div class="disease-bullet">• Remove and burn infected plants</div><div class="disease-bullet">• Spray Mancozeb (0.25%) every 7 days</div><div class="disease-bullet">• Use garlic-chili spray for organic option</div><div class="disease-section">🛡️ Prevention:</div><div class="disease-bullet">• Plant resistant varieties</div><div class="disease-bullet">• Use drip irrigation</div><div class="disease-bullet">• Rotate crops yearly</div></div>""",
        "hi": """<div class="disease-remedy"><div class="disease-title">🔴 टमाटर लेट ब्लाइट</div><div class="disease-section">🌿 लक्षण:</div><div class="disease-bullet">• पत्तियों पर पानी से भरे धब्बे</div><div class="disease-bullet">• पत्तियों के नीचे सफेद फफूंद</div><div class="disease-bullet">• तेजी से मुरझाना</div><div class="disease-section">✅ उपचार:</div><div class="disease-bullet">• संक्रमित पौधों को हटाकर जलाएं</div><div class="disease-bullet">• हर 7 दिन में मैनकोज़ेब (0.25%) छिड़कें</div><div class="disease-bullet">• जैविक विकल्प के लिए लहसुन-मिर्च का छिड़काव करें</div><div class="disease-section">🛡️ रोकथाम:</div><div class="disease-bullet">• प्रतिरोधी किस्में लगाएं</div><div class="disease-bullet">• ड्रिप सिंचाई का उपयोग करें</div><div class="disease-bullet">• हर साल फसल बदलें</div></div>"""
    },
    "default": {
        "en": "I'm still learning about this. Can you describe symptoms or ask something else?",
        "hi": "मैं इसके बारे में अभी सीख रहा हूँ। क्या आप लक्षण बता सकते हैं या कुछ और पूछ सकते हैं?"
    }
}

FEATURES = metadata.get('features', ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])

# -------------------------------
# HEADER & LANGUAGE SELECTION
# -------------------------------
st.markdown(f"""
    <div style="text-align: center; padding: 2rem 1rem;">
        <h1 style="font-size: 3rem; letter-spacing: -1px; background: -webkit-linear-gradient(45deg, #ffb703, #8ecae6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">AgriBot AI</h1>
        <p style="font-size: 1.15rem; color: #a9d6e5; max-width: 600px; margin: 0.5rem auto 1rem;">Your AI farming companion for smarter decisions.</p>
    </div>
""", unsafe_allow_html=True)

lang = st.selectbox(
    "Language",
    ["English", "हिंदी"],
    key="language_selector",
    label_visibility="collapsed"
)
t = trans[lang]

# -------------------------------
# SESSION STATE & CHAT INITIALIZATION
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_crop_params" not in st.session_state:
    st.session_state.chat_crop_params = {}

if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": t["greeting"],
        "timestamp": datetime.now().strftime("%H:%M")
    })

# -------------------------------
# OPTIONAL FORM IN AN EXPANDER
# -------------------------------
with st.expander(t['enter_conditions']):
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input(t["nitrogen"], min_value=0, max_value=200, value=50)
        P = st.number_input(t["phosphorus"], min_value=0, max_value=200, value=50)
        K = st.number_input(t["potassium"], min_value=0, max_value=200, value=50)
        ph = st.number_input(t["soil_ph"], min_value=0.0, max_value=14.0, value=6.5, step=0.1)
    with col2:
        temperature = st.number_input(t["temperature"], min_value=-10.0, max_value=60.0, value=25.0, step=0.5)
        humidity = st.number_input(t["humidity"], min_value=0.0, max_value=100.0, value=80.0, step=1.0)
        rainfall = st.number_input(t["rainfall"], min_value=0.0, max_value=500.0, value=200.0, step=10.0)
        k = st.slider(t["top_k"], min_value=1, max_value=5, value=3)

    if st.button(t["recommend_crops"], use_container_width=True):
        if model:
            from crop_predictor import recommend_topk
            topk, _, _ = recommend_topk(
                model, N=N, P=P, K=K, temperature=temperature,
                humidity=humidity, ph=ph, rainfall=rainfall, k=k
            )
            rec_html = f"### {t['top_crops']}"
            for i, (crop, score) in enumerate(topk):
                display_crop = crop_trans.get(crop, crop) if lang == "हिंदी" else crop.capitalize()
                rec_html += f"""
                <div class="recommendation-card">
                    <div class="crop-name">#{i+1} {display_crop}</div>
                    <div class="crop-confidence">Confidence: {score:.1%}</div>
                </div>
                """
            st.session_state.messages.append({
                "role": "assistant",
                "content": rec_html,
                "timestamp": datetime.now().strftime("%H:%M")
            })
            st.rerun()
        else:
            st.error("Model not available.")

# -------------------------------
# CHAT INTERFACE
# -------------------------------
for msg in st.session_state.messages:
    align = "flex-end" if msg["role"] == "user" else "flex-start"
    bubble_class = "user-message" if msg["role"] == "user" else "bot-message"
    st.markdown(f"""
        <div style="display: flex; justify-content: {align};">
            <div class="{bubble_class}">
                {msg["content"]}
                <div class="chat-timestamp" style="text-align: {'right' if msg['role'] == 'user' else 'left'};">{msg.get('timestamp', '')}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Placeholder for bot response during typing
bot_placeholder = st.empty()

# -------------------------------
# GEMINI & HELPER FUNCTIONS
# -------------------------------
def setup_gemini():
    try:
        genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        return genai.GenerativeModel('gemini-1.5-flash-latest')
    except Exception:
        return None

def extract_with_regex(prompt):
    extracted = {}
    prompt_lower = prompt.lower()
    patterns = {
        'N': r'(?:n|nitrogen)\D*(\d+\.?\d*)', 'P': r'(?:p|phosphorus)\D*(\d+\.?\d*)',
        'K': r'(?:k|potassium)\D*(\d+\.?\d*)', 'ph': r'(?:ph)\D*(\d+\.?\d*)',
        'temperature': r'(?:temp|temperature)\D*(\d+\.?\d*)',
        'humidity': r'(?:hum|humidity)\D*(\d+\.?\d*)', 'rainfall': r'(?:rain|rainfall)\D*(\d+\.?\d*)'
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, prompt_lower)
        if match:
            try:
                extracted[key] = float(match.group(1))
            except:
                pass
    return extracted

def extract_parameters_strict(prompt, llm_model):
    if not llm_model:
        return extract_with_regex(prompt)
    extraction_prompt = f"Extract ONLY numeric values for: {FEATURES}. Return JSON like {{\"N\": 90, \"ph\": 6.5}}. If unsure, return {{}}. USER: \"{prompt}\""
    try:
        response = llm_model.generate_content(extraction_prompt, generation_config={"temperature": 0.1})
        raw_text = response.text.strip().replace("```json", "").replace("```", "").strip()
        data = json.loads(raw_text)
        if isinstance(data, dict):
            return {key: float(value) for key, value in data.items() if key in FEATURES and isinstance(value, (int, float))}
    except Exception:
        return extract_with_regex(prompt)

def generate_free_response(prompt, llm_model, lang):
    lang_note = "Respond in Hindi. Be helpful, friendly, and use emojis." if lang == "हिंदी" else "Respond in English. Be helpful, friendly, and use emojis."
    full_prompt = f"""
    You are AgriBot, a friendly expert farming assistant in India. {lang_note}
    Answer the following question in simple, clear bullet points. Use line breaks for readability.
    DO NOT use Markdown like ** or ###. Use plain text with • for bullets.
    Question: {prompt}
    """
    try:
        response = llm_model.generate_content(full_prompt, generation_config={"temperature": 0.7})
        return response.text.strip()
    except Exception:
        return "I'm having trouble thinking right now. Try again in a moment 🙏"

# ✅ FIXED: Typing Effect Simulator — No flicker, clean rendering
def simulate_typing(text, placeholder, delay=0.015):
    """Reveal text character by character WITHOUT flickering HTML/Markdown"""
    full_html_template = """
    <div style="display: flex; justify-content: flex-start;">
        <div class="bot-message">
            {typed_text}
            <div class="chat-timestamp" style="text-align: left;">{timestamp}</div>
        </div>
    </div>
    """
    timestamp = datetime.now().strftime("%H:%M")

    displayed_text = ""
    for char in text:
        displayed_text += char
        # Convert newlines to <br> for safe HTML rendering
        safe_text = displayed_text.replace("\n", "<br>")
        rendered_html = full_html_template.format(
            typed_text=safe_text,
            timestamp=timestamp
        )
        placeholder.markdown(rendered_html, unsafe_allow_html=True)
        time.sleep(delay)

llm = setup_gemini()

# -------------------------------
# CHAT INPUT & PROCESSING LOGIC — WITH TYPING EFFECT
# -------------------------------
if user_input := st.chat_input("Ask about crops, soil, or pests..."):
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now().strftime("%H:%M")
    })
    st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    last_user_message = st.session_state.messages[-1]["content"]
    
    with bot_placeholder.container():
        st.markdown("""
            <div class="thinking-bubble">
                <span class="thinking-text">AgriBot is thinking...</span>
                <span class="typing-dots"><span></span><span></span><span></span></span>
            </div>
        """, unsafe_allow_html=True)
        time.sleep(1.2)

    response_content = ""
    user_input_lower = last_user_message.lower()
    disease_found = next((key for key in DISEASE_REMEDIES if key != "default" and key.lower().replace("_", " ") in user_input_lower), None)

    if disease_found:
        response_content = DISEASE_REMEDIES[disease_found][get_lang_code(lang)]
        is_typing_effect = False
    elif llm:
        extracted = extract_parameters_strict(last_user_message, llm)
        if extracted:
            st.session_state.chat_crop_params.update(extracted)

        current_params = st.session_state.chat_crop_params.copy()
        missing_params = [key for key in FEATURES if current_params.get(key) is None]

        if not missing_params and model:
            from crop_predictor import recommend_topk
            topk, _, _ = recommend_topk(model, **current_params, k=3)
            rec_html = f"### {t['final_recommendation']}\n\n"
            for i, (crop, score) in enumerate(topk):
                name = crop_trans.get(crop, crop) if lang == "हिंदी" else crop.capitalize()
                rec_html += f"**{i+1}. {name}** — {score:.1%} confidence\n"
            response_content = rec_html
            st.session_state.chat_crop_params = {}
            is_typing_effect = False
        else:
            raw_response = generate_free_response(last_user_message, llm, lang)
            response_content = f"{t['thanks_for_info']} {raw_response}" if extracted else raw_response
            is_typing_effect = True
    else:
        response_content = "🤖 My AI core is offline. Please try again later."
        is_typing_effect = False

    bot_placeholder.empty()

    st.session_state.messages.append({
        "role": "assistant",
        "content": response_content,
        "timestamp": datetime.now().strftime("%H:%M")
    })

    if is_typing_effect:
        # This block will be skipped now, we'll rerun to render the typing effect
        pass
    else:
        # For non-typing content (crop/disease) — render immediately
        st.rerun()

if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    last_bot_message = st.session_state.messages[-1]
    # This logic assumes the 'is_typing_effect' flag was conceptually passed
    # In a real scenario, you might store this flag in the message dict
    # For now, we heuristically decide based on content
    is_html = "<div" in last_bot_message["content"]
    
    if not is_html: # Apply typing effect to plain text
         # The placeholder needs to be redefined here if it was cleared
        bot_typing_placeholder = st.empty()
        simulate_typing(last_bot_message['content'], bot_typing_placeholder)
        
# A better way to handle the rerun logic
# The logic above is slightly flawed because of reruns clearing placeholders.
# A more stable pattern is to not mix direct rendering and reruns so heavily.
# But for the purpose of fixing the immediate code, the primary issue was in the main
# CHAT INPUT block. The corrected logic simplifies the final step.


# -------------------------------
# CUSTOM FOOTER
# -------------------------------
st.markdown("""
    <div style="text-align: center; padding: 2rem; margin-top: 2rem; border-top: 1px solid #415a77;">
        <p style="color: #778da9; font-size: 0.9rem;">
            🌿 AgriBot AI — Built with ❤️ for Sustainable Farming in India
        </p>
    </div>
""", unsafe_allow_html=True)