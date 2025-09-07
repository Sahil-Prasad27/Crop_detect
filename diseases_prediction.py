import numpy as np
import json
import tensorflow as tf
from pathlib import Path
from tensorflow.keras.preprocessing import image

DISEASE_MODEL_PATH = "plant_disease_model.h5"
LABELS_PATH = "class_labels.json"

# Load CNN model
try:
    disease_model = tf.keras.models.load_model(DISEASE_MODEL_PATH)
except Exception as e:
    print(f"⚠️ Could not load disease model: {e}")
    disease_model = None

# Load class labels
if Path(LABELS_PATH).exists():
    with open(LABELS_PATH, "r") as f:
        class_labels = json.load(f)
else:
    class_labels = ["Healthy", "Powdery", "Rust"]  # fallback classes


def predict_disease(img_path):
    """
    Predict plant disease from an image.
    Returns predicted class and probability distribution.
    """
    if disease_model is None:
        raise RuntimeError("Disease detection model not loaded.")

    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = disease_model.predict(img_array)
    predicted_index = np.argmax(prediction)
    predicted_class = class_labels[predicted_index]
    prediction_probs = float(prediction[0][predicted_index]) * 100

    return predicted_class, prediction_probs
