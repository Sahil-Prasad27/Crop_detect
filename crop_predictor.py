"""
Offline crop recommendation helper.
"""

from joblib import load
import numpy as np
import pandas as pd
import json
from pathlib import Path

# Feature names (must match training pipeline)
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_model(path="crop_recommender_rf.joblib"):
    """
    Load the trained crop recommendation model.
    """
    return load(path)


def load_metadata(path="metadata.json"):
    """
    Load model metadata from a JSON file if available.
    Returns dict with info like features, description, etc.
    """
    meta_path = Path(path)
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        # fallback metadata
        return {
            "features": FEATURES,
            "description": "AI-based Crop Recommendation Model (RandomForest)",
            "version": "1.0"
        }


def recommend_topk(model, N, P, K, temperature, humidity, ph, rainfall, k=5):
    """
    Recommend top-k crops given soil and weather inputs.
    """
    # Wrap inputs into a DataFrame with proper column names
    x = pd.DataFrame(
        [[N, P, K, temperature, humidity, ph, rainfall]],
        columns=FEATURES
    )

    # Predict probabilities for all crops
    proba = model.predict_proba(x)[0]
    labels = model.classes_

    # Get indices of top-k crops
    idx = np.argsort(proba)[::-1][:k]

    topk = [(labels[i], float(proba[i])) for i in idx]
    return topk, proba, labels
