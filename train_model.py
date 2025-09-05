# train_model.py
"""
Train a RandomForest-based crop recommendation pipeline from Crop_recommendation.csv.
Exports:
  - crop_recommender_rf.joblib
  - model_metadata.json
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, top_k_accuracy_score
from joblib import dump
import json

DATA_PATH = Path("Crop_recommendation.csv")
EXPORT_DIR = Path("export_model")
EXPORT_DIR.mkdir(exist_ok=True)

assert DATA_PATH.exists(), f"{DATA_PATH} not found. Place the CSV in the working folder."

df = pd.read_csv(DATA_PATH)
# Basic validation
expected_cols = {"N","P","K","temperature","humidity","ph","rainfall","label"}
if not expected_cols.issubset(set(df.columns)):
    raise SystemExit(f"CSV missing required columns. Found: {df.columns.tolist()}")

X = df[["N","P","K","temperature","humidity","ph","rainfall"]]
y = df["label"]

numeric_features = X.columns.tolist()
preprocessor = ColumnTransformer([("num", StandardScaler(), numeric_features)])

rf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline([("prep", preprocessor), ("model", rf)])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training model...")
pipe.fit(X_train, y_train)

# evaluation
y_pred = pipe.predict(X_test)
acc = accuracy_score(y_test, y_pred)
top3 = top_k_accuracy_score(y_test, pipe.predict_proba(X_test), k=3)

print("Accuracy (test):", acc)
print("Top-3 accuracy (test):", top3)
print("\nClassification report:\n")
print(classification_report(y_test, y_pred))

# feature importance (from the RF inside pipeline)
rf_model = pipe.named_steps["model"]
# get importance for scaled numeric features (order preserved)
feat_importances = dict(zip(numeric_features, rf_model.feature_importances_.tolist()))

# save model
model_path = EXPORT_DIR / "crop_recommender_rf.joblib"
dump(pipe, model_path)
print("Saved model to:", model_path)

# save metadata
meta = {
    "features": numeric_features,
    "n_classes": len(pipe.classes_),
    "classes": pipe.classes_.tolist(),
    "feature_importances": feat_importances,
    "test_accuracy": float(acc),
    "test_top3_accuracy": float(top3)
}
meta_path = EXPORT_DIR / "model_metadata.json"
meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("Saved metadata to:", meta_path)

print("Training complete.")
