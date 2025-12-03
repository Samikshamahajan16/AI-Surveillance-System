# scripts/train_catboost_from_xgb.py

import numpy as np
from catboost import CatBoostClassifier
import joblib
import os

DATASET = "features/xgb_dataset_full.npz"

OUTPUT_MODEL = "models/catboost_behavior.cbm"
OUTPUT_SCALER = "models/catboost_scaler.joblib"   # optional if scaler exists

# Create model directory
os.makedirs("models", exist_ok=True)

print("Loading dataset...")
data = np.load(DATASET)
X = data["X"]
y = data["y"]

print("Training CatBoost classifier...")
model = CatBoostClassifier(
    iterations=800,
    depth=8,
    learning_rate=0.03,
    loss_function="MultiClass",
    verbose=False
)

model.fit(X, y)

print("Saving CatBoost .cbm model...")
model.save_model(OUTPUT_MODEL, format="cbm")

# If scaler exists from XGB pipeline, load & save it
if os.path.exists("models/scaler_xgb_full.joblib"):
    scaler = joblib.load("models/scaler_xgb_full.joblib")
    joblib.dump(scaler, OUTPUT_SCALER)

print(" CatBoost training complete!")
print("Saved model:", OUTPUT_MODEL)
