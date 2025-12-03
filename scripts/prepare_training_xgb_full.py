# scripts/prepare_training_xgb_full.py
import numpy as np
import joblib
import os

FEATURE_DIR = "features"
OUT_DATASET = "features/xgb_dataset_full.npz"
OUT_SCALER = "models/scaler_xgb_full.joblib"

fight = np.load(os.path.join(FEATURE_DIR, "fight_features.npy"))
nofight = np.load(os.path.join(FEATURE_DIR, "nofight_features.npy"))

X = np.vstack([fight, nofight])
y = np.array([1]*len(fight) + [0]*len(nofight))

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, OUT_SCALER)
np.savez(OUT_DATASET, X=X_scaled, y=y)

print("✔ Dataset prepared")
print("Total samples:", len(X))
print("Feature size:", X.shape[1])
