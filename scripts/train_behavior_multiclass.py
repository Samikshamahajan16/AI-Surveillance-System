import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from joblib import dump

FEATURE_DIR = "features_advanced"

CLASSES = {
    "fight": 1,
    "nofight": 0
    # add more classes later like:
    # "running": 2,
    # "walking": 3,
    # "standing": 4
}

def load_folder(folder_path, label):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".npy"):
            arr = np.load(os.path.join(folder_path, file))
            arr_mean = np.mean(arr, axis=0)
            data.append((arr_mean, label))
    return data

dataset = []

for cls, label in CLASSES.items():
    folder = os.path.join(FEATURE_DIR, cls)
    if os.path.exists(folder):
        print(f"Loading {cls}...")
        dataset.extend(load_folder(folder, label))
    else:
        print(f"⚠ Warning: folder missing → {folder}")

X = np.array([d[0] for d in dataset])
y = np.array([d[1] for d in dataset])

print("Dataset size:", len(y))
print("Feature vector size:", X.shape[1])

# Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Train XGBoost classifier
model = XGBClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42,
    n_jobs=4
)

model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print("Model Accuracy:", acc)

# Save model + scaler
os.makedirs("models", exist_ok=True)
dump(model, "models/behavior_model_xgb.joblib")
dump(scaler, "models/behavior_scaler.joblib")

print("\n✔ Model saved!")
print("✔ Scaler saved!")
