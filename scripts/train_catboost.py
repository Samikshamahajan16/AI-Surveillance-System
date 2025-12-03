# scripts/train_catboost.py
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import os

FEATURE_DIR = "features"
MODEL_OUT = "models/catboost_behavior.joblib"
SCALER_OUT = "models/scaler_catboost.joblib"
os.makedirs("models", exist_ok=True)

X = np.load(f"{FEATURE_DIR}/clip_features.npy")
y = np.load(f"{FEATURE_DIR}/clip_labels.npy")

# shuffle & split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    loss_function='Logloss',
    eval_metric='F1',
    random_seed=42,
    verbose=100,
    early_stopping_rounds=50
)

model.fit(X_train_s, y_train, eval_set=(X_test_s, y_test))

y_pred = model.predict(X_test_s)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

joblib.dump(model, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
print("Saved model and scaler.")
