# scripts/train_xgb_full.py
import numpy as np
import joblib
from xgboost import XGBClassifier

DATASET_PATH = "features/xgb_dataset_full.npz"
MODEL_OUT = "models/xgb_behavior_full.joblib"

data = np.load(DATASET_PATH)
X, y = data["X"], data["y"]

pos = sum(y == 1)
neg = sum(y == 0)
balance_ratio = neg / pos

model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=0.001,
    reg_lambda=1,
    gamma=0.2,
    scale_pos_weight=balance_ratio,   # handle imbalance
    eval_metric="logloss",
)

model.fit(X, y)
joblib.dump(model, MODEL_OUT)

print("✔ XGBoost Model Trained and Saved!")
