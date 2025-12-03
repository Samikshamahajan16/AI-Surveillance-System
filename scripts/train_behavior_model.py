import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

FEATURE_DIR = "features/hockey"

def load_csv_folder(folder_path, label):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            path = os.path.join(folder_path, file)
            with open(path) as f:
                reader = csv.reader(f)
                for row in reader:
                    features = list(map(float, row[:-1]))  # all features
                    data.append(features + [label])
    return data

fight_data = load_csv_folder(os.path.join(FEATURE_DIR, "fight"), 1)
nofight_data = load_csv_folder(os.path.join(FEATURE_DIR, "nofight"), 0)

dataset = fight_data + nofight_data
dataset = np.array(dataset)

X = dataset[:, :-1]
y = dataset[:, -1]

print("Dataset size:", len(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

acc = model.score(X_test, y_test)
print("Model Accuracy:", acc)

os.makedirs("models", exist_ok=True)
dump(model, "models/fight_behavior_model.joblib")

print("\n✔ Model saved at models/fight_behavior_model.joblib")
