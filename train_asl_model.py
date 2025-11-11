import json
import os
import numpy as np
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# --------------------- Paths --------------------- #
dataset_path = Path("dataset/asl_data.json")
model_dir = Path("model")
model_dir.mkdir(exist_ok=True)
model_file = model_dir / "asl_knn_model.joblib"
encoder_file = model_dir / "label_encoder.joblib"

# ------------------- Load JSON ------------------- #
if not dataset_path.exists():
    raise FileNotFoundError(f"[❌] Dataset not found at {dataset_path}")

with open(dataset_path, "r") as f:
    data = json.load(f)

X = [sample["landmarks"] for sample in data]
y = [sample["label"] for sample in data]

X = np.array(X)
y = np.array(y)

# ----------------- Encode Labels ----------------- #
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# ------------------- Split Data ------------------ #
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# ---------------- Train Classifier --------------- #
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# ------------------- Evaluate -------------------- #
y_pred = model.predict(X_test)
print("\n[Model Evaluation]")
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# ------------------- Save Model ------------------ #
joblib.dump(model, model_file)
joblib.dump(encoder, encoder_file)

print(f"\n[✅] Model saved to:     {model_file}")
print(f"[✅] Label encoder saved: {encoder_file}")
