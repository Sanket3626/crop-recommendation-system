import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

# ----------- Load dataset ------------
# Change path to your dataset
DATA_PATH = r"F:\Data Science\CropProject\Crop_recommendation.csv"

df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)

# Identify label column
label_col = "label"
feature_cols = [c for c in df.columns if c != label_col]

print("Features:", feature_cols)

X = df[feature_cols].values
y = df[label_col].values

# Save feature names for Streamlit
os.makedirs("models", exist_ok=True)
with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_cols, f)

# -------- Encode label --------
le = LabelEncoder()
y_enc = le.fit_transform(y)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# -------- scale features -------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# ---- split ----
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# -------- Train Model ---------
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# -------- Accuracy ----------
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))

# -------- Save Model ---------
with open("models/crop_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nAll model files saved in:")
print("F:\\Data Science\\CropProject\\models")
