import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle, os

# ----------------- Load dataset -----------------
DATA_PATH = "Crop_recommendation.csv"
df = pd.read_csv(DATA_PATH)

os.makedirs("models", exist_ok=True)

# ----------------- Prepare data -----------------
feature_cols = [c for c in df.columns if c != "label"]
X = df[feature_cols].values
y = df["label"].values

# Save feature names
pickle.dump(feature_cols, open("models/feature_names.pkl","wb"))

# ----------------- Encode labels -----------------
le = LabelEncoder()
y_enc = le.fit_transform(y)
pickle.dump(le, open("models/label_encoder.pkl","wb"))

# ----------------- Scale data -----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pickle.dump(scaler, open("models/scaler.pkl","wb"))

# ----------------- Split data -----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# Save small sample for SHAP
pickle.dump(X_train[:300], open("models/X_train_sample.pkl","wb"))

# ----------------- Train model -----------------
model = RandomForestClassifier(n_estimators=250, random_state=42)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("models/crop_model.pkl","wb"))

print("Model training complete. Files saved in models/.")
