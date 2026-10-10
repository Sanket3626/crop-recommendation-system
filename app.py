import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import hashlib
import json


# SIMPLE LOGIN SYSTEM 
USERS_FILE = "users.json"     


def hash_password(password: str) -> str:
    """Convert plain password into SHA256 hash string."""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users() -> dict:
    """Load users from JSON file."""
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}


def save_users(users: dict) -> None:
    """Save users dict to JSON file."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def login_register_page():
    st.title("Login to continue")

    users = load_users()

    tab_login, tab_register = st.tabs(["Login", "New user / Register"])

    # ---------- LOGIN TAB ----------
    with tab_login:
        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")

        if st.button("Login", key="login_btn"):
            u = username.strip()
            p = password

            if u in users and (users[u] == hash_password(p) or users[u] == p):
                st.success(f"Welcome, {u}!")
                st.session_state.logged_in = True
                st.session_state.username = u
                # page ko refresh karo taki neeche wala main app load ho jaye
                st.rerun()
            else:
                st.error("Invalid username or password")

        #st.info("Demo credentials: admin / admin123, farmer / farmer123")

    # ---------- REGISTER TAB ----------
    with tab_register:
        new_user = st.text_input("👤 New username")
        new_pass = st.text_input("🔑 New password", type="password")
        confirm = st.text_input("✅ Confirm password", type="password")

        if st.button("Create account", key="register_btn"):
            nu = new_user.strip()

            if not nu or not new_pass or not confirm:
                st.warning("Please fill all fields.")
            elif nu in users:
                st.error("This username already exists. Try another.")
            elif new_pass != confirm:
                st.error("Passwords do not match.")
            else:
                users[nu] = new_pass
                save_users(users)
                st.success("Account created successfully! Now login from the Login tab.")
# ================== END LOGIN SYSTEM ==================

# --------------- SESSION INIT ---------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

# --------------- LOGIN GATE ---------------
if not st.session_state.logged_in:
    # sirf login/register page dikhega
    login_register_page()
    st.stop()        # yahi ruk jao, niche wala pura app code abhi run mat karo

# ================== END LOGIN SYSTEM ==================

# -------------- Yaha se tumhara model load wala code start karo --------------
# e.g.
# MODEL_DIR = "models"
# with open(os.path.join(MODEL_DIR, "crop_model.pkl"), "rb") as f:
#     model = pickle.load(f)

# ===================== MAIN APP (after login) =====================

st.sidebar.success(f"Logged in as: {st.session_state.username}")


# yahi se tumhara existing Home / Predict / Dataset / Insights ka code:
# if page == "Home":
#     ...
# elif page == "Predict":
#     ...
# etc...

# --------------------------------------------------------
# Load model + encoders + scaler + feature names + sample
# --------------------------------------------------------
@st.cache_resource
def load_all():
    model = pickle.load(open("models/crop_model.pkl","rb"))
    le = pickle.load(open("models/label_encoder.pkl","rb"))
    scaler = pickle.load(open("models/scaler.pkl","rb"))
    feature_names = pickle.load(open("models/feature_names.pkl","rb"))
    X_sample = pickle.load(open("models/X_train_sample.pkl","rb"))
    return model, le, scaler, feature_names, X_sample

model, le, scaler, feature_names, X_sample = load_all()

# SHAP Explainer
@st.cache_resource
def load_shap_explainer(_model, X_sample):
    # using leading underscore tells Streamlit not to hash `_model`
    import shap
    # TreeExplainer accepts model and background data (X_sample)
    explainer = shap.TreeExplainer(_model, data=X_sample)
    return explainer

# call it exactly the same way:
explainer = load_shap_explainer(model, X_sample)

# Load dataset for visualizations
df = pd.read_csv("Crop_recommendation.csv")

# --------------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------------
st.sidebar.title("🌱 Crop Recommender")
page = st.sidebar.radio("Menu", ["Home","Predict","Dataset","Insights"])

# --------------------------------------------------------
# HOME
# --------------------------------------------------------
if page == "Home":
    st.title("🌾 Smart Crop Recommendation System")
    st.write("""
    This system predicts the **best crop** for cultivation using  
    **soil nutrients + environmental factors** and provides  
    **explainability using SHAP**.
    """)

    st.image(
        "https://images.unsplash.com/photo-1501004318641-b39e6451bec6?q=80&w=1000",
        use_column_width=True
    )

# --------------------------------------------------------
# PREDICT
# --------------------------------------------------------
elif page == "Predict":
    st.header("🔍 Predict Best Crop")

    left, right = st.columns(2)
    inputs = {}

    for i, feat in enumerate(feature_names):
        col = left if i % 2 == 0 else right

        if feat.lower() == "ph":
            inputs[feat] = col.number_input(feat, 0.0, 14.0, 6.5)
        else:
            inputs[feat] = col.number_input(feat, 0.0, 300.0, 50.0)

    if st.button("Predict"):
        X = np.array([[inputs[f] for f in feature_names]])
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        crop = le.inverse_transform([pred])[0]

        st.success(f"🌱 **Recommended Crop: {crop}**")

        # Probability Table
        probs = model.predict_proba(X_scaled)[0]
        top_idx = probs.argsort()[-5:][::-1]

        st.subheader("Prediction Confidence")
        st.table(pd.DataFrame({
            "Crop": le.inverse_transform(top_idx),
            "Probability": [f"{probs[i]*100:.2f}%" for i in top_idx]
        }))

        # SHAP Local Explanation
        st.subheader("📌 Why This Crop Was Recommended?")
        # ---------- SHAP handling and safe DataFrame creation ----------
        # compute SHAP values for the input safely
        try:
            shap_vals = explainer.shap_values(X_scaled)
        except Exception as e:
            st.error(f"Could not compute SHAP values: {e}")
            shap_vals = None

        def _extract_local_shap(shap_vals, pred_class, n_features):
            # Return 1-D array length == n_features
            if shap_vals is None:
                return np.zeros(n_features)

            arr = None
            if isinstance(shap_vals, list):
            # multiclass: list of arrays
                try:
                    arr = np.array(shap_vals[pred_class])
                except Exception:
                    arr = np.array(shap_vals[0])
            else:
                arr = np.array(shap_vals)
            # Normalize to 1D vector
            if arr.ndim == 1:
                local = arr
            elif arr.ndim == 2:
            # (n_samples, n_features) -> take first row
                local = arr[0]
            elif arr.ndim == 3:
                # try common shapes
                if arr.shape[1] == 1:
                    local = arr[0, 0, :]
                elif arr.shape[0] == 1 and arr.shape[1] >= 1:
                    local = arr[0, 0, :]
                else:
                    local = arr.ravel()[:n_features]
            else:
                local = arr.ravel()[:n_features]

            # Ensure 1-D and correct length
            local = np.asarray(local).reshape(-1)
            if local.size < n_features:
            # pad with zeros
                pad = np.zeros(n_features - local.size)
                local = np.concatenate([local, pad])
            elif local.size > n_features:
                local = local[:n_features]
            return local

        # extract safe local_shap
        n_feats = len(feature_names)
        pred_class = pred  # integer predicted class
        local_shap = _extract_local_shap(shap_vals, pred_class, n_feats)

        # ensure input vector X is 1-D and same length
        X_vals = np.asarray(X).reshape(-1)[:n_feats]
        if X_vals.size < n_feats:
            pad = np.zeros(n_feats - X_vals.size)
            X_vals = np.concatenate([X_vals, pad])

        # finally build DataFrame safely
        shap_df = pd.DataFrame({
            "Feature": list(feature_names),
            "Value": X_vals,
            "Contribution": local_shap
        })

        # sort and display
        shap_df = shap_df.sort_values("Contribution", key=lambda x: x.abs(), ascending=False).reset_index(drop=True)

        st.write("### Top Contributing Features")
        st.table(shap_df.head(6))

        # barplot for top contributors
        top_plot = shap_df.head(8)
        fig, ax = plt.subplots(figsize=(7, min(0.4*len(top_plot) + 1.5, 6)))
        sns.barplot(x="Contribution", y="Feature", data=top_plot, ax=ax)
        ax.set_title("Local feature contributions (SHAP)")
        plt.tight_layout()
        st.pyplot(fig)

        # Natural Language Reason
        reason = ", ".join(
            [f"{row['Feature']}={row['Value']}" for _,row in shap_df.head(3).iterrows()]
        )
        st.info(f"Model chose **{crop}** mainly because: {reason}")

        # Sensitivity Plot for top feature
        st.subheader("🔧 Sensitivity of Most Important Feature")
        top_feature = shap_df.iloc[0]["Feature"]
        idx = feature_names.index(top_feature)

        vals = np.linspace(X[0][idx]*0.5, X[0][idx]*1.5, 25)
        prob_line = []

        for v in vals:
            temp = X.copy()
            temp[0][idx] = v
            prob_line.append(model.predict_proba(scaler.transform(temp))[0][pred_class])

        fig2, ax2 = plt.subplots(figsize=(6,3))
        ax2.plot(vals, prob_line)
        ax2.set_xlabel(top_feature)
        ax2.set_ylabel("Probability of predicted crop")
        st.pyplot(fig2)

# --------------------------------------------------------
# DATASET
# --------------------------------------------------------
elif page == "Dataset":
    st.header("📋 Dataset Overview")
    st.dataframe(df.head(20))
    st.write("Shape:", df.shape)

    st.subheader("Crop Frequency")
    st.bar_chart(df["label"].value_counts())

# --------------------------------------------------------
# INSIGHTS
# --------------------------------------------------------

elif page == "Insights":
    st.header("📊 Model Insights")

    # Feature Importance
    fi = pd.Series(model.feature_importances_, index=feature_names)
    fig, ax = plt.subplots(figsize=(7,4))
    sns.barplot(x=fi.values, y=fi.index, ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()

    fig2, ax2 = plt.subplots(figsize=(7,5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax2)
    st.pyplot(fig2)
    
