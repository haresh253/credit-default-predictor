import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Set page config
st.set_page_config(page_title="Credit Default Predictor", layout="wide")

# --- Title ---
st.title("📊 Credit Default Risk Dashboard")

# --- Load Data ---
@st.cache_data
def load_data():
    return pd.read_csv("data/cleaned_data.csv")  # Update this to match your path

data = load_data()
st.subheader("Dataset Preview")
st.dataframe(data.head(10))

# --- Upload Model (Optional) ---
st.sidebar.header("🔍 Load a Trained Model")
model = None
model_option = st.sidebar.radio("Use existing model or upload one:", ["Use built-in", "Upload .joblib"])

if model_option == "Use built-in":
    model = joblib.load("models/stacking_classifier.joblib")  # path to saved model
else:
    uploaded_model = st.sidebar.file_uploader("Upload a .joblib model", type=["joblib"])
    if uploaded_model is not None:
        model = joblib.load(uploaded_model)

# --- Predictions ---
if model is not None:
    X = data.drop("default", axis=1)
    y = data["default"]
    y_proba = model.predict_proba(X)[:, 1]

    # Threshold selection
    st.sidebar.header("⚙️ Classification Threshold")
    threshold = st.sidebar.slider("Choose a threshold", 0.0, 1.0, 0.5, 0.01)
    y_pred = (y_proba >= threshold).astype(int)

    # --- Metrics ---
    st.subheader("📈 Model Evaluation")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Classification Report**")
        report = classification_report(y, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    with col2:
        st.write("**Confusion Matrix**")
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    # --- ROC Curve ---
    fpr, tpr, _ = roc_curve(y, y_proba)
    roc_auc = auc(fpr, tpr)

    st.subheader(f"📉 ROC Curve (AUC = {roc_auc:.4f})")
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax2.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.set_title("Receiver Operating Characteristic")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

    # --- Export Predictions ---
    st.subheader("📤 Export Predictions")
    export_df = data.copy()
    export_df["Predicted Default"] = y_pred
    export_df["Probability"] = y_proba

    csv = export_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download as CSV", data=csv, file_name="predictions.csv", mime="text/csv")

else:
    st.warning("⚠️ Please load a model to proceed.")
