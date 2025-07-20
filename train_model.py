import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# -----------------------
# 📥 Load Cleaned Data
# -----------------------
data_path = Path("data/cleaned_data.csv")
if not data_path.exists():
    raise FileNotFoundError("❌ Cleaned data not found. Please run clean_data.py first.")

df = pd.read_csv(data_path)
X = df.drop("default", axis=1)
y = df["default"]

# -----------------------
# 🧪 Train/Test Split & Scaling
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------
# 🔁 SMOTE Oversampling
# -----------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

# -----------------------
# 🤖 Stacking Classifier
# -----------------------
base_learners = [
    ("lr", LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    ("rf", RandomForestClassifier(class_weight='balanced', random_state=42)),
    ("svc", SVC(probability=True, class_weight='balanced', random_state=42))
]
meta_model = GradientBoostingClassifier(random_state=42)

stack_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1,
    passthrough=False
)

stack_model.fit(X_train_res, y_train_res)

# -----------------------
# ✅ Evaluation
# -----------------------
y_probs = stack_model.predict_proba(X_test_scaled)[:, 1]
y_preds = (y_probs >= 0.4).astype(int)

print("\n[Stacking Classifier] Classification Report (Threshold = 0.4):")
print(classification_report(y_test, y_preds))

print("[Stacking Classifier] Confusion Matrix:")
print(confusion_matrix(y_test, y_preds))

print(f"[Stacking Classifier] AUC: {roc_auc_score(y_test, y_probs):.4f}")

# -----------------------
# 💾 Save Model
# -----------------------
output_path = Path("models/stacking_classifier.joblib")
output_path.parent.mkdir(exist_ok=True, parents=True)
joblib.dump(stack_model, output_path)
print(f"\n✅ Model saved to {output_path}")
