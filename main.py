import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

# ------------------------------
# 📌 Setup & Load Data
# ------------------------------
sns.set(style="whitegrid")
df = pd.read_excel("default of credit card clients.xls", header=1)
df.rename(columns={"default payment next month": "default"}, inplace=True)
df.drop("ID", axis=1, inplace=True)

print("Dataset shape after cleanup:", df.shape)
print("\nColumns:\n", df.columns)

# ------------------------------
# 📊 Default Rate
# ------------------------------
default_counts = df["default"].value_counts(normalize=True)
print("\nDefault rate:")
print(default_counts)

sns.countplot(x="default", data=df)
plt.title("Default Rate (0 = No Default, 1 = Default)")
plt.xlabel("Default")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("plots/default_rate.png")
plt.close()

# ------------------------------
# 📊 Default by Gender
# ------------------------------
plt.figure(figsize=(6, 4))
sns.countplot(x="SEX", hue="default", data=df)
plt.title("Default by Gender (1 = Male, 2 = Female)")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.legend(title="Default")
plt.tight_layout()
plt.savefig("plots/default_by_sex.png")
plt.close()

# ------------------------------
# 📊 Credit Limit vs Default
# ------------------------------
plt.figure(figsize=(8, 4))
sns.boxplot(x="default", y="LIMIT_BAL", data=df)
plt.title("Credit Limit by Default Status")
plt.xlabel("Default (0 = No, 1 = Yes)")
plt.ylabel("Credit Limit")
plt.tight_layout()
plt.savefig("plots/limit_by_default.png")
plt.close()

# ------------------------------
# 📊 Correlation Heatmap
# ------------------------------
plt.figure(figsize=(14, 10))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\nCorrelation with default:")
print(corr["default"].sort_values(ascending=False))

# ------------------------------
# 🧠 Feature Engineering
# ------------------------------
df["AVG_BILL_AMT"] = df[[f"BILL_AMT{i}" for i in range(1, 7)]].mean(axis=1)
df["AVG_PAY_AMT"] = df[[f"PAY_AMT{i}" for i in range(1, 7)]].mean(axis=1)
df["TOTAL_BILL_AMT"] = df[[f"BILL_AMT{i}" for i in range(1, 7)]].sum(axis=1)
df["TOTAL_PAY_AMT"] = df[[f"PAY_AMT{i}" for i in range(1, 7)]].sum(axis=1)
df["PAYMENT_RATIO"] = df["TOTAL_PAY_AMT"] / (df["TOTAL_BILL_AMT"] + 1)
df["NUM_LATE_PAYMENTS"] = df[[f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]].gt(0).sum(axis=1)
df["MAX_LATE_STATUS"] = df[[f"PAY_{i}" for i in [0, 2, 3, 4, 5, 6]]].max(axis=1)

# ------------------------------
# 🎯 Define Features and Target
# ------------------------------
X = df.drop("default", axis=1)
y = df["default"]

# 🔄 Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ------------------------------
# 🌲 Model 1: Original Random Forest
# ------------------------------
rf_original = RandomForestClassifier(random_state=42)
rf_original.fit(X_train, y_train)
y_pred_original = rf_original.predict(X_test)

print("\n[Original RF] Classification Report:")
print(classification_report(y_test, y_pred_original))
print("[Original RF] Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_original))

importances = rf_original.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices], palette="Blues")
plt.title("Feature Importance - Original RF")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("plots/feature_importance_original.png")
plt.show()

# ------------------------------
# 🌲 Model 2: Balanced Random Forest
# ------------------------------
rf_balanced = RandomForestClassifier(class_weight="balanced", random_state=42)
rf_balanced.fit(X_train, y_train)
y_pred_balanced = rf_balanced.predict(X_test)

print("\n[Balanced RF] Classification Report:")
print(classification_report(y_test, y_pred_balanced))
print("[Balanced RF] Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_balanced))

importances_bal = rf_balanced.feature_importances_
indices_bal = np.argsort(importances_bal)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances_bal[indices_bal], y=features[indices_bal], palette="Greens")
plt.title("Feature Importance - Balanced RF")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("plots/feature_importance_balanced.png")
plt.show()

# ------------------------------
# ➕ Model 3: Logistic Regression (with Scaling)
# ------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

log_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_model.fit(X_train_scaled, y_train_scaled)
y_pred_log = log_model.predict(X_test_scaled)

print("\n[Logistic Regression] Classification Report:")
print(classification_report(y_test_scaled, y_pred_log))
print("[Logistic Regression] Confusion Matrix:")
print(confusion_matrix(y_test_scaled, y_pred_log))

# ------------------------------
# 📈 ROC Curve Comparison
# ------------------------------
rf_probs = rf_original.predict_proba(X_test)[:, 1]
rf_bal_probs = rf_balanced.predict_proba(X_test)[:, 1]
log_probs = log_model.predict_proba(X_test_scaled)[:, 1]

rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
bal_fpr, bal_tpr, _ = roc_curve(y_test, rf_bal_probs)
log_fpr, log_tpr, _ = roc_curve(y_test_scaled, log_probs)

rf_auc = roc_auc_score(y_test, rf_probs)
bal_auc = roc_auc_score(y_test, rf_bal_probs)
log_auc = roc_auc_score(y_test_scaled, log_probs)

plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.2f})")
plt.plot(bal_fpr, bal_tpr, label=f"Balanced RF (AUC = {bal_auc:.2f})")
plt.plot(log_fpr, log_tpr, label=f"Logistic Regression (AUC = {log_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_comparison.png")
plt.show()

from sklearn.ensemble import VotingClassifier

# ------------------------------
# 🗳️ Model 4: Voting Classifier (Soft Voting)
# ------------------------------
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_original),
        ('rf_bal', rf_balanced),
        ('logreg', log_model)
    ],
    voting='soft'  # Use probability-based voting
)

voting_clf.fit(X_train_scaled, y_train_scaled)
y_pred_voting = voting_clf.predict(X_test_scaled)

print("\n[Voting Classifier] Classification Report:")
print(classification_report(y_test_scaled, y_pred_voting))

print("[Voting Classifier] Confusion Matrix:")
print(confusion_matrix(y_test_scaled, y_pred_voting))

# ROC Curve for VotingClassifier
voting_probs = voting_clf.predict_proba(X_test_scaled)[:, 1]
voting_fpr, voting_tpr, _ = roc_curve(y_test_scaled, voting_probs)
voting_auc = roc_auc_score(y_test_scaled, voting_probs)

# Add to ROC curve plot
plt.figure(figsize=(8, 6))
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.2f})")
plt.plot(bal_fpr, bal_tpr, label=f"Balanced RF (AUC = {bal_auc:.2f})")
plt.plot(log_fpr, log_tpr, label=f"Logistic Regression (AUC = {log_auc:.2f})")
plt.plot(voting_fpr, voting_tpr, label=f"VotingClassifier (AUC = {voting_auc:.2f})", linestyle="--", color="black")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_comparison_voting.png")
plt.show()

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

# ------------------------------
# 🌟 Model 4: Gradient Boosting
# ------------------------------
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)
gb_probs = gb_model.predict_proba(X_test)[:, 1]

print("\n[Gradient Boosting] Classification Report:")
print(classification_report(y_test, y_pred_gb))
print("[Gradient Boosting] Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gb))

# ------------------------------
# 🌟 Model 5: SVC (with probability)
# ------------------------------
svc_model = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
svc_model.fit(X_train_scaled, y_train_scaled)
y_pred_svc = svc_model.predict(X_test_scaled)
svc_probs = svc_model.predict_proba(X_test_scaled)[:, 1]

print("\n[Support Vector Classifier] Classification Report:")
print(classification_report(y_test_scaled, y_pred_svc))
print("[Support Vector Classifier] Confusion Matrix:")
print(confusion_matrix(y_test_scaled, y_pred_svc))

# ------------------------------
# 📈 Update ROC Curve Plot
# ------------------------------
gb_fpr, gb_tpr, _ = roc_curve(y_test, gb_probs)
svc_fpr, svc_tpr, _ = roc_curve(y_test_scaled, svc_probs)
gb_auc = roc_auc_score(y_test, gb_probs)
svc_auc = roc_auc_score(y_test_scaled, svc_probs)

plt.figure(figsize=(10, 6))
plt.plot(rf_fpr, rf_tpr, label=f"Random Forest (AUC = {rf_auc:.2f})")
plt.plot(bal_fpr, bal_tpr, label=f"Balanced RF (AUC = {bal_auc:.2f})")
plt.plot(log_fpr, log_tpr, label=f"Logistic Regression (AUC = {log_auc:.2f})")
plt.plot(gb_fpr, gb_tpr, label=f"Gradient Boosting (AUC = {gb_auc:.2f})")
plt.plot(svc_fpr, svc_tpr, label=f"SVC (AUC = {svc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_comparison_all_models.png")
plt.show()

from sklearn.ensemble import StackingClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# -------------------------------------------
# 🧪 Use SMOTE on scaled training data
# -------------------------------------------
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train_scaled)

# -------------------------------------------
# 🔧 Base models
# -------------------------------------------
base_learners = [
    ("lr", LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    ("rf", RandomForestClassifier(class_weight='balanced', random_state=42)),
    ("svc", SVC(probability=True, class_weight='balanced', random_state=42))
]

# -------------------------------------------
# 🎯 Meta-model: Gradient Boosting
# -------------------------------------------
meta_model = GradientBoostingClassifier(random_state=42)

# -------------------------------------------
# 🤖 Stacking Classifier
# -------------------------------------------
stack_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1,
    passthrough=False
)

stack_model.fit(X_train_res, y_train_res)

# -------------------------------------------
# 📊 Prediction and Threshold Tuning
# -------------------------------------------
y_probs_stack = stack_model.predict_proba(X_test_scaled)[:, 1]

# Tune threshold (default = 0.5)
threshold = 0.4
y_preds_stack = (y_probs_stack >= threshold).astype(int)

# -------------------------------------------
# 📈 Evaluation
# -------------------------------------------
print("\n[Stacking Classifier] Classification Report (Threshold = 0.4):")
print(classification_report(y_test_scaled, y_preds_stack))

print("[Stacking Classifier] Confusion Matrix:")
print(confusion_matrix(y_test_scaled, y_preds_stack))

stack_auc = roc_auc_score(y_test_scaled, y_probs_stack)
print(f"[Stacking Classifier] AUC: {stack_auc:.4f}")

# ROC Curve
fpr_stack, tpr_stack, _ = roc_curve(y_test_scaled, y_probs_stack)
plt.figure(figsize=(8, 6))
plt.plot(fpr_stack, tpr_stack, label=f"Stacking (AUC = {stack_auc:.2f})", color="purple")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Stacking Classifier")
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_stacking.png")
plt.show()
