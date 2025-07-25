
Last updated: July 20, 2025

# 📊 Credit Default Predictor


An interactive machine learning dashboard to predict credit default risk using the UCI dataset. Built with Python, Streamlit, and scikit-learn, this project allows users to explore model performance, adjust classification thresholds, and export predictions easily.

---

## 🚀 Features

- Cleaned and engineered financial dataset (UCI Credit Default)
- Multiple ML models:
  - Random Forest (original and balanced)
  - Logistic Regression
  - Voting Classifier
  - Gradient Boosting
  - Stacking Classifier (with SMOTE)
- Performance metrics: confusion matrix, classification report, ROC curves
- Export predictions with probabilities
- Interactive Streamlit dashboard

---

## 📁 Project Structure

```

credit-default-predictor/
├── data/
│   └── cleaned\_data.csv            # Cleaned dataset (excluded from repo; generated by clean\_data.py)
├── models/
│   └── stacking\_classifier.joblib  # Trained model (excluded from repo; generated by train\_model.py)
├── plots/
│   └── \*.png                       # Feature importance, ROC curves, etc.
├── clean\_data.py                   # Script to clean and save dataset
├── train\_model.py                  # Model training and export script
├── streamlit\_app.py                # Streamlit dashboard interface
├── main.py                         # Combined script for exploration and modeling
└── README.md                       # This file

````

---

## 🧪 How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/haresh253/credit-default-predictor.git
cd credit-default-predictor
````

### 2. Install Dependencies

If you have a `requirements.txt` file:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn streamlit joblib
```

### 3. Clean the Data (if needed)

```bash
python clean_data.py
```

This will generate `data/cleaned_data.csv`.

### 4. Train Models

```bash
python train_model.py
```

This saves the stacking model to `models/stacking_classifier.joblib`.

### 5. Launch the Dashboard

```bash
streamlit run streamlit_app.py
```

You can now view model performance, adjust thresholds, and export predictions interactively.

---

## 📉 Dataset

**UCI Default of Credit Card Clients Dataset**

* 30,000 samples
* Features: demographic info, credit limits, past payment history
* Binary target: Default payment next month (1 = Yes, 0 = No)

📎 [Link to Dataset](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)

---

## 📈 Model Performance

The stacking classifier combines:

* Base models: Logistic Regression, Random Forest, SVC
* Meta-learner: Gradient Boosting Classifier
* Class imbalance handled via SMOTE

Evaluation includes:

* Confusion Matrix
* Classification Report
* ROC AUC Score
* Threshold tuning for decision boundaries

📊 **Sample AUC Score:** > 0.75 (replace with actual when available)

---

## 🔮 Use Cases

This dashboard is ideal for:

* Financial risk analysis and credit scoring
* Exploring model trade-offs interactively
* Visualizing feature importance
* Benchmarking ML models on imbalanced data

---

## 🛠️ Built With

* Python 3
* pandas, NumPy
* scikit-learn, imbalanced-learn
* matplotlib, seaborn
* Streamlit
* joblib

---

## 🤝 Contributing

Pull requests and suggestions are welcome. For major changes, please open an issue first to discuss the idea.

---

## 📜 License

MIT License — free to use, modify, and distribute.

````

---

### ✅ What You Should Do Now

1. **Copy and paste** the above into your `README.md` file in PyCharm.
2. Save and stage it:

```bash
git add README.md
git commit -m "Update README with full project details"
git push
````
