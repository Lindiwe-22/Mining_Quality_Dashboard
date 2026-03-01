# ⛏️ Mining Quality Intelligence — Predictive Quality Control System

> A full-stack machine learning system that predicts silica concentrate quality failures in an iron ore flotation plant — hours before they happen.

---

## 🔗 Demo Link

👉 [View Live Streamlit Dashboard](https://miningqualitydashboard-lindiwesongewa.streamlit.app/)

---

## 📋 Table of Contents

- [Business Understanding](#business-understanding)
- [Screenshots](#screenshots)
- [Technologies](#technologies)
- [Setup](#setup)
- [Approach](#approach)
- [Status](#status)
- [Credits](#credits)

---

## 💼 Business Understanding

In iron ore flotation processing, silica is an unwanted impurity. When silica concentration exceeds 4% in the final concentrate, product quality fails industry standards, customers impose penalty fees or reject shipments, and revenue is lost while reprocessing costs increase.

By the time quality degradation is detected through lab analysis, it is already too late to intervene. This project solves that problem.

**A binary classification model trained on 737,453 hourly sensor readings predicts quality failures before they occur** — giving operators time to adjust process parameters and prevent off-spec production.

Operators receive a three-tier alert system:

| Alert | Meaning |
|---|---|
| 🟢 **GREEN** | Normal operation — no action needed |
| 🟡 **AMBER** | Early warning — monitor closely |
| 🔴 **RED** | Intervention required + specific recommended actions |

---

## 📸 Screenshots

<img width="1366" height="728" alt="Mining Quality" src="https://github.com/user-attachments/assets/65d8ec27-8820-4472-b933-7bed3009fd72" />
<img width="1366" height="728" alt="Mining Quality Prediction" src="https://github.com/user-attachments/assets/ae7ee70a-169a-4f60-9ead-5642837e44ed" />

---

## 🛠️ Technologies

**Languages & Environment**

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

**Data & EDA**

![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logo=python&logoColor=white)

**Machine Learning**

![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Imbalanced-learn](https://img.shields.io/badge/Imbalanced--learn%20%7C%20SMOTE-4B8BBE?style=for-the-badge&logo=python&logoColor=white)

**Deployment**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Joblib](https://img.shields.io/badge/Joblib-3776AB?style=for-the-badge&logo=python&logoColor=white)

| Category | Tools |
|---|---|
| **Data & EDA** | Python, Pandas, NumPy, Matplotlib, Seaborn |
| **Machine Learning** | XGBoost, TensorFlow/Keras, Scikit-learn |
| **Class Imbalance** | Imbalanced-learn (SMOTE) |
| **Deployment** | Streamlit Cloud, Plotly, Joblib |

---

## ⚙️ Setup

```bash
# Clone the repository
git clone https://github.com/your-username/mining-quality-intelligence.git
cd mining-quality-intelligence

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard locally
streamlit run app.py
```

> **Dataset:** [Quality Prediction in a Mining Process](https://www.kaggle.com/datasets/edumagalhaes/quality-prediction-in-a-mining-process) — available on Kaggle.

---

## 🔍 Approach

### Phase 1 — Data Loading
The dataset comprises 737,453 rows × 24 sensor columns sourced from a real iron ore flotation plant, with zero missing values confirmed across all features.

### Phase 2 — Exploratory Data Analysis
Twelve new derived features were engineered, followed by univariate, correlation, and time series analysis. Quality thresholds were defined as: **Premium < 2%**, **Good < 3%**, **Acceptable < 4%**, **Poor ≥ 4%**. Analysis revealed weekly operational cycles and shift-change patterns in the data.

### Phase 3 — Machine Learning
Data was preprocessed using StandardScaler with SMOTE to address class imbalance, on an 80/20 stratified split. Two models were trained and compared:

- **Model 1 — XGBoost:** 300 estimators, max_depth=6, learning rate=0.05
- **Model 2 — Neural Network:** Architecture 128→64→32→1 with BatchNorm and Dropout layers

The primary evaluation metric was **F1-Score**, with threshold tuning applied to maximise net financial benefit for the business.

### Phase 4 — Deployment & Monitoring
A real-time `MiningQualityScorer` pipeline class was built and deployed via a Streamlit dashboard with four interactive pages:

| Page | Description |
|---|---|
| 🏠 **Live Scoring** | Enter sensor readings → instant alert + recommended actions |
| 📈 **Historical Trends** | 168-hour probability trend + alert distribution |
| 🔍 **Feature Inspector** | Feature vs failure probability analysis |
| 📊 **Drift Monitor** | PSI heatmap — flags when model needs retraining |

PSI drift monitoring with automated retraining triggers ensures the model remains accurate as plant conditions evolve over time.

---

## 📌 Status

![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

The full pipeline is complete and deployed. Future iterations may include:
- Multi-plant generalisation across different flotation configurations
- Integration with SCADA/DCS systems for direct operator alerts
- Expanded drift monitoring with automated model versioning

---

## 🙏 Credits

**Developed by Lindiwe Songelwa — Data Scientist | Developer | Insight Creator**

| Platform | Link |
|---|---|
| 💼 LinkedIn | [Lindiwe S.](https://www.linkedin.com/in/lindiwe-songelwa) |
| 🌐 Portfolio | [Creative Portfolio](https://lindiwe-22.github.io/Portfolio-Website/) |
| 🏅 Credly | [Lindiwe Songelwa – Badges](https://www.credly.com/users/samnkelisiwe-lindiwe-songelwa) |
| 🚀 Live App | [Streamlit Dashboard](https://miningqualitydashboard-lindiwesongewa.streamlit.app/) |
| 📧 Email | [sl.songelwa@hotmail.co.za](mailto:sl.songelwa@hotmail.co.za) |

---

*© 2026 Lindiwe Songelwa. All rights reserved.*
