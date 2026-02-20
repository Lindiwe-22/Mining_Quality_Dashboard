⛏️ Mining Quality Intelligence — Predictive Quality Control System
A full-stack machine learning system that predicts silica concentrate quality failures in an iron ore flotation plant — hours before they happen.

🎯 Problem Statement
In iron ore flotation processing, silica is an unwanted impurity. When silica concentration exceeds 4% in the final concentrate:

Product quality fails industry standards
Customers impose penalty fees or reject shipments
Revenue is lost and reprocessing costs increase

By the time quality degradation is detected through lab analysis, it is already too late to intervene. This project solves that problem.

💡 Solution
A binary classification model trained on 737,453 hourly sensor readings that predicts quality failures before they occur, giving operators time to adjust process parameters and prevent off-spec production.

Operators receive:

🟢 GREEN — Normal operation, no action needed
🟡 AMBER — Early warning, monitor closely
🔴 RED — Intervention required + specific recommended actions


🗂️ Project Structure

mining-quality-dashboard/
├── app.py                    # Streamlit dashboard (4 pages)
├── requirements.txt          # Python dependencies
├── model/
│   ├── xgb_model.pkl         # Trained XGBoost champion model
│   ├── scaler.pkl            # Fitted StandardScaler
│   └── config.json           # Deployment threshold + metadata
└── data/
    └── scored_history.csv    # 720 hours of scored production data

📓 Notebook Phases

Phase 1 — Data Loading

Dataset: Quality Prediction in a Mining Process (Kaggle)
737,453 rows × 24 sensor columns
Zero missing values confirmed

Phase 2 — Exploratory Data Analysis

Feature engineering: 12 new derived features
Univariate, correlation, and time series analysis
Quality thresholds defined (Premium <2%, Good <3%, Acceptable <4%, Poor ≥4%)
Weekly operational cycles and shift-change patterns revealed

Phase 3 — Machine Learning

Preprocessing: StandardScaler + SMOTE (80/20 stratified split)
Model 1: XGBoost (300 estimators, max_depth=6, lr=0.05)
Model 2: Neural Network (128→64→32→1, BatchNorm, Dropout)
Primary Metric: F1-Score
Threshold Tuning: Business-optimal (maximises net financial benefit)

Phase 4 — Deployment & Monitoring

Real-time MiningQualityScorer pipeline class
Streamlit dashboard with 4 interactive pages
PSI drift monitoring with automated retraining triggers


📊 Dashboard Pages
🏠 Live Scoring: Enter sensor readings → instant alert + actions
📈 Historical Trends: 168-hour probability trend + alert distribution
🔍 Feature Inspector: Feature vs failure probability analysis
📊 Drift Monitor: PSI heatmap — flags when model needs retraining

🛠️ Tech Stack
CategoryToolsData & EDA: Python, Pandas, NumPy, Matplotlib, Seaborn
Machine Learning: XGBoost, TensorFlow/Keras, Scikit-learn
Class Imbalance: Imbalanced-learn (SMOTE)
Deployment: Streamlit Cloud, Plotly, Joblib

👩🏾‍💻 Author
Lindiwe Songelwa — Data Scientist | Developer | Insight Creator

🌐 Portfolio https://lindiwe-22.github.io/Portfolio-Website/
💼 LinkedIn https://www.linkedin.com/in/lindiwe-songelwa
🏅 [Credly] https://www.credly.com/users/samnkelisiwe-lindiwe-songelwa
[![Streamlit App](https://miningqualitydashboard-lindiwesongewa.streamlit.app/)]
📧 sl.songelwa@hotmail.co.za


© 2026 Lindiwe Songelwa. All rights reserved.
