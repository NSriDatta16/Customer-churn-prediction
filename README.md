# 🧠 Customer Churn Pro

An end-to-end **Customer Churn Prediction System** that predicts whether a customer is likely to leave a service and provides a **beautiful, emoji-driven Streamlit UI** to display results.  
This project demonstrates **data science, machine learning, and deployment** skills together in a professional, production-style structure.

---

## 🚀 Overview
This project analyzes customer behavior and subscription data to predict churn probability.  
The model is trained using **XGBoost** within a Scikit-Learn pipeline and deployed via a **Streamlit app** that shows:

- ✅ **Eligible / Low Risk** screen with happy emojis (😊 👍 🕺)  
- ❌ **At Risk** screen with sad emojis (😢 👎 🙍‍♀️)

---

## 📁 Repository Structure
```plaintext
customer-churn-prediction/
├── app/
│   └── streamlit_app.py           # Streamlit UI for predictions
│
├── data/
│   ├── data_raw/                  # Raw CSV or Excel files (gitignored)
│   ├── data_processed/            # Cleaned or feature-engineered data (optional)
│   └── README.md
│
├── models/
│   └── churn_xgb.joblib           # Saved model artifact (auto-generated)
│
├── reports/                       # Visuals or monitoring outputs
│
├── src/
│   └── churn/
│       ├── __init__.py
│       └── train.py               # Training pipeline
│
├── .env                           # Local config (gitignored)
├── .env.example                   # Template for environment variables
├── requirements.txt               # Python dependencies
├── .gitignore                     # Ignored files/folders
└── README.md                      # Project documentation
