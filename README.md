# Customer Churn Pro

An end-to-end **Customer Churn Prediction System** that predicts whether a customer is likely to leave a service and provides a **beautiful, emoji-driven Streamlit UI** to display results.  
The project demonstrates **data science, machine learning, and deployment** skills together in one production-style repo.

---

## 🚀 Overview
This project analyzes customer behavior and subscription data to predict churn probability.  
The model is trained using **XGBoost** within a Scikit-Learn pipeline and served via a **Streamlit app** that shows:
- ✅ **“Eligible / Low Risk”** screen with happy emojis (😊 👍 🕺)  
- ❌ **“At Risk”** screen with sad emojis (😢 👎 🙍‍♀️)

---

## 🗂️ Repository Structure
customer-churn-prediction/
├── app/
│ └── streamlit_app.py # Streamlit UI for predictions
├── data/
│ ├── data_raw/ # Raw CSV or Excel files (gitignored)
│ ├── data_processed/ # Cleaned or feature-engineered data (optional)
│ └── README.md
├── models/
│ └── churn_xgb.joblib # Saved model artifact (auto-generated)
├── reports/ # Visuals or monitoring outputs
├── src/
│ └── churn/
│ ├── init.py
│ └── train.py # Training pipeline
├── .env # Local config (gitignored)
├── .env.example # Template for environment variables
├── requirements.txt # Python dependencies
├── .gitignore # Ignored files/folders
└── README.md # Project documentation


---

## 🎯 Project Objectives
1. **Develop** a robust churn prediction model on real-world-like customer data.  
2. **Automate** preprocessing and model training using clean, modular Python code.  
3. **Deploy** the model through an interactive UI for easy business use.  
4. **Visualize** the results with engaging emoji-based feedback.

---

## 📊 Dataset Description
**Source:** Kaggle – *Customer Churn Dataset*  
Each record represents a customer with demographic, usage, and subscription details.

| Column | Type | Description |
|---------|------|-------------|
| `CustomerID` | String | Unique identifier (optional) |
| `Age` | Numeric | Customer age |
| `Gender` | Categorical | Male / Female / Other |
| `Tenure` | Numeric | Months with company |
| `Usage` | Numeric | Usage score or frequency |
| `Support` | Numeric | Number of support calls |
| `PaymentDelay` | Numeric | Days payment delayed |
| `Subscription` | Categorical | Basic / Standard / Premium |
| `Contract` | Categorical | Monthly / Quarterly / Annual |
| `TotalSpend` | Numeric | Cumulative amount spent |
| `LastInteraction` | Numeric | Days since last contact |
| `Churn` | Binary | 1 = churned, 0 = active (target) |

> Place your data files inside:
> ```
> data/data_raw/
> ├── customer_churn_dataset-training-master.csv
> └── customer_churn_dataset-testing-master.csv
> ```

---

## 🧱 System Architecture
### 🔹 Data Pipeline
1. **Load & Clean Data** – Auto-maps messy headers (e.g., “Total Spen” → “TotalSpend”)  
2. **Preprocessing** –  
   - Numerical: passed through as-is  
   - Categorical: one-hot encoded  
3. **Model Training** – XGBoost classifier with AUC/F1 evaluation  
4. **Model Persistence** – Saves trained pipeline to `models/churn_xgb.joblib`

### 🔹 Serving / Prediction Pipeline
1. **User Input** through Streamlit form  
2. **Model Inference** using trained pipeline  
3. **UI Output** showing probability & emoji feedback  

  ┌───────────────┐
  │  Raw Dataset  │
  └──────┬────────┘
         ▼
         ▼
┌────────────────────┐
│ Training Script │
│ (train.py) │
└──────┬─────────────┘
▼
models/churn_xgb.joblib
│
▼
┌────────────────────┐
│ Streamlit App │
│ (streamlit_app.py) │
└────────────────────┘
