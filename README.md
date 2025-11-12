# Customer Churn Pro

An end-to-end **Customer Churn Prediction System** that predicts whether a customer is likely to leave a service and provides a **beautiful, emoji-driven Streamlit UI** to display results.  
This project demonstrates **data science, machine learning, and deployment** skills together in a professional, production-style structure.

---

## 🚀 Overview
This project analyzes customer behavior and subscription data to predict churn probability.  
The model is trained using **XGBoost** within a Scikit-Learn pipeline and deployed via a **Streamlit app** that shows:

---

## 📁 Repository Structure
```plaintext
customer-churn-prediction/
├── app/
│   └── streamlit_app.py           
│
├── data/
│   ├── data_raw/                  
│   ├── data_processed/            
│   └── README.md
│
├── models/
│   └── churn_xgb.joblib           
│
├── reports/                       
│
├── src/
│ └── churn/
│ ├── init.py
│ └── train.py
├── .env
├── .env.example
├── requirements.txt
├── .gitignore
└── README.md


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

---

## 🧱 System Architecture
### 🔹 Data Pipeline
1. **Load & Clean Data**
2. **Preprocessing** –  
   - Numerical. 
   - Categorical: one-hot encoded  
3. **Model Training** – XGBoost classifier with AUC/F1 evaluation  
4. **Model Persistence**

### 🔹 Serving / Prediction Pipeline
1. **User Input** through Streamlit form  
2. **Model Inference** using trained pipeline  
3. **UI Output** showing probability & feedback  

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
