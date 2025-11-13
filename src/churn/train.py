from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score
from xgboost import XGBClassifier

# Prefer your Kaggle CSV; fall back to the small sample
RAW_TRAIN = Path("data/data_raw/customer_churn_dataset-training-master.csv")
FALLBACK_SAMPLE = Path("data/data_raw/sample_customers.csv")
MODEL_PATH = Path("models/churn_xgb.joblib")

# Header canonicalization (robust to prefixes/typos shown in your screenshots)
HEADER_CANON = {
    "customer": "CustomerID",
    "customerid": "CustomerID",
    "age": "Age",
    "gender": "Gender",
    "tenure": "Tenure",
    "usage": "Usage",
    "support": "Support",
    "payment": "PaymentDelay",
    "paymentdelay": "PaymentDelay",
    "subscription": "Subscription",
    "contract": "Contract",
    "total": "TotalSpend",
    "totalspend": "TotalSpend",
    "last": "LastInteraction",
    "lastinteraction": "LastInteraction",
    "churn": "Churn",
}

NEEDED = {
    "Age","Gender","Tenure","Usage","Support","PaymentDelay",
    "Subscription","Contract","TotalSpend","LastInteraction","Churn"
}

NUMERIC_COLS_HINT = {"Age","Tenure","Usage","Support","PaymentDelay","TotalSpend","LastInteraction"}

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        key = "".join(ch for ch in str(c).lower() if ch.isalnum())
        mapped = None
        for k, v in HEADER_CANON.items():
            if key.startswith(k):
                mapped = v
                break
        new_cols.append(mapped or c)
    df.columns = new_cols
    df = df.loc[:, ~df.columns.duplicated()]
    return df

def _normalize_churn(series: pd.Series) -> pd.Series:
    # Trim strings and map common values to 0/1, then coerce to numeric
    s = series.astype(str).str.strip().str.lower()
    mapping = {
        "yes": 1, "y": 1, "true": 1, "t": 1, "1": 1,
        "no": 0, "n": 0, "false": 0, "f": 0, "0": 0,
        "nan": np.nan, "": np.nan
    }
    s = s.map(lambda x: mapping[x] if x in mapping else x)
    s = pd.to_numeric(s, errors="coerce")
    return s

def load_training_df() -> pd.DataFrame:
    path = RAW_TRAIN if RAW_TRAIN.exists() else FALLBACK_SAMPLE
    df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)

    df = _canonicalize_columns(df)

    missing = NEEDED - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns after canonicalization: {missing}")

    if "CustomerID" in df.columns:
        df = df.drop(columns=["CustomerID"])  # ID not used for modeling

    # Normalize target
    df["Churn"] = _normalize_churn(df["Churn"])
    dropped = int(df["Churn"].isna().sum())
    if dropped:
        print(f"[clean] Dropping {dropped} rows with missing/invalid Churn")
    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].astype(int)

    # Coerce numerics & fill NA
    for col in NUMERIC_COLS_HINT & set(df.columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    # Fill remaining missing values
    for col in num_cols:
        if col != "Churn":
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    print(f"[clean] Final training frame: {df.shape[0]} rows x {df.shape[1]} cols")
    return df

def build_pipeline(num_cols, cat_cols) -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])
    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, eval_metric="auc",
        tree_method="hist", random_state=42
    )
    return Pipeline([("pre", pre), ("model", model)])

def main():
    df = load_training_df()
    y = df["Churn"].values
    X = df.drop(columns=["Churn"])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pipe = build_pipeline(num_cols, cat_cols)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    pipe.fit(X_tr, y_tr)

    proba = pipe.predict_proba(X_te)[:, 1]
    preds = (proba >= 0.5).astype(int)
    auc = roc_auc_score(y_te, proba)
    f1 = f1_score(y_te, preds)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipe, "num_cols": num_cols, "cat_cols": cat_cols, "metrics": {"AUC": float(auc), "F1": float(f1)}},
        MODEL_PATH
    )

    print(f"Saved model → {MODEL_PATH}")
    print(f"Validation AUC: {auc:.4f} | F1: {f1:.4f}")

if __name__ == "__main__":
    main()
