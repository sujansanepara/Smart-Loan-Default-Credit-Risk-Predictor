"""
preprocessing.py
----------------
Handles all the messy data cleaning stuff.
Fill missing values, encode categories, that kind of thing.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def clean_data(df):
    # Drop Loan_ID — it's just an identifier, useless for prediction
    df = df.drop(columns=["Loan_ID"]).copy()

    # --- Fill missing values ---

    # For categorical columns, fill with the most common value (mode)
    cat_cols_with_nulls = ["Gender", "Married", "Self_Employed", "Credit_History", "Loan_Amount_Term"]
    for col in cat_cols_with_nulls:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0])

    # Dependents has a weird '3+' value — replace it first, then fill nulls
    df["Dependents"] = df["Dependents"].replace("3+", "3")
    df["Dependents"] = df["Dependents"].fillna(df["Dependents"].mode()[0])
    df["Dependents"] = df["Dependents"].astype(float).astype(int)

    # For numeric columns, use median (less affected by outliers)
    df["LoanAmount"] = df["LoanAmount"].fillna(df["LoanAmount"].median())

    print(f"✅ Missing values after cleaning: {df.isnull().sum().sum()}")
    return df


def encode_features(df):
    """
    Convert text categories to numbers so the model can understand them.
    Using LabelEncoder for binary categories, and get_dummies for multi-class.
    """
    # These columns only have 2 options — LabelEncoder works fine
    binary_cols = ["Gender", "Married", "Education", "Self_Employed"]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])

    # Property_Area has 3 options — one-hot encoding is cleaner here
    df = pd.get_dummies(df, columns=["Property_Area"], drop_first=True)

    # Encode the target: Y=1 (approved), N=0 (not approved)
    df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

    print("✅ Encoding done. Final columns:", df.columns.tolist())
    return df


def get_features_and_target(df):
    X = df.drop(columns=["Loan_Status"])
    y = df["Loan_Status"]
    return X, y
