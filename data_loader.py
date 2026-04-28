"""
data_loader.py
--------------
Just loads the CSV and gives us a quick look at the data.
Nothing fancy here — keep it simple.
"""

import pandas as pd
import os


def load_data(filepath="data/loan_data.csv"):
    # Make sure the file actually exists before trying to load it
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)
    print(f"✅ Data loaded successfully! Shape: {df.shape}")
    return df


def quick_summary(df):
    print("\n--- Dataset Overview ---")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nColumn names:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nData types:")
    print(df.dtypes)
    print("\nTarget variable (Loan_Status) distribution:")
    print(df["Loan_Status"].value_counts())


if __name__ == "__main__":
    df = load_data()
    quick_summary(df)
