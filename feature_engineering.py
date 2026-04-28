"""
feature_engineering.py
-----------------------
Create new features from existing ones.
Sometimes combining columns gives the model better signals.
"""

import numpy as np


def add_features(df):
    """
    A few simple engineered features that might help the model.
    """
    # Total income = applicant + co-applicant
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]

    # Income to loan ratio — can they afford this?
    # Adding 1 to avoid division by zero just in case
    df["IncomeToLoanRatio"] = df["TotalIncome"] / (df["LoanAmount"] + 1)

    # Log transform of income — reduces the effect of extreme outliers
    df["LogTotalIncome"] = np.log1p(df["TotalIncome"])
    df["LogLoanAmount"] = np.log1p(df["LoanAmount"])

    # EMI estimate (Loan amount divided by tenure in months)
    df["EMI"] = df["LoanAmount"] / (df["Loan_Amount_Term"] + 1)

    print("✅ Feature engineering done. New features added:")
    new_feats = ["TotalIncome", "IncomeToLoanRatio", "LogTotalIncome", "LogLoanAmount", "EMI"]
    print("  ", new_feats)
    return df


if __name__ == "__main__":
    from data_loader import load_data
    from preprocessing import clean_data, encode_features
    df = load_data()
    df = clean_data(df)
    df = add_features(df)
    df = encode_features(df)
    print("\nFinal dataframe shape:", df.shape)
    print("Columns:", df.columns.tolist())
