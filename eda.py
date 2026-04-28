"""
eda.py
------
Exploratory Data Analysis — let's understand the data before we model it.
Saves charts to the static folder so Flask can display them.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Save all plots here so the web app can show them
PLOT_DIR = "static/plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# Set a clean style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


def plot_loan_status(df):
    fig, ax = plt.subplots()
    counts = df["Loan_Status"].value_counts()
    colors = ["#4CAF50", "#f44336"]
    ax.bar(["Approved (Y)", "Rejected (N)"], counts.values, color=colors, width=0.5)
    ax.set_title("Loan Approval Distribution", fontsize=14)
    ax.set_ylabel("Count")
    for i, v in enumerate(counts.values):
        ax.text(i, v + 3, str(v), ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/loan_status.png")
    plt.close()
    print("📊 Saved: loan_status.png")


def plot_income_vs_loan(df):
    fig, ax = plt.subplots()
    colors = {"Y": "#4CAF50", "N": "#f44336"}
    for status, group in df.groupby("Loan_Status"):
        ax.scatter(group["ApplicantIncome"], group["LoanAmount"],
                   label=f"Status: {status}", alpha=0.5,
                   color=colors.get(status, "blue"), s=30)
    ax.set_title("Applicant Income vs Loan Amount", fontsize=14)
    ax.set_xlabel("Applicant Income")
    ax.set_ylabel("Loan Amount")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/income_vs_loan.png")
    plt.close()
    print("📊 Saved: income_vs_loan.png")


def plot_credit_history(df):
    fig, ax = plt.subplots()
    credit_counts = df.groupby(["Credit_History", "Loan_Status"]).size().unstack(fill_value=0)
    credit_counts.plot(kind="bar", ax=ax, color=["#f44336", "#4CAF50"], width=0.5)
    ax.set_title("Credit History vs Loan Approval", fontsize=14)
    ax.set_xlabel("Credit History (0 = Bad, 1 = Good)")
    ax.set_ylabel("Count")
    ax.set_xticklabels(["Bad Credit", "Good Credit"], rotation=0)
    ax.legend(["Rejected", "Approved"])
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/credit_history.png")
    plt.close()
    print("📊 Saved: credit_history.png")


def plot_education_approval(df):
    fig, ax = plt.subplots()
    edu_counts = df.groupby(["Education", "Loan_Status"]).size().unstack(fill_value=0)
    edu_counts.plot(kind="bar", ax=ax, color=["#f44336", "#4CAF50"], width=0.4)
    ax.set_title("Education vs Loan Approval", fontsize=14)
    ax.set_xlabel("Education Level")
    ax.set_ylabel("Count")
    ax.set_xticklabels(["Graduate", "Not Graduate"], rotation=0)
    ax.legend(["Rejected", "Approved"])
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/education_approval.png")
    plt.close()
    print("📊 Saved: education_approval.png")


def plot_correlation_heatmap(df):
    # Only numeric columns for correlation
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="RdYlGn",
                linewidths=0.5, ax=ax, center=0)
    ax.set_title("Feature Correlation Heatmap", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/correlation_heatmap.png")
    plt.close()
    print("📊 Saved: correlation_heatmap.png")


def run_eda(df):
    print("\n--- Running EDA ---")
    plot_loan_status(df)
    plot_income_vs_loan(df)
    plot_credit_history(df)
    plot_education_approval(df)
    plot_correlation_heatmap(df)
    print("✅ All EDA plots saved to static/plots/")


if __name__ == "__main__":
    from data_loader import load_data
    df = load_data()
    run_eda(df)
