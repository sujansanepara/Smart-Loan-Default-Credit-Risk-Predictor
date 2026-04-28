"""
app.py
------
Flask web app — the UI layer of the whole project.
Loads the saved model and handles prediction requests from the form.

Make sure you've run train_model.py first!
"""

import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for

app = Flask(__name__)

# Load the saved model, scaler, and feature names
try:
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print("⚠️  Model not found. Please run train_model.py first.")
    model = None
    scaler = None
    feature_names = None


@app.route("/")
def home():
    # Just show the input form
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return "Model not loaded. Run train_model.py first.", 500

    try:
        # Grab values from the HTML form
        gender = int(request.form["gender"])               # 1=Male, 0=Female
        married = int(request.form["married"])             # 1=Yes, 0=No
        dependents = int(request.form["dependents"])       # 0, 1, 2, 3
        education = int(request.form["education"])         # 1=Graduate, 0=Not Graduate
        self_employed = int(request.form["self_employed"]) # 1=Yes, 0=No
        applicant_income = float(request.form["applicant_income"])
        coapplicant_income = float(request.form["coapplicant_income"])
        loan_amount = float(request.form["loan_amount"])
        loan_term = float(request.form["loan_term"])
        credit_history = int(request.form["credit_history"]) # 1=Good, 0=Bad
        property_area = request.form["property_area"]       # Semiurban, Urban, Rural

        # --- Replicate the same feature engineering from training ---
        total_income = applicant_income + coapplicant_income
        income_to_loan_ratio = total_income / (loan_amount + 1)
        log_total_income = np.log1p(total_income)
        log_loan_amount = np.log1p(loan_amount)
        emi = loan_amount / (loan_term + 1)

        # Property area one-hot (same columns as training: drop_first=True → Rural is baseline)
        property_semiurban = 1 if property_area == "Semiurban" else 0
        property_urban = 1 if property_area == "Urban" else 0

        # Build the feature dict — must match training columns exactly
        input_data = {
            "Gender": gender,
            "Married": married,
            "Dependents": dependents,
            "Education": education,
            "Self_Employed": self_employed,
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "TotalIncome": total_income,
            "IncomeToLoanRatio": income_to_loan_ratio,
            "LogTotalIncome": log_total_income,
            "LogLoanAmount": log_loan_amount,
            "EMI": emi,
            "Property_Area_Semiurban": property_semiurban,
            "Property_Area_Urban": property_urban,
        }

        # Convert to DataFrame and align columns to training order
        input_df = pd.DataFrame([input_data])
        input_df = input_df.reindex(columns=feature_names, fill_value=0)

        # Predict!
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]

        # 1 = Approved, 0 = Rejected
        result = "Approved ✅" if prediction == 1 else "Rejected ❌"
        confidence = round(max(probability) * 100, 2)
        risk_level = get_risk_level(probability[0])  # probability of rejection

        return render_template(
            "result.html",
            result=result,
            confidence=confidence,
            risk_level=risk_level,
            approved=(prediction == 1),
            input_data=request.form,
        )

    except Exception as e:
        return f"Something went wrong: {str(e)}", 400


def get_risk_level(rejection_prob):
    """Give a human-readable risk rating based on rejection probability"""
    if rejection_prob < 0.2:
        return "Low Risk 🟢"
    elif rejection_prob < 0.5:
        return "Medium Risk 🟡"
    else:
        return "High Risk 🔴"


@app.route("/eda")
def eda_page():
    """Simple page that shows all the EDA plots"""
    return render_template("eda.html")


if __name__ == "__main__":
    app.run(debug=True)
