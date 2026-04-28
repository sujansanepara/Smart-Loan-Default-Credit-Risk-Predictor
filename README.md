# 🏦 Smart Loan Default & Credit Risk Predictor

A machine learning project that predicts whether a loan application will be **approved or rejected** based on applicant details like income, credit history, education, and more.

Built with Python, scikit-learn, and Flask — with a clean web UI for real-time predictions.

---

## 📁 Project Structure

```
smart_loan_predictor/
│
├── data/
│   └── loan_data.csv            # The dataset (614 records, 13 features)
│
├── models/
│   ├── model.pkl                # Saved trained model (generated after training)
│   ├── scaler.pkl               # Saved feature scaler
│   └── feature_names.pkl        # Column order used during training
│
├── static/
│   ├── css/
│   │   └── style.css            # All the styling for the web app
│   └── plots/                   # EDA charts (auto-generated during training)
│
├── templates/
│   ├── index.html               # Main prediction form
│   ├── result.html              # Prediction result page
│   └── eda.html                 # Analytics/charts page
│
├── data_loader.py               # Loads the CSV, prints a quick summary
├── preprocessing.py             # Cleans data, fills missing values, encodes categories
├── feature_engineering.py       # Creates new features (total income, EMI, etc.)
├── eda.py                       # Generates and saves all the analysis charts
├── train_model.py               # Full training pipeline — run this first!
├── app.py                       # Flask web app
├── requirements.txt             # Python dependencies
└── README.md                    # You're reading this :)
```

---

## 🚀 Getting Started

### 1. Clone or download the project

```bash
git clone https://github.com/sujansanepara/smart_loan_predictor.git
cd smart_loan_predictor
```

### 2. Create a virtual environment (optional but recommended)

```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

This step loads the data, runs EDA, trains both models, and saves the best one.

```bash
python train_model.py
```

You'll see accuracy scores for both Logistic Regression and Random Forest printed in the terminal. Charts will be saved to `static/plots/`.

### 5. Start the web app

```bash
python app.py
```

Open your browser and go to: **http://127.0.0.1:5000**

---

## 🧠 How It Works

### Models Used
| Model | Notes |
|-------|-------|
| Logistic Regression | Fast, interpretable, great baseline |
| Random Forest | Usually more accurate, handles non-linearity well |

The project trains both and saves **Random Forest** as the production model (you can change this in `train_model.py` if LR performs better on your run).

### Features Used
The model uses these input features:

| Feature | Description |
|---------|-------------|
| Gender | Male / Female |
| Married | Yes / No |
| Dependents | Number of dependents (0–3+) |
| Education | Graduate / Not Graduate |
| Self Employed | Yes / No |
| Applicant Income | Monthly income of applicant |
| Co-Applicant Income | Monthly income of co-applicant |
| Loan Amount | Requested loan amount (in thousands) |
| Loan Term | Repayment duration in months |
| Credit History | 1 = Good, 0 = Bad |
| Property Area | Urban / Semiurban / Rural |

Plus engineered features:
- **TotalIncome** = Applicant + Co-applicant income
- **IncomeToLoanRatio** = TotalIncome / LoanAmount
- **LogTotalIncome** = log(TotalIncome) — reduces skew
- **LogLoanAmount** = log(LoanAmount) — reduces skew
- **EMI** = LoanAmount / LoanTerm

### Dataset
- **614 rows**, 13 columns
- Target: `Loan_Status` (Y = Approved, N = Rejected)
- ~69% approved, ~31% rejected
- Source: Loan Prediction Problem Dataset (Kaggle)

---

## 📊 Sample Results

After training, you'll typically see:

- **Logistic Regression**: ~80–82% accuracy
- **Random Forest**: ~78–82% accuracy

Results may vary slightly due to random state and data splits.

---

## 🌐 Web App Pages

| URL | Description |
|-----|-------------|
| `/` | Prediction form — fill in applicant details |
| `/predict` | POST endpoint — processes form and shows result |
| `/eda` | Analytics page with all the charts |

---

## 📌 Notes

- Run `train_model.py` **before** starting the Flask app — it generates the `models/` files
- All EDA plots are auto-saved to `static/plots/` during training
- The model file is about 1–2 MB — small enough to include in version control

---

## 🛠️ Built With

- **Python 3.10+**
- **pandas** — data manipulation
- **scikit-learn** — machine learning
- **matplotlib + seaborn** — visualizations
- **Flask** — web framework

---

