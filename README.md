# 🏦 Smart Loan Default & Credit Risk Predictor

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**An end-to-end Machine Learning project that predicts whether a loan application will be approved or rejected — with a Flask web app for real-time predictions and an interactive EDA dashboard.**

[Features](#-features) • [Installation](#-getting-started) • [How It Works](#-how-it-works) • [Results](#-sample-results) • [Web App](#-web-app-pages)

</div>

---

## 📌 Overview

This project builds a complete ML pipeline to predict **loan default risk** based on applicant details such as income, credit history, education, and employment status.

Built with **Python**, **scikit-learn**, and **Flask** — with a clean web UI for real-time predictions.

### What's included:
- 🔍 **Exploratory Data Analysis (EDA)** with auto-generated charts
- ⚙️ **Feature Engineering** to create meaningful derived features
- 🤖 **Two ML Models**: Logistic Regression & Random Forest (with comparison)
- 🌐 **Flask Web App** with a clean UI for real-time predictions

---

## ✨ Features

- 📊 Interactive EDA dashboard with visual insights
- 🧹 Automated data cleaning, missing value handling & encoding
- 🧠 Dual model training with accuracy comparison
- 💡 Smart feature engineering (EMI, Income Ratio, Log transforms)
- 🌐 User-friendly web interface for real-time loan prediction
- 💾 Model persistence using `pickle`

---

## 📁 Project Structure

```
Smart-Loan-Default-Credit-Risk-Predictor/
│
├── data/
│   └── loan_data.csv              # Dataset (614 records, 13 features)
│
├── models/                        # Auto-generated after running train_model.py
│   ├── model.pkl                  # Saved trained model
│   ├── scaler.pkl                 # Saved feature scaler
│   └── feature_names.pkl          # Column order used during training
│
├── static/
│   ├── css/
│   │   └── style.css              # All the styling for the web app
│   └── plots/                     # EDA charts (auto-generated during training)
│
├── templates/
│   ├── index.html                 # Main prediction form
│   ├── result.html                # Prediction result page
│   └── eda.html                   # Analytics / charts page
│
├── data_loader.py                 # Loads the CSV, prints a quick summary
├── preprocessing.py               # Cleans data, fills missing values, encodes categories
├── feature_engineering.py         # Creates new features (total income, EMI, etc.)
├── eda.py                         # Generates and saves all the analysis charts
├── train_model.py                 # Full training pipeline — run this first!
├── app.py                         # Flask web app
├── requirements.txt               # Python dependencies
└── README.md                      # You're reading this :)
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/sujansanepara/Smart-Loan-Default-Credit-Risk-Predictor.git
cd Smart-Loan-Default-Credit-Risk-Predictor
```

### 2. Create a Virtual Environment *(Recommended)*

```bash
# Create
python -m venv venv

# Activate
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

> ⚠️ **Run this before starting the web app!**

```bash
python train_model.py
```

This will:
- Load and preprocess the dataset
- Run EDA and save charts to `static/plots/`
- Train both Logistic Regression & Random Forest
- Print accuracy scores in the terminal
- Save the best model to the `models/` folder

### 5. Start the Web App

```bash
python app.py
```

Open your browser and go to: **http://127.0.0.1:5000**

---

## 🧠 How It Works

### Pipeline

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Flask App
```

### Models Used

| Model | Notes |
|---|---|
| 🔵 Logistic Regression | Fast, interpretable, great baseline |
| 🌲 Random Forest | Usually more accurate, handles non-linearity well |

> The project trains both and saves **Random Forest** as the production model. You can switch to Logistic Regression in `train_model.py` if it performs better on your run.

### Input Features

| Feature | Description |
|---|---|
| Gender | Male / Female |
| Married | Yes / No |
| Dependents | Number of dependents (0–3+) |
| Education | Graduate / Not Graduate |
| Self Employed | Yes / No |
| Applicant Income | Monthly income of the applicant |
| Co-Applicant Income | Monthly income of the co-applicant |
| Loan Amount | Requested loan amount (in thousands) |
| Loan Term | Repayment duration in months |
| Credit History | 1 = Good credit history, 0 = Bad |
| Property Area | Urban / Semiurban / Rural |

### Engineered Features

| Feature | Formula / Purpose |
|---|---|
| `TotalIncome` | Applicant Income + Co-applicant Income |
| `IncomeToLoanRatio` | TotalIncome / LoanAmount |
| `LogTotalIncome` | log(TotalIncome) — reduces skewness |
| `LogLoanAmount` | log(LoanAmount) — reduces skewness |
| `EMI` | LoanAmount / LoanTerm |

### Dataset

| Property | Value |
|---|---|
| Total Records | 614 rows |
| Features | 13 columns |
| Target | `Loan_Status` (Y = Approved, N = Rejected) |
| Class Distribution | ~69% Approved, ~31% Rejected |
| Source | Loan Prediction Problem Dataset (Kaggle) |

---

## 📊 Sample Results

After training, you'll typically see:

| Model | Accuracy |
|---|---|
| 🔵 Logistic Regression | ~80–82% |
| 🌲 Random Forest | ~78–82% |

> Results may vary slightly due to random state and data splits.

---

## 🌐 Web App Pages

| URL | Description |
|---|---|
| `/` | Prediction form — fill in applicant details |
| `/predict` | POST endpoint — processes form and shows result |
| `/eda` | Analytics page with all the EDA charts |

---

## 🛠️ Built With

| Library | Purpose |
|---|---|
| [Python 3.10+](https://www.python.org/) | Core programming language |
| [pandas](https://pandas.pydata.org/) | Data loading & manipulation |
| [scikit-learn](https://scikit-learn.org/) | ML models & preprocessing |
| [matplotlib](https://matplotlib.org/) + [seaborn](https://seaborn.pydata.org/) | Visualizations |
| [Flask](https://flask.palletsprojects.com/) | Web framework |

---

## 📌 Notes

- Run `train_model.py` **before** starting the Flask app — it generates the required `models/` files
- All EDA plots are auto-saved to `static/plots/` during training
- The model file is about 1–2 MB — small enough to include in version control

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👨‍💻 Author

**Sujan Sanepara**

[![GitHub](https://img.shields.io/badge/GitHub-sujansanepara-181717?style=flat&logo=github)](https://github.com/sujansanepara)
