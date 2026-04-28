"""
train_model.py
--------------
This is where all the training happens.
We train two models — Logistic Regression and Random Forest —
compare them, and save the better one.

Run this file once before starting the Flask app.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from data_loader import load_data
from preprocessing import clean_data, encode_features, get_features_and_target
from feature_engineering import add_features
from eda import run_eda

# Where we'll save the trained model and scaler
MODEL_DIR = "models"
PLOT_DIR = "static/plots"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


def prepare_data():
    """Full pipeline: load → clean → engineer → encode → split"""
    df = load_data()

    # Run EDA and save the plots
    run_eda(df)

    df = clean_data(df)
    df = add_features(df)
    df = encode_features(df)

    X, y = get_features_and_target(df)

    # Split: 80% training, 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTraining set: {X_train.shape}, Test set: {X_test.shape}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


def scale_features(X_train, X_test):
    """Scale features — helps Logistic Regression a lot"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def train_logistic_regression(X_train, y_train):
    print("\n--- Training Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    cv_scores = cross_val_score(lr, X_train, y_train, cv=5)
    print(f"Cross-val accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return lr


def train_random_forest(X_train, y_train):
    print("\n--- Training Random Forest ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf.fit(X_train, y_train)
    cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
    print(f"Cross-val accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    return rf


def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n--- Evaluation: {model_name} ---")
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))
    return acc, y_pred


def save_confusion_matrix(y_test, y_pred, model_name):
    """Save a nice-looking confusion matrix plot"""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Rejected", "Approved"],
        yticklabels=["Rejected", "Approved"],
        ax=ax
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    filename = model_name.lower().replace(" ", "_")
    plt.savefig(f"{PLOT_DIR}/cm_{filename}.png")
    plt.close()
    print(f"📊 Confusion matrix saved: cm_{filename}.png")


def save_feature_importance(model, feature_names):
    """Only Random Forest gives us feature importance"""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(importances)), importances[indices], color="#2196F3", alpha=0.8)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right")
    ax.set_title("Feature Importance — Random Forest", fontsize=14)
    ax.set_ylabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/feature_importance.png")
    plt.close()
    print("📊 Feature importance plot saved.")


def save_model(model, scaler, feature_names):
    """Pickle the model and scaler so the Flask app can load them"""
    with open(f"{MODEL_DIR}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{MODEL_DIR}/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)
    print(f"\n✅ Model saved to {MODEL_DIR}/model.pkl")
    print(f"✅ Scaler saved to {MODEL_DIR}/scaler.pkl")


def main():
    print("=" * 50)
    print("  Smart Loan Default & Credit Risk Predictor")
    print("  Model Training Pipeline")
    print("=" * 50)

    # Step 1: Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data()

    # Step 2: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Step 3: Train both models
    lr_model = train_logistic_regression(X_train_scaled, y_train)
    rf_model = train_random_forest(X_train, y_train)  # RF doesn't need scaling

    # Step 4: Evaluate both
    lr_acc, lr_preds = evaluate_model(lr_model, X_test_scaled, y_test, "Logistic Regression")
    rf_acc, rf_preds = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Step 5: Save confusion matrices
    save_confusion_matrix(y_test, lr_preds, "Logistic Regression")
    save_confusion_matrix(y_test, rf_preds, "Random Forest")

    # Step 6: Feature importance for RF
    save_feature_importance(rf_model, feature_names)

    # Step 7: Pick the winner and save it
    print("\n--- Model Comparison ---")
    print(f"Logistic Regression Accuracy: {lr_acc:.4f}")
    print(f"Random Forest Accuracy:       {rf_acc:.4f}")

    # We'll save Random Forest as the main model — usually performs better
    # But you can swap this to lr_model if LR wins on your data
    best_model = rf_model
    best_name = "Random Forest"
    print(f"\n🏆 Saving {best_name} as the production model.")
    save_model(best_model, scaler, feature_names)

    print("\n✅ Training complete! You can now run: python app.py")


if __name__ == "__main__":
    main()
