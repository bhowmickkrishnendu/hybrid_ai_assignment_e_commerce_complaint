"""
train_classical_model.py
------------------------
Trains a Classical ML baseline (Random Forest) on synthetic tabular + TF-IDF text features.
Saves the trained model and vectorizer for use in hybrid_inference.py.

Business Goal: Predict complaint escalation risk (Low=0 / Medium=1 / High=2)
using structured customer data and complaint text.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from scipy.sparse import hstack, csr_matrix
import joblib

# ── 1. SYNTHETIC DATA GENERATION ─────────────────────────────────────────────
np.random.seed(42)

def generate_synthetic_data(n_samples=1000):
    """Generate realistic e-commerce complaint data."""

    low_texts = [
        "The delivery was slightly delayed.",
        "Product is fine but packaging was a bit damaged.",
        "Minor issue with sizing, otherwise happy.",
        "Good product, just took longer than expected.",
        "Small scratch on the box, product itself is perfect.",
        "Delivery was a day late but everything arrived.",
        "Product works but manual is unclear.",
        "The color is slightly different from the photo.",
    ]
    medium_texts = [
        "I received the wrong item and need a replacement.",
        "Product stopped working after one week.",
        "I've been waiting for a refund for two weeks now.",
        "The item is defective and I want a replacement.",
        "Customer service was unhelpful when I called.",
        "My order was partially fulfilled, missing two items.",
        "Wrong size delivered despite correct order.",
        "The product quality is much worse than advertised.",
    ]
    high_texts = [
        "I am extremely angry. Product broke immediately and nobody replies.",
        "This is fraud! I've contacted support 5 times and no response.",
        "Completely broken product. I will escalate to consumer protection.",
        "Worst experience ever. File a complaint with the bank.",
        "Product is dangerous and caused injury. Escalate immediately.",
        "I have been waiting 3 months for a refund. Taking legal action.",
        "Product arrived broken and support is useless. Furious.",
        "This company is a scam. Reporting to trading standards.",
    ]

    records = []
    for i in range(n_samples):
        risk_label = np.random.choice(["low", "medium", "high"], p=[0.4, 0.35, 0.25])

        if risk_label == "low":
            text = np.random.choice(low_texts)
            prev_complaints = np.random.randint(0, 2)
            order_value = np.random.uniform(10, 80)
            tenure = np.random.uniform(1, 10)
        elif risk_label == "medium":
            text = np.random.choice(medium_texts)
            prev_complaints = np.random.randint(1, 4)
            order_value = np.random.uniform(50, 200)
            tenure = np.random.uniform(0.5, 5)
        else:
            text = np.random.choice(high_texts)
            prev_complaints = np.random.randint(3, 8)
            order_value = np.random.uniform(80, 500)
            tenure = np.random.uniform(0, 3)

        # Add noise
        text = text + " " + np.random.choice(["", "Please help.", "Thanks.", "Urgent!"])

        records.append({
            "complaint_text": text.strip(),
            "previous_complaints": prev_complaints,
            "order_value": order_value,
            "customer_tenure_years": tenure,
            "account_type": np.random.choice(["standard", "premium"], p=[0.7, 0.3]),
            "risk_label": risk_label,
        })

    return pd.DataFrame(records)


# ── 2. FEATURE ENGINEERING ────────────────────────────────────────────────────

def prepare_features(df):
    """Combine TF-IDF text features with scaled tabular features."""

    # Encode labels
    label_map = {"low": 0, "medium": 1, "high": 2}
    y = df["risk_label"].map(label_map).values

    # TF-IDF on complaint text
    tfidf = TfidfVectorizer(max_features=200, ngram_range=(1, 2), stop_words="english")
    X_text = tfidf.fit_transform(df["complaint_text"])

    # Tabular features
    account_enc = (df["account_type"] == "premium").astype(int).values.reshape(-1, 1)
    tabular = np.hstack([
        df[["previous_complaints", "order_value", "customer_tenure_years"]].values,
        account_enc
    ])

    scaler = StandardScaler()
    tabular_scaled = scaler.fit_transform(tabular)

    X = hstack([X_text, csr_matrix(tabular_scaled)])

    return X, y, tfidf, scaler


# ── 3. TRAIN / EVALUATE ───────────────────────────────────────────────────────

def train_and_evaluate():
    print("=" * 55)
    print("  Classical ML Baseline – Random Forest Classifier")
    print("=" * 55)

    df = generate_synthetic_data(n_samples=1200)
    print(f"\nDataset shape : {df.shape}")
    print(f"Target distribution:\n{df['risk_label'].value_counts()}\n")

    X, y, tfidf, scaler = prepare_features(df)

    # 70 / 15 / 15 split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    print(f"Train size : {X_train.shape[0]}")
    print(f"Val size   : {X_val.shape[0]}")
    print(f"Test size  : {X_test.shape[0]}\n")

    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)

    # Validation
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1  = f1_score(y_val, y_val_pred, average="weighted")
    print(f"── Validation Results ──")
    print(f"Accuracy : {val_acc:.4f}")
    print(f"F1 Score : {val_f1:.4f}\n")

    # Test
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1  = f1_score(y_test, y_test_pred, average="weighted")
    print(f"── Test Results ──")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"F1 Score : {test_f1:.4f}\n")
    print(classification_report(y_test, y_test_pred, target_names=["low", "medium", "high"]))

    # Save model artifacts
    joblib.dump(model, "classical_model.pkl")
    joblib.dump(tfidf,  "tfidf_vectorizer.pkl")
    joblib.dump(scaler, "tabular_scaler.pkl")
    print("✅  Saved: classical_model.pkl, tfidf_vectorizer.pkl, tabular_scaler.pkl")

    # Save metrics
    metrics = {
        "model": "RandomForestClassifier",
        "val_accuracy": round(val_acc, 4),
        "val_f1":       round(val_f1,  4),
        "test_accuracy": round(test_acc, 4),
        "test_f1":       round(test_f1,  4),
    }
    with open("classical_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model, tfidf, scaler, metrics


if __name__ == "__main__":
    train_and_evaluate()
