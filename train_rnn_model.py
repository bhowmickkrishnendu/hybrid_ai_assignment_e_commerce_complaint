"""
train_rnn_model.py
------------------
Trains a Bidirectional LSTM on synthetic complaint text to predict
escalation risk (Low=0 / Medium=1 / High=2).

Why Bidirectional LSTM?
  • Reads text both forward and backward, capturing context that a simple
    RNN can miss.
  • Handles variable-length complaint sequences naturally.
  • Outperforms vanilla RNN on longer, angrier complaints (vanishing gradient
    is less of a problem).
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

# ── 1. DATA ───────────────────────────────────────────────────────────────────

def generate_text_data(n_samples=1500):
    """Generate labelled complaint text with realistic language."""

    low_templates = [
        "The delivery arrived one day late but the product is fine.",
        "Minor cosmetic damage on the packaging, product works well.",
        "Product is slightly different shade than shown in photo.",
        "Expected faster shipping but happy with the item overall.",
        "The item works perfectly, just the box was a bit squashed.",
        "Small defect noticed but nothing that affects functionality.",
        "Delivery was okay. Slight delay but no major issues.",
        "Product is as described. No complaints really.",
    ]
    medium_templates = [
        "I received the wrong colour and need a replacement urgently.",
        "The product stopped working after just one week of use.",
        "I am still waiting for my refund after two weeks.",
        "Customer service has not responded to my emails.",
        "The item arrived damaged and I would like a replacement.",
        "Wrong size was delivered despite the correct order placed.",
        "I have been waiting for a resolution for several days.",
        "The product quality does not match what was advertised.",
        "Support team is not helpful and my issue is unresolved.",
    ]
    high_templates = [
        "I am furious. The product is completely broken and nobody responds.",
        "This is fraud. I have contacted support five times with no reply.",
        "The product caused injury and I will escalate this immediately.",
        "Worst purchase ever. I am reporting this to consumer protection.",
        "Three months and no refund. I am taking legal action now.",
        "Product arrived completely shattered. Support ignores my messages.",
        "I am extremely angry. This is completely unacceptable behaviour.",
        "The company is unresponsive and the product is dangerous.",
        "I demand a refund immediately or I will charge back my credit card.",
    ]

    np.random.seed(42)
    texts, labels = [], []
    fillers = ["", "Please help.", "This is urgent.", "Very disappointed.", "Thanks for nothing."]

    for _ in range(n_samples):
        label = np.random.choice([0, 1, 2], p=[0.38, 0.35, 0.27])
        if label == 0:
            text = np.random.choice(low_templates)
        elif label == 1:
            text = np.random.choice(medium_templates)
        else:
            text = np.random.choice(high_templates)
        text = text + " " + np.random.choice(fillers)
        texts.append(text.strip())
        labels.append(label)

    return texts, np.array(labels)


# ── 2. TOKENISE & PAD ─────────────────────────────────────────────────────────

def tokenise(texts, max_words=5000, max_len=50):
    tok = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tok.fit_on_texts(texts)
    seqs   = tok.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")
    return padded, tok


# ── 3. BUILD BILSTM ───────────────────────────────────────────────────────────

def build_bilstm(vocab_size=5001, embed_dim=64, max_len=50, num_classes=3):
    model = keras.Sequential([
        layers.Embedding(vocab_size, embed_dim, input_length=max_len),
        layers.SpatialDropout1D(0.2),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ], name="complaint_bilstm")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── 4. TRAIN & EVALUATE ───────────────────────────────────────────────────────

def train_and_evaluate():
    print("=" * 55)
    print("  RNN Component – Bidirectional LSTM on Complaint Text")
    print("=" * 55)

    MAX_WORDS = 5000
    MAX_LEN   = 50

    texts, y = generate_text_data(n_samples=1500)
    X, tokenizer = tokenise(texts, MAX_WORDS, MAX_LEN)

    # 70 / 15 / 15 split
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
    X_val,   X_test, y_val,   y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

    print(f"\nTrain: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n")

    model = build_bilstm(vocab_size=MAX_WORDS + 1, embed_dim=64, max_len=MAX_LEN)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=0),
    ]

    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    y_val_pred  = np.argmax(model.predict(X_val,  verbose=0), axis=1)
    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    val_acc  = accuracy_score(y_val,  y_val_pred)
    val_f1   = f1_score(y_val,  y_val_pred, average="weighted")
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1  = f1_score(y_test, y_test_pred, average="weighted")

    print(f"\n── Validation  Accuracy: {val_acc:.4f}  F1: {val_f1:.4f}")
    print(f"── Test        Accuracy: {test_acc:.4f}  F1: {test_f1:.4f}\n")
    print(classification_report(y_test, y_test_pred, target_names=["low", "medium", "high"]))

    # Save
    model.save("rnn_model.keras")
    joblib.dump(tokenizer, "rnn_tokenizer.pkl")
    print("✅  Saved: rnn_model.keras, rnn_tokenizer.pkl")

    metrics = {
        "model": "Bidirectional LSTM",
        "val_accuracy":  round(val_acc,  4),
        "val_f1":        round(val_f1,   4),
        "test_accuracy": round(test_acc, 4),
        "test_f1":       round(test_f1,  4),
    }
    with open("rnn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model, tokenizer, metrics


if __name__ == "__main__":
    train_and_evaluate()
