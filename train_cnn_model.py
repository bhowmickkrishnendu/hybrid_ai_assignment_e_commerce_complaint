"""
train_cnn_model.py
------------------
Trains a small CNN on Fashion-MNIST (10 clothing categories).
In the hybrid system, the CNN score represents the image-based risk signal.

Business Mapping:
  Certain product categories (e.g., sneakers, bags) correlate with higher
  return / complaint rates. The CNN learns to classify product images;
  this category prediction is then mapped to a complaint risk probability.

Fashion-MNIST label → risk weight mapping:
  T-shirt/top(0), Trouser(1), Pullover(2), Dress(3), Coat(4) → lower risk
  Sandal(5), Shirt(6), Sneaker(7), Bag(8), Ankle boot(9)    → higher risk
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

# ── 1. LOAD DATA ──────────────────────────────────────────────────────────────

def load_fashion_mnist():
    """Load and preprocess Fashion-MNIST."""
    (X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # Normalise pixel values to [0, 1]
    X_train_full = X_train_full.astype("float32") / 255.0
    X_test       = X_test.astype("float32") / 255.0

    # Add channel dimension: (N, 28, 28) → (N, 28, 28, 1)
    X_train_full = X_train_full[..., np.newaxis]
    X_test       = X_test[..., np.newaxis]

    # Carve out validation set (15% of train = 9000 samples)
    val_size    = int(len(X_train_full) * 0.15)
    X_val       = X_train_full[:val_size]
    y_val       = y_train_full[:val_size]
    X_train     = X_train_full[val_size:]
    y_train     = y_train_full[val_size:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# ── 2. BUILD CNN ──────────────────────────────────────────────────────────────

def build_cnn(input_shape=(28, 28, 1), num_classes=10):
    """Small CNN suitable for Fashion-MNIST."""
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Classifier head
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ], name="complaint_cnn")

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ── 3. CATEGORY → RISK SCORE MAPPING ─────────────────────────────────────────

# Higher-value / fragile product categories have higher complaint risk
CATEGORY_RISK_WEIGHTS = {
    0: 0.2,  # T-shirt/top  – low return rate
    1: 0.1,  # Trouser      – low return rate
    2: 0.3,  # Pullover     – medium
    3: 0.3,  # Dress        – medium
    4: 0.35, # Coat         – medium
    5: 0.55, # Sandal       – higher (size issues common)
    6: 0.4,  # Shirt        – medium-high
    7: 0.65, # Sneaker      – high (most returned)
    8: 0.7,  # Bag          – high (quality complaints)
    9: 0.6,  # Ankle boot   – high (size + quality)
}

def image_to_risk_score(model, image):
    """
    Given a (28,28,1) image, return a risk score in [0,1].
    Uses the weighted sum of category probabilities × risk weights.
    """
    if image.ndim == 3:
        image = image[np.newaxis, ...]          # add batch dim
    probs = model.predict(image, verbose=0)[0]  # shape (10,)
    risk_score = sum(probs[cat] * weight for cat, weight in CATEGORY_RISK_WEIGHTS.items())
    return float(np.clip(risk_score, 0.0, 1.0))


# ── 4. TRAIN & EVALUATE ───────────────────────────────────────────────────────

def train_and_evaluate():
    print("=" * 55)
    print("  CNN Component – Fashion-MNIST Image Classifier")
    print("=" * 55)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_fashion_mnist()
    print(f"\nTrain : {X_train.shape[0]} | Val : {X_val.shape[0]} | Test : {X_test.shape[0]}\n")

    model = build_cnn()
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        epochs=15,
        batch_size=64,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    y_val_pred  = np.argmax(model.predict(X_val,  verbose=0), axis=1)
    y_test_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    val_acc  = accuracy_score(y_val,  y_val_pred)
    val_f1   = f1_score(y_val,  y_val_pred, average="weighted")
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1  = f1_score(y_test, y_test_pred, average="weighted")

    print(f"\n── Validation  Accuracy: {val_acc:.4f}  F1: {val_f1:.4f}")
    print(f"── Test        Accuracy: {test_acc:.4f}  F1: {test_f1:.4f}\n")

    CLASS_NAMES = ["T-shirt/top","Trouser","Pullover","Dress","Coat",
                   "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
    print(classification_report(y_test, y_test_pred, target_names=CLASS_NAMES))

    # Demo: compute a risk score for first test image
    demo_score = image_to_risk_score(model, X_test[0])
    print(f"Demo CNN risk score for test[0] (label={CLASS_NAMES[y_test[0]]}): {demo_score:.4f}")

    # Save
    model.save("cnn_model.keras")
    print("✅  Saved: cnn_model.keras")

    metrics = {
        "model": "CNN (Fashion-MNIST)",
        "val_accuracy":  round(val_acc,  4),
        "val_f1":        round(val_f1,   4),
        "test_accuracy": round(test_acc, 4),
        "test_f1":       round(test_f1,  4),
        "category_risk_weights": CATEGORY_RISK_WEIGHTS,
    }
    with open("cnn_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return model, metrics


if __name__ == "__main__":
    train_and_evaluate()
