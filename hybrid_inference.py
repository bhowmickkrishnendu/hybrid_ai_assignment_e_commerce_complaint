"""
hybrid_inference.py
-------------------
End-to-end inference pipeline that combines:
  1. Classical ML model (tabular + TF-IDF text features)
  2. CNN model (image category → risk score)
  3. RNN/BiLSTM model (complaint text sequence)
  4. Weighted hybrid scoring
  5. LLM explanation layer

Usage:
    python hybrid_inference.py                    # uses sample_inputs.json
    python hybrid_inference.py --input my.json    # custom input file
"""

import os
import json
import argparse
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from scipy.sparse import hstack, csr_matrix

from llm_explainer import generate_explanation

# ── HYBRID SCORING WEIGHTS ────────────────────────────────────────────────────
W_CLASSICAL = 0.40
W_CNN       = 0.30
W_RNN       = 0.30

# ── RISK THRESHOLDS ───────────────────────────────────────────────────────────
def score_to_risk(score: float) -> str:
    if score < 0.45:
        return "low"
    elif score < 0.75:
        return "medium"
    else:
        return "high"

# ── CNN RISK MAPPING (must match train_cnn_model.py) ─────────────────────────
CATEGORY_RISK_WEIGHTS = {
    0: 0.2, 1: 0.1, 2: 0.3, 3: 0.3, 4: 0.35,
    5: 0.55, 6: 0.4, 7: 0.65, 8: 0.7, 9: 0.6,
}

IMAGE_CATEGORY_MAP = {
    "normal_product":   7,   # Sneaker – most returned, medium-high risk
    "damaged_product":  8,   # Bag – high complaint risk
    "wrong_item":       6,   # Shirt – wrong item, medium-high risk
    "missing_item":     8,   # Bag – high risk
    "defective":        9,   # Ankle boot – high quality complaints
    "packaging_issue":  2,   # Pullover – medium risk
    "default":          4,   # Coat – medium risk
}


# ── MODEL LOADING ─────────────────────────────────────────────────────────────

class HybridModel:
    def __init__(self):
        print("Loading models…")
        self.classical_model = joblib.load("classical_model.pkl")
        self.tfidf            = joblib.load("tfidf_vectorizer.pkl")
        self.scaler           = joblib.load("tabular_scaler.pkl")
        self.cnn_model        = tf.keras.models.load_model("cnn_model.keras")
        self.rnn_model        = tf.keras.models.load_model("rnn_model.keras")
        self.rnn_tokenizer    = joblib.load("rnn_tokenizer.pkl")
        print("✅  All models loaded.\n")

    # ── CLASSICAL SCORE ───────────────────────────────────────────────────────

    def classical_score(self, complaint_text: str, previous_complaints: int,
                        order_value: float, tenure: float, account_type: str) -> float:
        """Returns probability of high escalation risk from Classical ML."""
        X_text   = self.tfidf.transform([complaint_text])
        acc_enc  = 1 if account_type == "premium" else 0
        tabular  = np.array([[previous_complaints, order_value, tenure, acc_enc]])
        tab_scaled = self.scaler.transform(tabular)
        X        = hstack([X_text, csr_matrix(tab_scaled)])
        probs    = self.classical_model.predict_proba(X)[0]
        # probs[0]=low, probs[1]=medium, probs[2]=high
        # risk score = weighted sum leaning toward high
        risk_score = 0.0 * probs[0] + 0.5 * probs[1] + 1.0 * probs[2]
        return float(np.clip(risk_score, 0.0, 1.0))

    # ── CNN SCORE ─────────────────────────────────────────────────────────────

    def cnn_score(self, image_category: str) -> float:
        """
        Converts image category string to a risk score using the CNN.
        In a real system, the actual image would be passed to the CNN.
        Here we simulate by using the category → Fashion-MNIST label mapping.
        """
        cat_idx = IMAGE_CATEGORY_MAP.get(image_category, IMAGE_CATEGORY_MAP["default"])

        # Simulate CNN by generating a Fashion-MNIST-like random image for
        # this category and running it through the CNN
        np.random.seed(cat_idx * 7)
        dummy_img = np.random.rand(1, 28, 28, 1).astype("float32") * 0.3
        probs     = self.cnn_model.predict(dummy_img, verbose=0)[0]
        # Use category-weighted risk score
        risk_score = sum(probs[c] * w for c, w in CATEGORY_RISK_WEIGHTS.items())
        # Blend with deterministic category weight so results are sensible
        deterministic = CATEGORY_RISK_WEIGHTS[cat_idx]
        blended = 0.5 * risk_score + 0.5 * deterministic
        return float(np.clip(blended, 0.0, 1.0))

    # ── RNN SCORE ─────────────────────────────────────────────────────────────

    def rnn_score(self, complaint_text: str) -> float:
        """Returns probability of high escalation risk from BiLSTM."""
        MAX_LEN = 50
        seq     = self.rnn_tokenizer.texts_to_sequences([complaint_text])
        padded  = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        probs   = self.rnn_model.predict(padded, verbose=0)[0]
        risk_score = 0.0 * probs[0] + 0.5 * probs[1] + 1.0 * probs[2]
        return float(np.clip(risk_score, 0.0, 1.0))

    # ── HYBRID INFERENCE ──────────────────────────────────────────────────────

    def predict(self, complaint: dict) -> dict:
        """
        Full end-to-end prediction for one complaint.

        Parameters
        ----------
        complaint : dict with keys:
            customer_id, complaint_text, customer_tenure_years,
            previous_complaints, order_value, account_type, image_category

        Returns
        -------
        dict with model_outputs and llm_explanation
        """
        text         = complaint.get("complaint_text", "")
        tenure       = complaint.get("customer_tenure_years", 1.0)
        prev         = complaint.get("previous_complaints", 0)
        order_val    = complaint.get("order_value", 50.0)
        account_type = complaint.get("account_type", "standard")
        img_cat      = complaint.get("image_category", "default")

        c_score = self.classical_score(text, prev, order_val, tenure, account_type)
        cnn_s   = self.cnn_score(img_cat)
        rnn_s   = self.rnn_score(text)

        final = W_CLASSICAL * c_score + W_CNN * cnn_s + W_RNN * rnn_s
        risk  = score_to_risk(final)

        model_outputs = {
            "classical_score": round(c_score, 4),
            "cnn_score":       round(cnn_s,   4),
            "rnn_score":       round(rnn_s,   4),
            "final_score":     round(final,   4),
            "risk_level":      risk,
        }

        llm_exp = generate_explanation(text, model_outputs)

        return {
            "customer_id":   complaint.get("customer_id", "unknown"),
            "model_outputs": model_outputs,
            "llm_explanation": llm_exp,
        }

    def predict_batch(self, complaints: list) -> list:
        return [self.predict(c) for c in complaints]


# ── DECISION POLICY ───────────────────────────────────────────────────────────

def apply_decision_policy(result: dict) -> dict:
    """
    Applies the business decision policy on top of the model output.
    Returns an enriched result with the final_action field.
    """
    mo = result["model_outputs"]
    exp = result["llm_explanation"]
    risk = mo["risk_level"]
    score = mo["final_score"]

    # Check for model disagreement (high variance between scores)
    scores = [mo["classical_score"], mo["cnn_score"], mo["rnn_score"]]
    score_range = max(scores) - min(scores)
    disagree = score_range > 0.35

    human_required = (
        risk in ("medium", "high") or
        exp.get("human_review_required", False) or
        disagree
    )

    if risk == "low" and not disagree:
        action = "auto_reply"
    elif risk == "medium" or disagree:
        action = "manual_review"
    else:
        action = "escalate_immediately"

    result["decision"] = {
        "final_action":    action,
        "human_required":  human_required,
        "model_disagree":  disagree,
        "score_range":     round(score_range, 4),
    }
    return result


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hybrid AI Inference")
    parser.add_argument("--input", default="sample_inputs.json")
    parser.add_argument("--output", default="sample_outputs.json")
    args = parser.parse_args()

    with open(args.input) as f:
        complaints = json.load(f)

    hybrid = HybridModel()
    results = []

    for complaint in complaints:
        print(f"Processing: {complaint.get('customer_id', '?')}  ", end="")
        result = hybrid.predict(complaint)
        result = apply_decision_policy(result)
        results.append(result)
        print(f"→ Risk: {result['model_outputs']['risk_level'].upper():6s}  "
              f"Score: {result['model_outputs']['final_score']:.3f}  "
              f"Action: {result['decision']['final_action']}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅  Results saved to {args.output}")
    print(json.dumps(results[0], indent=2))


if __name__ == "__main__":
    main()
