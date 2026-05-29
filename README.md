# Hybrid AI Complaint Escalation System by Krishnendu

## Problem Framing

### Business Problem
An e-commerce company receives thousands of customer complaints daily. Manual triage is slow and inconsistent. The system predicts **complaint escalation risk** (Low / Medium / High) and recommends a business action (auto-reply / manual review / escalate immediately).

### Prediction Target
Escalation risk level: `low`, `medium`, `high`

### Users
Customer support team, automated support platform, operations managers.

### Business Actions
| Risk | Action |
|------|--------|
| Low | Auto-reply with template |
| Medium | Queue for human agent |
| High | Escalate to senior manager immediately |

### False Positive Impact (predicted High but actually Low)
Wastes agent time on low-priority tickets. Minor cost.

### False Negative Impact (predicted Low but actually High)
High-risk complaint is auto-replied and unresolved. Customer escalates externally (legal, social media). **High business cost.**

### Human Review
Always required for medium and high risk cases.

---

## System Architecture

```
Customer Complaint Input
         |
         ├── Tabular features + TF-IDF text ──► Classical ML (Random Forest)
         │                                              │
         ├── Image/product category ──────────► CNN (Fashion-MNIST)
         │                                              │
         ├── Complaint text sequence ──────────► RNN / BiLSTM
         │                                              │
         └─────────────────── Weighted Hybrid Score ────┘
                                       │
                              final_score = 0.4×ML + 0.3×CNN + 0.3×RNN
                                       │
                              LLM Explanation (Gemini)
                                       │
                             Final Business Decision
```

---

## Project Structure

```
hybrid_ai_assignment/
│
├── README.md                  ← You are here
├── requirements.txt           ← Python dependencies
│
├── data_exploration.ipynb     ← Dataset EDA and visualisations
│
├── train_classical_model.py   ← Train Random Forest on tabular + TF-IDF
├── train_cnn_model.py         ← Train CNN on Fashion-MNIST
├── train_rnn_model.py         ← Train Bidirectional LSTM on text
│
├── hybrid_inference.py        ← End-to-end inference pipeline
├── llm_explainer.py           ← LLM explanation layer (Gemini + fallback)
│
├── evaluation_report.md       ← Component and system metrics
├── decision_policy.md         ← Business rules for auto-reply / escalation
├── mini_model_card.md         ← Model documentation card
│
├── sample_inputs.json         ← Example input records
├── sample_outputs.json        ← Example output records
│
└── app.py                     ← Optional FastAPI endpoint
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

```bash
# Create a .env file
echo "GEMINI_API_KEY=your_gemini_key_here" > .env
```

### 3. Train All Models

```bash
python train_classical_model.py
python train_cnn_model.py
python train_rnn_model.py
```

### 4. Run Inference

```bash
python hybrid_inference.py
# Output saved to sample_outputs.json
```

### 5. Start FastAPI Server 

```bash
uvicorn app:app --reload --port 8000

```

---

## Model Components

| Component | Algorithm | Input | Purpose |
|-----------|-----------|-------|---------|
| Classical ML | Random Forest (100 trees) | TF-IDF + tabular | Structured risk signal |
| CNN | 2-block Conv2D + Dense | 28×28 image | Image-based risk signal |
| RNN | Bidirectional LSTM | Tokenised text | Sequence/text risk signal |
| LLM | Google Gemini 1.5 Flash | Complaint + scores | Human-readable explanation |

## Hybrid Scoring

```
final_score = 0.40 × classical_score + 0.30 × cnn_score + 0.30 × rnn_score

final_score < 0.45        → Low risk   → Auto-reply
0.45 ≤ final_score < 0.75 → Medium risk → Manual review
final_score ≥ 0.75        → High risk  → Escalate immediately
```

---

## Sample Output

```json
{
  "customer_id": "C001",
  "model_outputs": {
    "classical_score": 0.7812,
    "cnn_score": 0.6950,
    "rnn_score": 0.8540,
    "final_score": 0.7760,
    "risk_level": "high"
  },
  "llm_explanation": {
    "summary": "Customer reports a broken product with no support response.",
    "recommended_action": "escalate_immediately",
    "reason": "High risk score driven by escalation language and repeated contact attempts.",
    "human_review_required": true,
    "risk_notes": ["Product damage mentioned", "Support non-responsiveness indicated"]
  },
  "decision": {
    "final_action": "escalate_immediately",
    "human_required": true,
    "model_disagree": false,
    "score_range": 0.159
  }
}
```

---

## Key Design Decisions

1. **LLM explains, not decides.** The LLM summarises and explains risk but cannot override model scores or approve auto-replies for high-risk cases.
2. **Humans stay in the loop.** All medium and high-risk cases require human review.
3. **Graceful degradation.** If Gemini is unavailable, a rule-based fallback produces the same JSON format.
4. **Model disagreement detection.** If the three model scores disagree by > 0.35, the case is automatically escalated to human review.

## Developed by

Krishnendu Bhowmick