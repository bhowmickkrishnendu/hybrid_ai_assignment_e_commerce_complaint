# Mini Model Card — Hybrid AI Complaint Escalation System

## Model Details

| Field             | Value                                                  |
|-------------------|--------------------------------------------------------|
| Model name        | Hybrid Complaint Escalation System v1.0                |
| Model type        | Hybrid (Classical ML + CNN + RNN + LLM)                |
| Developed by      | AI/MLOps Engineering Assignment                        |
| Date              | May 2026                                               |
| Framework         | scikit-learn, TensorFlow/Keras, Google Gemini          |
| License           | Educational use only                                   |

---

## Intended Use

**Primary use case:**  
Predict the escalation risk of customer complaints for an e-commerce company and recommend a business action.

**Intended users:**  
Customer support team, support automation platform, operations managers.

**Out-of-scope uses:**  
- Legal or regulatory decision-making.
- High-stakes decisions without human review.
- Languages other than English (model trained on English text only).

---

## Model Components

| Component       | Architecture              | Input                         | Output              |
|-----------------|---------------------------|-------------------------------|---------------------|
| Classical ML    | Random Forest (100 trees) | TF-IDF text + tabular features | Risk probability     |
| CNN             | 2-block Conv2D + Dense    | Product image (28×28)         | Category → risk score|
| RNN             | Bidirectional LSTM        | Complaint text (tokenised)    | Risk probability     |
| LLM             | Google Gemini 1.5 Flash   | Complaint + scores            | JSON explanation     |

---

## Inputs

| Input Field              | Type    | Required | Notes                             |
|--------------------------|---------|----------|-----------------------------------|
| complaint_text           | string  | Yes      | Raw complaint text                |
| previous_complaints      | integer | Yes      | Number of prior complaints        |
| order_value              | float   | Yes      | Order value in USD                |
| customer_tenure_years    | float   | Yes      | How long customer has been active |
| account_type             | string  | Yes      | "standard" or "premium"           |
| image_category           | string  | No       | Product category or image label   |

---

## Outputs

| Output Field          | Type    | Values                                            |
|-----------------------|---------|---------------------------------------------------|
| classical_score       | float   | 0.0 – 1.0                                         |
| cnn_score             | float   | 0.0 – 1.0                                         |
| rnn_score             | float   | 0.0 – 1.0                                         |
| final_score           | float   | 0.0 – 1.0                                         |
| risk_level            | string  | "low" / "medium" / "high"                         |
| recommended_action    | string  | "auto_reply" / "manual_review" / "escalate_immediately" |
| human_review_required | boolean | true / false                                      |
| risk_notes            | list    | Short text signals flagged by LLM                |

---

## Performance Metrics

| Component    | Accuracy | F1 (weighted) |
|--------------|----------|---------------|
| Classical ML | ~0.81    | ~0.80         |
| CNN          | ~0.90    | ~0.89         |
| RNN/LSTM     | ~0.84    | ~0.83         |
| Hybrid       | ~0.83    | ~0.82         |
| LLM (JSON valid) | 100% | N/A          |
| LLM (faithful)   | 100% | N/A          |

---

## Risk Thresholds

| Score        | Risk Level | Action                 |
|--------------|------------|------------------------|
| < 0.45       | Low        | Auto-reply             |
| 0.45 – 0.74  | Medium     | Manual review          |
| ≥ 0.75       | High       | Escalate immediately   |

---

## Known Failure Modes

| Failure Mode                         | Impact         | Mitigation                          |
|--------------------------------------|----------------|--------------------------------------|
| Short or ambiguous complaints        | Misclassification | Minimum length check + human review |
| Polite but high-risk language        | Under-prediction  | LLM secondary check                 |
| Poor image quality                   | CNN score noise   | Confidence threshold + flag         |
| Model disagreement (score range > 0.35) | Inconsistency | Auto-escalate to human review       |
| No Gemini API key                    | No LLM output     | Rule-based fallback                 |
| Non-English complaints               | Unpredictable  | Language detection + reject/flag    |

---

## Human Review Rules

Human review is mandatory when:
- Risk level is medium or high.
- Model scores disagree by more than 0.35.
- Complaint contains safety or injury language.
- Customer has 5+ previous complaints.
- LLM sets `human_review_required: true`.

---

## Deployment Assumptions

- Models are pre-trained and saved as `.pkl` / `.keras` files.
- Gemini API key is stored in environment variables (not in code).
- The system is stateless — each request is independent.
- CPU inference is sufficient; GPU is not required for serving.
- Estimated latency per request: < 2 seconds (without LLM), < 5 seconds (with LLM).

---

## Monitoring Suggestions

- Log `final_score`, `risk_level`, and `recommended_action` for every request.
- Track false negatives (escalations that were auto-replied) weekly.
- Monitor disagreement rate (score_range > 0.35) daily.
- Set up alerts if high-risk predictions exceed 30% of total in any hour.
- Retrain models quarterly with real labelled complaint data.
- Monitor LLM valid JSON rate; alert if it drops below 95%.
