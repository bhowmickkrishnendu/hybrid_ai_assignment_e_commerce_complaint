# Decision Policy - Complaint Escalation System

## 1. Risk Levels and Primary Actions

| Risk Level | Score Range     | Primary Action        |
|------------|-----------------|------------------------|
| Low        | < 0.45          | Auto-reply allowed     |
| Medium     | 0.45 – 0.74     | Manual review required |
| High       | ≥ 0.75          | Escalate immediately   |

---

## 2. Auto-Reply (Low Risk)

**Trigger:** final_score < 0.45 AND no model disagreement AND LLM does not flag review.

**Action:** The system sends a templated, empathetic auto-reply.

**Example reply:**
> "Thank you for contacting us. We have logged your complaint and our team will review it within 2 business days. Your order reference is #XXXX."

**Not allowed when:**
- The complaint mentions injury, safety, or legal action (even if score is low).
- Model scores disagree by more than 0.35.
- Image quality is flagged as poor.

---

## 3. Manual Review (Medium Risk)

**Trigger:** 0.45 ≤ final_score < 0.75 OR model disagreement is detected.

**Action:** The complaint is queued for a human support agent.

**Agent receives:**
- Full complaint text.
- Model scores and risk level.
- LLM-generated summary and risk notes.
- Recommended action.

**SLA:** Respond within 4 business hours.

---

## 4. Escalate Immediately (High Risk)

**Trigger:** final_score ≥ 0.75 OR LLM flags escalation signals.

**Action:** The complaint is escalated to a senior support manager.

**Escalation signals include:**
- Mentions of legal action, chargebacks, or consumer protection agencies.
- Safety or injury language.
- 5+ previous complaints in the account history.
- LLM risk notes flag two or more escalation signals.

**SLA:** Respond within 1 business hour.

---

## 5. Human Review Override Rules

Human review is **always required** when any of the following are true:

| Condition                                      | Override Action       |
|------------------------------------------------|-----------------------|
| Risk level is medium or high                   | Manual review / escalate |
| Model scores disagree by > 0.35                | Escalate to human     |
| Complaint input is unclear or very short (< 5 words) | Manual review   |
| Image quality is flagged as poor or missing    | Manual review         |
| LLM sets `human_review_required: true`         | Manual review minimum |
| Customer has prior unresolved complaints        | Escalate minimum       |

---

## 6. LLM Role Boundaries

The LLM:
✅ May summarise the complaint.  
✅ May explain the risk score in plain language.  
✅ May suggest an action consistent with the model risk level.  
✅ May flag additional risk signals not captured by structured features.  

The LLM must NOT:
❌ Override or change the model-assigned risk level.  
❌ Make the final decision for high-risk cases unilaterally.  
❌ Approve auto-replies for complaints it flags as needing review.  

---

## 7. Monitoring and Review

- Weekly: Review false negatives (high-risk complaints that were auto-replied).
- Monthly: Recalibrate scoring weights based on real escalation outcomes.
- Quarterly: Retrain all models on new complaint data.
- Alert: If disagreement rate > 20% over any 7-day window, trigger model review.
