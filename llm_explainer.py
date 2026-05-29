"""
llm_explainer.py
----------------
Uses Google Gemini to generate a structured JSON explanation for each
complaint. The LLM explains the risk; it does NOT override model scores.

Setup:
  1. Create a .env file:        GEMINI_API_KEY=your_key_here
  2. Or export the variable:    export GEMINI_API_KEY=your_key_here

If no API key is set, the module falls back to a deterministic rule-based
explanation so the pipeline keeps working offline.
"""

import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# ── CONFIGURE ─────────────────────────────────────────────────────────────────

_GEMINI_CONFIGURED = False

def _configure_gemini():
    global _GEMINI_CONFIGURED
    if _GEMINI_CONFIGURED:
        return True
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return False
    genai.configure(api_key=api_key)
    _GEMINI_CONFIGURED = True
    return True


# ── PROMPT TEMPLATE ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI assistant helping an e-commerce customer support team 
understand complaint escalation risk decisions.

You will receive model scores and complaint details. Your job is to explain the 
decision in plain business language and produce a structured JSON output.

RULES:
- You must NOT override the risk_level determined by the model scores.
- You must NOT make final high-risk decisions on your own.
- Return ONLY valid JSON — no markdown, no preamble, no explanation outside the JSON.
"""

def _build_user_prompt(complaint_text, model_outputs):
    return f"""Complaint text: "{complaint_text}"

Model scores:
- Classical ML score : {model_outputs['classical_score']}
- CNN score          : {model_outputs['cnn_score']}
- RNN score          : {model_outputs['rnn_score']}
- Final hybrid score : {model_outputs['final_score']}
- Risk level         : {model_outputs['risk_level']}

Return a JSON object with exactly these keys:
{{
  "summary": "One sentence summary of the complaint",
  "recommended_action": "auto_reply | manual_review | escalate_immediately",
  "reason": "One sentence explaining the risk level",
  "human_review_required": true | false,
  "risk_notes": ["note 1", "note 2"]
}}

Rules for recommended_action:
- risk_level is "low"    → "auto_reply"
- risk_level is "medium" → "manual_review"
- risk_level is "high"   → "escalate_immediately"

Return ONLY the JSON. No markdown. No backticks.
"""


# ── RULE-BASED FALLBACK (no API key needed) ───────────────────────────────────

def _rule_based_explanation(complaint_text, model_outputs):
    """Deterministic fallback explanation when Gemini is unavailable."""
    risk = model_outputs["risk_level"]
    score = model_outputs["final_score"]

    action_map = {
        "low":    "auto_reply",
        "medium": "manual_review",
        "high":   "escalate_immediately",
    }
    review_map = {"low": False, "medium": True, "high": True}

    notes = []
    lower = complaint_text.lower()
    if any(w in lower for w in ["broken", "damaged", "defective", "shattered"]):
        notes.append("Product damage mentioned")
    if any(w in lower for w in ["support", "nobody", "ignores", "no reply", "no response"]):
        notes.append("Support non-responsiveness indicated")
    if any(w in lower for w in ["legal", "refund", "chargeback", "fraud", "escalate"]):
        notes.append("Escalation language detected")
    if any(w in lower for w in ["angry", "furious", "worst", "unacceptable"]):
        notes.append("High negative sentiment detected")
    if not notes:
        notes.append("Standard complaint pattern")

    short_text = complaint_text[:60] + "..." if len(complaint_text) > 60 else complaint_text

    return {
        "summary": f"Customer complaint: {short_text}",
        "recommended_action": action_map.get(risk, "manual_review"),
        "reason": f"Final hybrid score is {score:.2f}, classified as {risk} risk.",
        "human_review_required": review_map.get(risk, True),
        "risk_notes": notes,
    }


# ── MAIN FUNCTION ─────────────────────────────────────────────────────────────

def generate_explanation(complaint_text: str, model_outputs: dict) -> dict:
    """
    Generate LLM explanation for a complaint.

    Parameters
    ----------
    complaint_text : str
        The raw customer complaint.
    model_outputs : dict
        Keys: classical_score, cnn_score, rnn_score, final_score, risk_level

    Returns
    -------
    dict with keys: summary, recommended_action, reason,
                    human_review_required, risk_notes
    """
    use_gemini = _configure_gemini()

    if use_gemini:
        try:
            model = genai.GenerativeModel(
                model_name="gemini-3.5-flash",
                system_instruction=SYSTEM_PROMPT,
            )
            user_prompt = _build_user_prompt(complaint_text, model_outputs)
            response = model.generate_content(user_prompt)
            raw = response.text.strip()

            # Strip markdown fences if present
            raw = re.sub(r"^```json\s*", "", raw)
            raw = re.sub(r"\s*```$",    "", raw)

            result = json.loads(raw)
            result["source"] = "gemini"
            return result

        except Exception as e:
            print(f"⚠️  Gemini error ({e}), using rule-based fallback.")

    result = _rule_based_explanation(complaint_text, model_outputs)
    result["source"] = "rule_based"
    return result


# ── BATCH EVALUATION ──────────────────────────────────────────────────────────

def evaluate_llm(test_cases: list) -> dict:
    """
    Measure LLM output quality on a list of test cases.

    Parameters
    ----------
    test_cases : list of dicts, each with keys:
        complaint_text, model_outputs, expected_action (optional)

    Returns
    -------
    dict with valid_json_rate, action_accuracy, faithfulness_score
    """
    total        = len(test_cases)
    valid_json   = 0
    action_match = 0
    faithful     = 0

    REQUIRED_KEYS = {"summary", "recommended_action", "reason",
                     "human_review_required", "risk_notes"}

    for case in test_cases:
        result = generate_explanation(case["complaint_text"], case["model_outputs"])

        # Valid JSON check (has all required keys)
        if REQUIRED_KEYS.issubset(result.keys()):
            valid_json += 1

        # Action faithfulness: action must match risk_level → action rule
        risk = case["model_outputs"]["risk_level"]
        expected_action = {
            "low": "auto_reply", "medium": "manual_review", "high": "escalate_immediately"
        }.get(risk, "manual_review")

        if result.get("recommended_action") == expected_action:
            action_match += 1
            faithful += 1  # count as faithful if action matches

    metrics = {
        "total_cases":         total,
        "valid_json_count":    valid_json,
        "valid_json_rate":     round(valid_json / total, 4) if total else 0,
        "action_match_count":  action_match,
        "faithfulness_score":  round(faithful / total, 4) if total else 0,
    }
    print(f"\n── LLM Evaluation ──")
    print(f"Valid JSON rate      : {metrics['valid_json_rate']:.2%}")
    print(f"Faithfulness score   : {metrics['faithfulness_score']:.2%}")
    return metrics


# ── STANDALONE TEST ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_complaint = (
        "My product arrived broken and I have contacted support three times. "
        "Nobody has replied. This is absolutely unacceptable."
    )
    test_outputs = {
        "classical_score": 0.71,
        "cnn_score":       0.65,
        "rnn_score":       0.82,
        "final_score":     0.74,
        "risk_level":      "medium",
    }
    result = generate_explanation(test_complaint, test_outputs)
    print(json.dumps(result, indent=2))

    # Batch evaluation demo
    test_cases = [
        {"complaint_text": test_complaint,            "model_outputs": test_outputs},
        {"complaint_text": "Delivery was slightly late.",
         "model_outputs": {"classical_score":0.2,"cnn_score":0.15,"rnn_score":0.2,
                           "final_score":0.18,"risk_level":"low"}},
        {"complaint_text": "I will take legal action if this is not resolved today.",
         "model_outputs": {"classical_score":0.9,"cnn_score":0.85,"rnn_score":0.92,
                           "final_score":0.89,"risk_level":"high"}},
    ]
    llm_metrics = evaluate_llm(test_cases)
    with open("llm_metrics.json", "w") as f:
        json.dump(llm_metrics, f, indent=2)
    print("\n✅  Saved: llm_metrics.json")
