"""
app.py
------
FastAPI endpoint for the hybrid AI complaint escalation system.

Run locally:
    uvicorn app:app --reload --port 8000

Test with curl:
    curl -X POST http://localhost:8000/predict \
         -H "Content-Type: application/json" \
         -d '{
               "customer_id": "C001",
               "complaint_text": "Product broke on day one. Nobody answers my emails.",
               "customer_tenure_years": 0.5,
               "previous_complaints": 3,
               "order_value": 149.99,
               "account_type": "standard",
               "image_category": "damaged_product"
             }'
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import json

from hybrid_inference import HybridModel, apply_decision_policy

app = FastAPI(
    title="Hybrid AI Complaint Escalation API",
    description="Predicts complaint escalation risk using Classical ML + CNN + RNN + LLM",
    version="1.0.0",
)

# Load models once at startup (not per request)
_model: Optional[HybridModel] = None

@app.on_event("startup")
def load_models():
    global _model
    try:
        _model = HybridModel()
    except Exception as e:
        print(f"⚠️  Model loading failed: {e}")
        print("Run train_classical_model.py, train_cnn_model.py, train_rnn_model.py first.")


# ── SCHEMAS ───────────────────────────────────────────────────────────────────

class ComplaintRequest(BaseModel):
    customer_id:           str = "C001"
    complaint_text:        str
    customer_tenure_years: float = 1.0
    previous_complaints:   int   = 0
    order_value:           float = 50.0
    account_type:          str   = "standard"
    image_category:        str   = "default"

class BatchRequest(BaseModel):
    complaints: list[ComplaintRequest]


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "service": "Hybrid AI Complaint Escalation API"}

@app.get("/health")
def health():
    model_ready = _model is not None
    return {"status": "healthy" if model_ready else "models_not_loaded",
            "model_ready": model_ready}

@app.post("/predict")
def predict(request: ComplaintRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Models not loaded. Train models first.")
    complaint = request.dict()
    result    = _model.predict(complaint)
    result    = apply_decision_policy(result)
    return result

@app.post("/predict/batch")
def predict_batch(request: BatchRequest):
    if _model is None:
        raise HTTPException(status_code=503, detail="Models not loaded.")
    results = []
    for c in request.complaints:
        result = _model.predict(c.dict())
        result = apply_decision_policy(result)
        results.append(result)
    return {"count": len(results), "results": results}


# ── SAMPLE REQUEST / RESPONSE (for documentation) ─────────────────────────────

@app.get("/docs/sample-request")
def sample_request():
    return {
        "customer_id": "C001",
        "complaint_text": "Product broke on day one. Nobody answers my emails.",
        "customer_tenure_years": 0.5,
        "previous_complaints": 3,
        "order_value": 149.99,
        "account_type": "standard",
        "image_category": "damaged_product",
    }

@app.get("/docs/sample-response")
def sample_response():
    return {
        "customer_id": "C001",
        "model_outputs": {
            "classical_score": 0.71,
            "cnn_score": 0.65,
            "rnn_score": 0.82,
            "final_score": 0.74,
            "risk_level": "medium",
        },
        "llm_explanation": {
            "summary": "Customer reports receiving a broken product and delayed support.",
            "recommended_action": "manual_review",
            "reason": "The complaint shows frustration and the final risk score is medium.",
            "human_review_required": True,
            "risk_notes": [
                "Broken product mentioned",
                "Support delay mentioned",
                "Previous complaints increase risk",
            ],
        },
        "decision": {
            "final_action": "manual_review",
            "human_required": True,
            "model_disagree": False,
            "score_range": 0.17,
        },
    }
