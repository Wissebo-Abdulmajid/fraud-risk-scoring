from __future__ import annotations

from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from api.schemas import RiskRequest, RiskResponse
from api.model_loader import RiskModel

# ---- Load model once at startup
BUNDLE_PATH = Path("runs/2026-02-17_220041/bundle.joblib")  # change per deployment
model = RiskModel(BUNDLE_PATH)

app = FastAPI(
    title="Fraud Risk Scoring API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/version")
def version():
    return {"model_name": model.model_name}


@app.post("/score", response_model=RiskResponse)
def score(req: RiskRequest):
    try:
        result = model.predict(req.features)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
