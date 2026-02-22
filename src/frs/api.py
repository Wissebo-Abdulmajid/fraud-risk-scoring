from __future__ import annotations

import io
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from functools import lru_cache

from frs.cli import score_cmd
from frs.explainability import explain_cmd as explain_engine
from frs.reporting import build_report_from_run
from frs.calibration import apply_calibrator, predict_proba


# =========================================================
# Logging
# =========================================================

logger = logging.getLogger("frs.api")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)


# =========================================================
# App
# =========================================================

app = FastAPI(
    title="FRS — Fraud / Risk Scoring System",
    version="1.0.0",
    description="Production-grade Fraud / Risk Scoring API built on FRS run artifacts.",
)


# =========================================================
# Environment / Hardening
# =========================================================

API_KEY = os.getenv("FRS_API_KEY", "").strip()
RUNS_BASE = Path(os.getenv("FRS_RUNS_BASE", "/app/runs")).resolve()
REPORTS_BASE = Path(os.getenv("FRS_REPORTS_BASE", "/app/reports")).resolve()

MAX_UPLOAD_MB = int(os.getenv("FRS_MAX_UPLOAD_MB", "200"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024


# =========================================================
# Security
# =========================================================

def require_api_key(x_api_key: Optional[str] = Header(None)) -> None:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server misconfigured: FRS_API_KEY not set.")
    if not x_api_key or x_api_key.strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")


class MaxUploadSizeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl:
            try:
                if int(cl) > MAX_UPLOAD_BYTES:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": f"Upload too large. Max is {MAX_UPLOAD_MB} MB."},
                    )
            except Exception:
                pass
        return await call_next(request)


app.add_middleware(MaxUploadSizeMiddleware)


# =========================================================
# Models
# =========================================================

class ScoreEvent(BaseModel):
    features: dict[str, float | int | str | bool | None]


class ScoreBatchRequest(BaseModel):
    events: list[ScoreEvent] = Field(..., min_length=1)
    threshold: float | None = Field(None, ge=0.0, le=1.0)
    return_features: bool = False
    strict: bool = True


class ScoreOneResponse(BaseModel):
    risk_proba: float
    alert: int
    threshold: float
    model_name: str
    run_dir: str


class ScoreBatchResponse(BaseModel):
    run_dir: str
    model_name: str
    threshold: float
    n_events: int
    n_alerts: int
    alert_rate: float
    results: list[dict]


# =========================================================
# Bundle Cache
# =========================================================

@lru_cache(maxsize=8)
def _load_bundle_cached(run_dir_str: str):
    import joblib
    bundle_path = Path(run_dir_str) / "bundle.joblib"
    if not bundle_path.exists():
        raise HTTPException(status_code=404, detail="bundle.joblib not found.")
    return joblib.load(bundle_path)


# =========================================================
# Utilities
# =========================================================

def _resolve_run_dir(run_dir: str) -> Path:
    candidate = (RUNS_BASE / run_dir).resolve()

    try:
        candidate.relative_to(RUNS_BASE)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid run_dir (path traversal).")

    if not candidate.exists():
        raise HTTPException(status_code=404, detail="run_dir not found.")
    if not (candidate / "bundle.joblib").exists():
        raise HTTPException(status_code=400, detail="bundle.joblib missing in run_dir.")

    return candidate


def _read_csv_upload(file: UploadFile) -> pd.DataFrame:
    raw = file.file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty upload.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload too large.")
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}") from e


# =========================================================
# Health
# =========================================================

@app.get("/")
def root() -> dict[str, Any]:
    return {
        "service": "FRS — Fraud / Risk Scoring System",
        "status": "ok",
        "docs": "/docs",
        "health": "/health",
        "version": app.version,
        "auth": "x-api-key required for /score, /score-json, /score-batch-json, /explain, /report",
    }


# =========================================================
# v1 Endpoints
# =========================================================

@app.post("/v1/score-json", response_model=ScoreOneResponse, dependencies=[Depends(require_api_key)])
def score_json(payload: ScoreEvent, run_dir: str = Query(...), threshold: float | None = None):

    run_dir_p = _resolve_run_dir(run_dir)
    bundle = _load_bundle_cached(str(run_dir_p))

    pipeline = bundle["pipeline"]
    calibrator = bundle.get("calibrator")
    policy = bundle["policy"]
    feature_spec = bundle["feature_spec"]
    model_name = bundle.get("model_name", "unknown")

    numeric = list(feature_spec["numeric"])
    categorical = list(feature_spec["categorical"])
    required_cols = numeric + categorical

    missing = [c for c in required_cols if c not in payload.features]
    if missing:
        raise HTTPException(status_code=400, detail={"missing": missing})

    df = pd.DataFrame([payload.features])[required_cols]

    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    proba = predict_proba(pipeline, df)
    if calibrator:
        proba = apply_calibrator(calibrator, proba)

    proba = float(np.asarray(proba)[0])
    thr = float(threshold) if threshold is not None else float(policy["threshold"])

    return ScoreOneResponse(
        risk_proba=proba,
        alert=int(proba >= thr),
        threshold=thr,
        model_name=model_name,
        run_dir=str(run_dir_p),
    )


@app.post("/v1/score-csv", dependencies=[Depends(require_api_key)])
async def score_csv(
    run_dir: str = Query(...),
    file: UploadFile = File(...),
    threshold: float | None = None,
):
    run_dir_p = _resolve_run_dir(run_dir)
    df = _read_csv_upload(file)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_path = td / "input.csv"
        out_path = td / "scored.csv"
        summary_path = td / "summary.json"

        df.to_csv(in_path, index=False)

        score_cmd(
            run_dir=run_dir_p,
            input_path=in_path,
            output_path=out_path,
            threshold=threshold,
            summary_path=summary_path,
        )

        scored_bytes = out_path.read_bytes()

    return StreamingResponse(
        io.BytesIO(scored_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=scored.csv"},
    )


@app.post("/v1/explain", dependencies=[Depends(require_api_key)])
async def explain_csv(
    run_dir: str = Query(...),
    file: UploadFile = File(...),
    top_k: int = 10,
    max_rows: int = 2000,
    method: str = "auto",
):
    run_dir_p = _resolve_run_dir(run_dir)
    df = _read_csv_upload(file)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        in_path = td / "scored.csv"
        df.to_csv(in_path, index=False)

        out_csv = run_dir_p / "explain" / "explanations.csv"
        out_json = run_dir_p / "explain" / "summary.json"
        figs_dir = run_dir_p / "explain" / "figures"

        res = explain_engine(
            run_dir=run_dir_p,
            input_path=in_path,
            output_path=out_csv,
            summary_path=out_json,
            figures_dir=figs_dir,
            top_k=top_k,
            max_rows=max_rows,
            method=method,
        )

    return {
        "explanations_csv": str(res.out_csv),
        "summary_json": str(res.out_json),
        "figures_dir": str(res.figures_dir),
    }


@app.post("/v1/report", dependencies=[Depends(require_api_key)])
def report_from_run(run_dir: str = Query(...), out_dir: str = Query(...)):

    run_dir_p = _resolve_run_dir(run_dir)
    out_dir_p = (REPORTS_BASE / out_dir).resolve()

    try:
        out_dir_p.relative_to(REPORTS_BASE)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid out_dir.")

    rp = build_report_from_run(run_dir=run_dir_p, out_dir=out_dir_p)

    return {
        "report_md": str(rp.report_md),
        "report_json": str(rp.report_json),
        "figures_dir": str(rp.figures_dir),
    }