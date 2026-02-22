from pydantic import BaseModel, Field
from typing import Dict


class RiskRequest(BaseModel):
    features: Dict[str, float]


class RiskResponse(BaseModel):
    risk_proba: float
    alert: int
    threshold: float
    model_name: str
