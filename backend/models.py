# backend/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class SimParams(BaseModel):
    gamma_min: float = 0.0
    gamma_max: float = 0.05
    gamma_steps: int = 21
    T: float = 150.0
    dt: float = 0.5
    k_sink: float = 0.1
    k_loss: float = 0.01
    static_sigma: float = 0.03
    seed: int = 42
    n_core: int = 7

class SimResult(BaseModel):
    sample_id: str
    label: Optional[str] = None
    ETE_peak: float
    gamma_star: float
    QLS_star: float

class EdgeSensReq(BaseModel):
    frac: float = 0.10

class TSNEReq(BaseModel):
    feat_cols: List[str] = ["ETE_peak","gamma_star","QLS_star"]
