# backend/main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np, pandas as pd
from typing import List, Dict
from .io_utils import read_any_table
from .models import SimParams, SimResult
from .core import (zscore_cols, map_expr_to_eps_J, simulate_enaqt_curve,
                   index_layout, hamiltonian_from_params, line_topology)

app = FastAPI(title="Quantum Diagnostics API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health")
def health():
    return {"status":"ok"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    b = await file.read()
    df = read_any_table(b, file.filename)
    # minimal normalization
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    # require sample_id + label (label optional but helpful)
    if "sample_id" not in df.columns:
        df.insert(0,"sample_id", df.index.astype(str))
    return {"columns": list(df.columns), "rows": min(len(df), 5)}

@app.post("/simulate")
async def simulate(file: UploadFile = File(...), params: str = Form(...)):
    """
    Accepts any tabular file; expects columns: sample_id, label (optional), + gene columns.
    """
    p = SimParams.model_validate_json(params)
    b = await file.read()
    df = read_any_table(b, file.filename)
    if "sample_id" not in df.columns:
        df.insert(0,"sample_id", df.index.astype(str))
    gene_names = [c for c in df.columns if c not in ("sample_id","label")]
    if len(gene_names) < 2:
        return {"error":"Need >=2 gene columns"}
    Z, mu, sd = zscore_cols(df, gene_names)
    zdf = pd.DataFrame(Z, columns=gene_names)
    work = pd.concat([df[["sample_id"]], df["label"]] if "label" in df.columns else [df[["sample_id"]]], axis=1)
    work = pd.concat([work, zdf], axis=1)

    gamma_grid = np.linspace(p.gamma_min, p.gamma_max, p.gamma_steps)
    results: List[SimResult] = []
    for _, row in work.iterrows():
        row_dict = row.to_dict()
        eps, J_edges = map_expr_to_eps_J(row_dict, gene_names[:p.n_core])
        etes, i_star = simulate_enaqt_curve(
            n_core=min(p.n_core, len(gene_names)),
            eps=eps,
            J_edges=J_edges,
            gamma_grid=gamma_grid,
            params=p.model_dump()
        )
        ete_star = float(etes[i_star])
        gamma_star = float(gamma_grid[i_star])
        # quick QLS proxy (you can swap with your fuller tau_c calc)
        qls_star = 0.6*ete_star + 0.4*(i_star/(len(gamma_grid)-1))
        results.append(SimResult(
            sample_id=str(row_dict["sample_id"]),
            label=str(row_dict.get("label")) if "label" in row_dict else None,
            ETE_peak=ete_star, gamma_star=gamma_star, QLS_star=qls_star
        ))
    return {"metrics":[r.model_dump() for r in results]}
