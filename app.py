"""
Quantum Energy Mapping of Disease (QEMD)
Streamlit Dashboard — Production Build
Author: Jachin Thilak (2025)
"""

import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, plotly.graph_objects as go
import streamlit as st
from qemd.engine import (index_layout, hamiltonian_from_params, collapse_ops,
                         evolve_with_propagator, ete_final_sink, edge_sensitivity)
from qemd.io import (load_topology, load_expr_parquet, zscore_cols, map_expr_to_eps_J)

# --- Streamlit configuration ---
st.set_page_config(page_title="Quantum Energy Mapping of Disease", layout="wide")
st.title("🧬 Quantum Energy Mapping of Disease (QEMD)")
st.caption("Mechanistic AI-biology platform modeling mitochondrial energy transport across disease states.")

# --- Default paths ---
BASE_DIR = os.getenv("QEMD_BASE", "./data")
default_topo = os.path.join(BASE_DIR, "ETC_topology.json")
default_expr = os.path.join(BASE_DIR, "GSE14520_ETC_expression.parquet")

# --- Inputs ---
st.sidebar.header("⚙️ Simulation Controls")
topo_path = st.sidebar.text_input("Topology JSON path", default_topo)
expr_path = st.sidebar.text_input("Expression Parquet path", default_expr)

# --- Load data ---
try:
    nodes, base_edges = load_topology(topo_path)
    expr_df, gene_names = load_expr_parquet(expr_path)
    st.success(f"✅ Loaded {len(nodes)} nodes, {len(base_edges)} edges, "
               f"{expr_df.shape[0]} samples × {len(gene_names)} genes.")
    ok = True
except Exception as e:
    st.error(f"Data load failed: {e}")
    ok = False

if ok:
    sample_id = st.selectbox("Select Sample", expr_df["sample_id"].astype(str), index=0)
    col1, col2, col3, col4 = st.columns(4)
    with col1: gamma_min = st.number_input("γ min", 0.0, 0.2, 0.00, 0.005)
    with col2: gamma_max = st.number_input("γ max", 0.0, 0.2, 0.05, 0.005)
    with col3: k_sink = st.number_input("k_sink", 0.0, 1.0, 0.10, 0.01)
    with col4: k_loss = st.number_input("k_loss", 0.0, 1.0, 0.01, 0.01)

    # --- Map expression row to Hamiltonian ---
    z_df = zscore_cols(expr_df, gene_names)
    row = pd.concat([expr_df[["sample_id", "label"]], z_df], axis=1).set_index("sample_id").loc[str(sample_id)]
    eps, J_edges = map_expr_to_eps_J(row, base_edges, gene_names)

    n_core = len(gene_names)
    sink_idx, loss_idx, D = index_layout(n_core, True, True)
    H = hamiltonian_from_params(n_core, eps, J_edges, static_sigma=0.07, seed=42, add_sink=True, add_loss=True)

    gamma_grid = np.linspace(gamma_min, gamma_max, 21)
    psi0 = np.zeros((D, 1), dtype=np.complex128); psi0[0, 0] = 1.0
    rho0 = psi0 @ psi0.conj().T

    # --- Compute ETE curve ---
    etes = []
    for g in gamma_grid:
        Ls = collapse_ops(n_core, D, g, sink_idx, n_core - 1, k_sink, k_loss)
        _, rho_last = evolve_with_propagator(H, Ls, rho0, T=200.0, dt=0.5, sink_idx=sink_idx, early_stop=True)
        etes.append(ete_final_sink(rho_last, sink_idx))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gamma_grid, y=etes, mode="lines+markers", name="ETE"))
    fig.update_layout(title="ETE vs γ (Dephasing Rate)", xaxis_title="γ", yaxis_title="ETE (sink population)")
    st.plotly_chart(fig, use_container_width=True)

    # --- Edge sensitivity map ---
    with st.expander("🔍 Edge Sensitivity (+10% coupling)"):
        sens = edge_sensitivity(
            hamiltonian_from_params,
            collapse_ops,
            dict(n_core=n_core, eps=eps, J_edges=J_edges,
                 static_sigma=0.07, seed=42, k_sink=k_sink,
                 k_loss=k_loss, T=200.0, dt=0.5, gamma_grid=gamma_grid),
            edges=J_edges, frac=0.10
        )
        sens_df = pd.DataFrame(
            [(idx, J_edges[idx][0], J_edges[idx][1], float(delta)) for idx, delta in sens.items()],
            columns=["edge_idx", "u", "v", "ΔETE"]
        )
        st.dataframe(sens_df.sort_values("ΔETE", ascending=False), use_container_width=True)
