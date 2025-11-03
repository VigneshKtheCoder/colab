# app.py
# Streamlit app converted from Colab notebook for ENAQT / ETC pipeline
# Usage: `streamlit run app.py`

import streamlit as st
st.set_page_config(page_title="ENAQT / ETC Simulator", layout="wide")

import os
import json
import time
import re
import math
import numpy as np
import numpy.linalg as npl
import scipy.linalg as spla
from scipy.integrate import solve_ivp
from scipy.special import expit
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Utility helpers
# -------------------------
np.set_printoptions(precision=4, suppress=True)
RNG = np.random.default_rng(42)

def cplx(a): return np.asarray(a, dtype=np.complex128)
def dagger(M): return M.conj().T
def is_hermitian(M, tol=1e-9): return npl.norm(M - M.conj().T) <= tol
def psd_eig_min(M):
    ev = npl.eigvalsh((M + M.conj().T) / 2)
    return float(ev.min().real)

def vec(rho): return rho.reshape((-1,), order='F')
def unvec(v, dim): return v.reshape((dim, dim), order='F')

# -------------------------
# Liouvillian builders
# -------------------------
def liouvillian(H, collapses):
    H = cplx(H)
    d = H.shape[0]
    I = np.eye(d, dtype=np.complex128)
    LH = -1j * (np.kron(I, H) - np.kron(H.T, I))
    LD = np.zeros((d*d, d*d), dtype=np.complex128)
    for Lk in collapses:
        Lk = cplx(Lk)
        LdL = Lk.conj().T @ Lk
        term1 = np.kron(Lk, Lk.conj())
        term2 = 0.5 * (np.kron(np.eye(d), LdL.T) + np.kron(LdL, np.eye(d)))
        LD += (term1 - term2)
    return LH + LD

def liouvillian_fast(H, Ls):
    # alias for performance (same formula)
    return liouvillian(H, Ls)

# -------------------------
# Topology & Hamiltonian
# -------------------------
def line_topology(n_sites):
    return [(i, i+1) for i in range(n_sites-1)]

def hamiltonian_from_params(n_core, eps, J_edges, static_sigma=0.0, seed=0, add_sink=True, add_loss=True):
    rng = np.random.default_rng(seed)
    eps = np.array(eps, dtype=float)
    if static_sigma > 0:
        eps = eps + rng.normal(0.0, static_sigma, size=eps.shape)
    d = n_core + (1 if add_sink else 0) + (1 if add_loss else 0)
    H = np.zeros((d, d), dtype=np.complex128)
    for i in range(n_core):
        H[i, i] = eps[i]
    for (i, j, Jij) in J_edges:
        H[i, j] = H[j, i] = Jij
    return H

def index_layout(n_core, have_sink=True, have_loss=True):
    sink_idx = n_core if have_sink else None
    loss_idx = n_core + 1 if have_sink and have_loss else (n_core if (not have_sink and have_loss) else None)
    d = n_core + (1 if have_sink else 0) + (1 if have_loss else 0)
    return sink_idx, loss_idx, d

def collapse_ops(n_core, d, gamma, sink_idx, sink_target_idx, k_sink, k_loss):
    cols = []
    if gamma > 0:
        for j in range(n_core):
            Lj = np.zeros((d, d), dtype=np.complex128)
            Lj[j, j] = np.sqrt(gamma)
            cols.append(Lj)
    if sink_idx is not None and sink_target_idx is not None and k_sink > 0:
        Ls = np.zeros((d, d), dtype=np.complex128)
        Ls[sink_idx, sink_target_idx] = np.sqrt(k_sink)
        cols.append(Ls)
    if k_loss > 0:
        loss_idx = d-1
        for j in range(n_core):
            Lj = np.zeros((d, d), dtype=np.complex128)
            Lj[loss_idx, j] = np.sqrt(k_loss)
            cols.append(Lj)
    return cols

# -------------------------
# Integrators & metrics
# -------------------------
def simulate_dynamics(H, collapses, rho0, T=150.0, dt=0.5, method='BDF', rtol=1e-7, atol=1e-9):
    H = cplx(H)
    d = H.shape[0]
    L = liouvillian(H, collapses)
    def dyn(t, v): return (L @ v)
    t_eval = np.arange(0.0, T + 1e-12, dt)
    sol = solve_ivp(dyn, (0.0, T), vec(rho0), method=method, t_eval=t_eval, rtol=rtol, atol=atol, max_step=dt)
    if not sol.success:
        raise RuntimeError(f"Integrator failed: {sol.message}")
    rhos = np.stack([unvec(v, d) for v in sol.y.T], axis=0)
    return sol.t, rhos

def ete_from_sink_population(rhos, sink_idx):
    if sink_idx is None:
        return np.nan
    return float(np.real(rhos[-1][sink_idx, sink_idx]))

def coherence_lifetime(rhos, core_slice, threshold=np.exp(-1)):
    norms = []
    for rho in rhos:
        core = rho[np.ix_(core_slice, core_slice)]
        off_diag = core - np.diag(np.diag(core))
        norms.append(npl.norm(off_diag, 'fro'))
    norms = np.array(norms)
    if norms.size == 0 or norms[0] <= 0:
        return 0.0
    target = norms[0] * threshold
    idx = np.where(norms <= target)[0]
    return float(idx[0]) if len(idx) else float(len(norms)-1)

def qls_composite(ete, tau_c, tau_max):
    ete_n = np.clip(ete, 0.0, 1.0)
    tau_n = np.clip(tau_c / max(1.0, tau_max), 0.0, 1.0)
    return 0.6*ete_n + 0.4*tau_n

def psd_trace_checks(rho, tol=1e-8):
    tr = np.trace(rho)
    herm_ok = is_hermitian(rho)
    min_ev = psd_eig_min(rho)
    return float(tr.real), herm_ok, float(min_ev)

# -------------------------
# A simpler propagate-by-expm routine (faster for fixed dt)
# -------------------------
def evolve_with_propagator(H, Ls, rho0, T, dt, sink_idx):
    d = H.shape[0]
    L = liouvillian_fast(H, Ls)
    P = spla.expm(L * dt)
    steps = int(round(T / dt)) + 1
    tgrid = np.linspace(0.0, T, steps)
    v = rho0.reshape((-1,), order='F')
    sink_hist = []
    rho = None
    for k, _t in enumerate(tgrid):
        rho = v.reshape((d, d), order='F')
        sink_hist.append(float(np.real(rho[sink_idx, sink_idx])))
        v = P @ v
    return tgrid, rho

# -------------------------
# ENAQT benchmark (streamlined)
# -------------------------
def enaqt_benchmark_core(n_core=7, J=0.05, static_sigma=0.01, k_sink=0.1, k_loss=0.01, init_site=0, sink_target=None, gamma_grid=None, T=150.0, dt=0.5, seed=42, verbose=False):
    if gamma_grid is None:
        gamma_grid = np.linspace(0.0, 0.05, 21)
    edges = line_topology(n_core)
    eps = np.zeros(n_core, dtype=float)
    J_edges = [(i, j, J) for (i, j) in edges]
    sink_idx, loss_idx, d = index_layout(n_core, have_sink=True, have_loss=True)
    if sink_target is None:
        sink_target = n_core - 1
    H = hamiltonian_from_params(n_core=n_core, eps=eps, J_edges=J_edges, static_sigma=static_sigma, seed=seed, add_sink=True, add_loss=True)
    psi0 = np.zeros((d, 1), dtype=np.complex128); psi0[init_site, 0] = 1.0
    rho0 = psi0 @ psi0.conj().T
    etes, taus, qls = [], [], []
    last_times, last_rhos = None, None
    for g in gamma_grid:
        cols = collapse_ops(n_core=n_core, d=d, gamma=g, sink_idx=sink_idx, sink_target_idx=sink_target, k_sink=k_sink, k_loss=k_loss)
        t, rhos = simulate_dynamics(H, cols, rho0, T=T, dt=dt, method='BDF')
        ete = ete_from_sink_population(rhos, sink_idx)
        tau_c = coherence_lifetime(rhos, core_slice=list(range(n_core)))
        qls_val = qls_composite(ete, tau_c, tau_max=len(t))
        tr, herm_ok, min_ev = psd_trace_checks(rhos[-1])
        if not herm_ok or min_ev < -1e-6 or abs(tr.real - 1.0) > 5e-3:
            raise AssertionError(f"ρ(T) sanity check failed: trace={tr:.4f}, herm={herm_ok}, min_ev={min_ev:.3e}")
        etes.append(ete); taus.append(tau_c); qls.append(qls_val)
        last_times, last_rhos = t, rhos
    etes = np.array(etes); taus = np.array(taus); qls = np.array(qls)
    return gamma_grid, etes, taus, qls, last_times, last_rhos, (sink_idx, loss_idx, d)

# -------------------------
# GEOparse optional fetcher (attempts to import; environment may block)
# -------------------------
def try_fetch_gse14520(destdir="."):
    try:
        import GEOparse
    except Exception as e:
        raise RuntimeError("GEOparse not available in environment. Install via requirements or upload the precomputed file.") from e
    gse = GEOparse.get_GEO("GSE14520", destdir=destdir, how="quick")
    return gse

# -------------------------
# Expression mapping helpers (from Colab notebook)
# -------------------------
def extract_probe2sym(gpl):
    df = gpl.table.copy()
    df.columns = [str(c) for c in df.columns]
    probe_col = "ID" if "ID" in df.columns else ("ID_REF" if "ID_REF" in df.columns else df.columns[0])
    sym_candidates = [c for c in df.columns if re.search(r"(gene.*symbol|symbol)$", c, re.I)]
    sym_col = sym_candidates[0] if sym_candidates else None
    if sym_col:
        probe2sym = df[[probe_col, sym_col]].rename(columns={probe_col:"PROBE", sym_col:"SYMBOL"})
    else:
        ga_candidates = [c for c in df.columns if re.search(r"gene[_\s]*assign", c, re.I)]
        if ga_candidates:
            ga_col = ga_candidates[0]
            tmp = df[[probe_col, ga_col]].rename(columns={probe_col:"PROBE", ga_col:"gene_assignment"})
            def parse_symbol(s):
                if pd.isna(s): return None
                m = re.search(r"\b[A-Z0-9\-]{2,}\b", str(s))
                return m.group(0) if m else None
            tmp["SYMBOL"] = tmp["gene_assignment"].map(parse_symbol)
            probe2sym = tmp[["PROBE","SYMBOL"]]
        else:
            return pd.DataFrame(columns=["PROBE","SYMBOL"])
    probe2sym = probe2sym.dropna()
    probe2sym = probe2sym[probe2sym["SYMBOL"].astype(str).str.len() > 0]
    probe2sym = probe2sym[probe2sym["SYMBOL"] != "---"]
    return probe2sym.drop_duplicates(subset=["PROBE"])

# -------------------------
# Mapping expression -> Hamiltonian params
# -------------------------
def zscore_cols(df, cols):
    X = df[cols].values.astype(np.float64)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0); sd[sd==0]=1.0
    Z = (X - mu)/sd
    return pd.DataFrame(Z, index=df.index, columns=cols)

EPS0, ALPHA, J0, J_MAX = 0.0, 0.50, 0.30, 0.30
def map_expr_to_eps_J(row, genes, alpha=ALPHA, j0=J0, jmax=J_MAX):
    z = np.array([row[g] for g in genes], dtype=np.float64)
    pos = expit(z)
    eps = EPS0 - alpha * z
    J_edges = [(u, v, float(min(j0 * math.sqrt(pos[u] * pos[v]), jmax))) for u, v in line_topology(len(genes))]
    return eps, J_edges

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.title("ENAQT / ETC cohort simulator")
    st.markdown("Upload a precomputed expression parquet/CSV (from the notebook) or try to fetch GSE14520 from GEO (requires GEOparse & internet).")
    col1, col2 = st.columns([1,2])

    with col1:
        uploaded = st.file_uploader("Upload expression (parquet or csv) with sample_id,label and gene columns", type=["parquet","csv"])
        fetch_btn = st.button("Fetch GSE14520 from GEO (optional)")
        st.markdown("**Parameters**")
        n_core_override = st.number_input("n_core (override)", min_value=2, max_value=50, value=7)
        GAMMA_MIN = st.number_input("gamma min", value=0.0, format="%.4f")
        GAMMA_MAX = st.number_input("gamma max", value=0.05, format="%.4f")
        GAMMA_N = st.number_input("gamma steps", value=21, min_value=3, max_value=201)
        T_FINAL = st.number_input("T final", value=150.0, step=50.0)
        DT = st.number_input("dt", value=0.5, format="%.3f")
        K_SINK = st.number_input("k_sink", value=0.1, format="%.3f")
        K_LOSS = st.number_input("k_loss", value=0.0, format="%.3f")
        run_btn = st.button("Run cohort simulation")

    with col2:
        st.info("Preview / logs")
        log_box = st.empty()

    # Load expression data
    expr_df = None
    gene_names = None
    if uploaded is not None:
        try:
            if uploaded.name.endswith(".parquet"):
                expr_df = pd.read_parquet(uploaded)
            else:
                expr_df = pd.read_csv(uploaded)
            log_box.write(f"Loaded {expr_df.shape[0]} samples × {expr_df.shape[1]} columns from {uploaded.name}")
        except Exception as e:
            st.error(f"Failed to read uploaded file: {e}")

    if fetch_btn:
        try:
            st.info("Attempting to fetch GSE14520 (may take a while)...")
            gse = try_fetch_gse14520(destdir=".")
            platform_ids = sorted({ (gsm.metadata.get("platform_id",[None])[0] or "").strip() for gsm in gse.gsms.values() if gsm.metadata.get("platform_id")})
            platform_ids = [pid for pid in platform_ids if pid.startswith("GPL")]
            all_maps = []
            for pid in platform_ids:
                st.write("Fetching platform", pid)
                import GEOparse
                gpl = GEOparse.get_GEO(pid, destdir=".", how="quick")
                m = extract_probe2sym(gpl)
                all_maps.append(m)
            probe2sym = pd.concat(all_maps, ignore_index=True).drop_duplicates("PROBE")
            expr_dict, meta_blobs = {}, {}
            for sid, gsm in gse.gsms.items():
                if gsm.table is None or gsm.table.empty:
                    continue
                df = gsm.table
                try:
                    probes = df.iloc[:,0].astype(str).values
                    values = pd.to_numeric(df.iloc[:,-1], errors="coerce").values
                except Exception:
                    continue
                expr_dict[sid] = pd.Series(values, index=probes, name=sid)
                flat = " | ".join([" ".join(v) if isinstance(v, list) else str(v) for k, v in gsm.metadata.items()])
                meta_blobs[sid] = flat.lower()
            expr_probes = pd.DataFrame(expr_dict)
            expr_probes = expr_probes.replace([np.inf,-np.inf],np.nan).fillna(method="ffill").fillna(method="bfill")
            expr_sym = probe2sym.set_index("PROBE").join(expr_probes, how="inner")
            expr_sym = expr_sym.reset_index(drop=True).groupby("SYMBOL").median()
            ETC_GENES = [
                "NDUFS1","NDUFS2","NDUFA9","NDUFV1",
                "SDHB","UQCRC1","UQCRB","CYC1",
                "COX4I1","COX5A","COX6C","COX7A2",
                "ATP5F1A","ATP5F1B","ATP5MC1","ATP5ME"
            ]
            present = [g for g in ETC_GENES if g in expr_sym.index]
            missing = [g for g in ETC_GENES if g not in expr_sym.index]
            if len(present) < 4:
                all_genes = list(expr_sym.index)
                alt = [g for g in all_genes if re.match(r"^(MT-|ATP|COX|NDUF|UQCR)", g)]
                available_genes = alt[:8] if len(alt) >= 4 else all_genes[:8]
            else:
                available_genes = present
            expr_etc = expr_sym.loc[available_genes]
            expr_etc = expr_etc.replace([np.inf,-np.inf],np.nan)
            expr_etc = expr_etc.apply(lambda col: col.fillna(col.median()), axis=0)
            def label_from_meta(blob):
                if re.search(r"\b(non[-\s]?tumou?r|non[-\s]?tumor|adjacent|normal|healthy)\b", blob): return "normal"
                if re.search(r"\b(tumou?r|hcc|carcinoma|cancer|malignant)\b", blob): return "tumor"
                return "unknown"
            labels = {sid: label_from_meta(meta_blobs.get(sid, "")) for sid in expr_etc.columns}
            keep_cols = [sid for sid,lab in labels.items() if lab in ("tumor","normal")]
            expr_etc = expr_etc[keep_cols]
            labels = {sid: labels[sid] for sid in keep_cols}
            expr_df = expr_etc.T.copy()
            expr_df.insert(0,"sample_id",expr_df.index)
            expr_df.insert(1,"label",[labels[sid] for sid in expr_df.index])
            expr_df.reset_index(drop=True,inplace=True)
            log_box.write(f"Fetched GSE14520 -> {expr_df.shape[0]} samples × {expr_df.shape[1]-2} genes")
        except Exception as e:
            st.error(f"Failed to fetch/parse GSE14520: {e}")

    # If we have expression dataframe, let user run cohort simulation
    if expr_df is not None and run_btn:
        with st.spinner("Running cohort simulations (this can take time)..."):
            gene_names = [c for c in expr_df.columns if c not in ("sample_id","label")]
            if len(gene_names) < 2:
                st.error("Not enough gene columns found. Need at least 2 genes.")
            else:
                z_df = zscore_cols(expr_df, gene_names)
                work_df = pd.concat([expr_df[["sample_id","label"]], z_df], axis=1)
                n_core = int(min(n_core_override, len(gene_names)))
                GAMMA_GRID = np.linspace(float(GAMMA_MIN), float(GAMMA_MAX), int(GAMMA_N))
                sink_idx, loss_idx, D = index_layout(n_core, have_sink=True, have_loss=True)
                records = []
                start = time.time()
                for s_idx, row in work_df.iterrows():
                    eps, J_edges = map_expr_to_eps_J(row, gene_names[:n_core])
                    H = hamiltonian_from_params(n_core, eps, J_edges, static_sigma=0.03, seed=42, add_sink=True, add_loss=True)
                    psi0 = np.zeros((D,1), dtype=np.complex128); psi0[0,0] = 1.0
                    rho0 = psi0 @ psi0.conj().T
                    etes = []
                    for g in GAMMA_GRID:
                        Ls = collapse_ops(n_core, D, gamma=g, sink_idx=sink_idx, sink_target_idx=n_core-1, k_sink=float(K_SINK), k_loss=float(K_LOSS))
                        try:
                            t, rho_last = evolve_with_propagator(H, Ls, rho0, T=float(T_FINAL), dt=float(DT), sink_idx=sink_idx)
                        except Exception:
                            t, rhos = simulate_dynamics(H, Ls, rho0, T=float(T_FINAL), dt=float(DT))
                            rho_last = rhos[-1]
                        ete = float(np.real(rho_last[sink_idx, sink_idx]))
                        tau_c = float(len(t))
                        qls = 0.6*ete + 0.4*(tau_c / (float(T_FINAL)/float(DT) + 1.0))
                        etes.append(ete)
                    i_star = int(np.argmax(etes))
                    records.append({
                        "sample_id": row["sample_id"],
                        "label": row["label"],
                        "ETE_peak": float(etes[i_star]),
                        "gamma_star": float(GAMMA_GRID[i_star]),
                        "tau_c_star": float(len(GAMMA_GRID)),
                    })
                    if (s_idx+1) % 5 == 0 or (s_idx+1) == len(work_df):
                        log_box.write(f"[{s_idx+1}/{len(work_df)}] {row['sample_id']} — ETE*={etes[i_star]:.3f} at γ*={GAMMA_GRID[i_star]:.3f}")
                df_metrics = pd.DataFrame(records)
                st.success("Simulation complete.")
                st.dataframe(df_metrics.head(20))
                out_csv = "cohort_metrics_streamlit.csv"
                df_metrics.to_csv(out_csv, index=False)
                st.markdown(f"Metrics saved to `{out_csv}` (in app working dir).")

    st.markdown("---")
    st.caption("Converted from a Colab notebook. If you need a version that only runs the single-site ENAQT example or want plots exported, ask and I will produce a trimmed script.")

if __name__ == "__main__":
    main()


# analysis.py
"""
Cohort analysis utilities: load metrics, plot cohort summaries, run stats,
select top-correlating gene, network plot, and t-SNE visualization.

Usage examples:
    from analysis import (load_metrics, plot_cohort_summary, run_group_stats,
                          pick_best_gene_and_plot, network_gene_correlation,
                          tsne_qmetrics)
"""

import os
from pathlib import Path
import warnings
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.stats import mannwhitneyu, ttest_ind, spearmanr
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

plt.style.use("seaborn-v0_8-notebook")


def safe_mkdir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def load_metrics(metrics_path):
    """Load cohort metrics CSV, return DataFrame."""
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    df = pd.read_csv(metrics_path)
    return df


def plot_cohort_summary(df_metrics, base_dir=".", out_name="ETE_peak_summary.png", show=False):
    """
    Plot mean ETE_peak by cohort label and save figure.
    Returns saved path.
    """
    base_dir = Path(base_dir)
    safe_mkdir(base_dir)
    nonzero = (df_metrics["ETE_peak"].fillna(0) > 1e-12).sum()
    out_path = base_dir / out_name

    if nonzero == 0:
        # still save an empty placeholder figure to be consistent
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "All ETE values are zero", ha="center", va="center", fontsize=12)
        ax.axis("off")
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        if show:
            plt.show()
        plt.close(fig)
        return out_path

    avg_ete = df_metrics.groupby("label")["ETE_peak"].mean()
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#E64A19" if lab == "normal" else "#2E7D32" for lab in avg_ete.index]
    ax.bar(avg_ete.index, avg_ete.values, color=colors)
    ax.set_ylabel("Average ETE_peak")
    ax.set_title("Mean Quantum Transport Efficiency by Cohort")
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def run_group_stats(df_metrics, group_col="label", value_col="ETE_peak"):
    """
    Run Mann–Whitney U, Welch t-test, and compute Cliff's delta approximation
    between groups 'normal' and 'tumor'. Returns a dict with stats and arrays.
    """
    assert {group_col, value_col}.issubset(df_metrics.columns), f"Need columns: {group_col}, {value_col}"
    sub = df_metrics[df_metrics[group_col].isin(["tumor", "normal"])].copy()
    x_norm = sub.loc[sub[group_col] == "normal", value_col].dropna().values
    x_tum = sub.loc[sub[group_col] == "tumor", value_col].dropna().values

    n1, n2 = len(x_norm), len(x_tum)
    if n1 == 0 or n2 == 0:
        raise ValueError("Not enough samples in one of the groups for statistical testing.")

    u_stat, p_mw = mannwhitneyu(x_norm, x_tum, alternative="two-sided")
    t_stat, p_t = ttest_ind(x_norm, x_tum, equal_var=False)
    delta = 2.0 * u_stat / (n1 * n2) - 1.0

    out = {
        "n_normal": n1,
        "n_tumor": n2,
        "mannwhitney_u": float(u_stat),
        "p_mannwhitney": float(p_mw),
        "t_welch": float(t_stat),
        "p_t_welch": float(p_t),
        "cliffs_delta_approx": float(delta),
        "x_normal": x_norm,
        "x_tumor": x_tum,
    }
    return out


def pick_best_gene_and_plot(expr_df, df_metrics, candidates, base_dir=".", out_name=None, show=False):
    """
    Given expr_df (samples × genes, must include 'sample_id') and df_metrics (must include 'sample_id','ETE_peak'),
    compute Spearman correlations over candidate genes, find the top gene, and save scatter + regression line.
    Returns (best_gene, rho, pval, out_path).
    """
    base_dir = Path(base_dir)
    safe_mkdir(base_dir)

    # Merge expression and metrics
    expr_sub_all = expr_df[["sample_id"] + candidates].copy()
    merge_all = pd.merge(df_metrics[["sample_id", "ETE_peak", "label"]], expr_sub_all, on="sample_id", how="inner").dropna()
    if merge_all.empty:
        raise ValueError("No overlapping samples between expr_df and df_metrics after merge.")

    best_gene, best_rho, best_p = None, 0.0, 1.0
    for g in candidates:
        r, p = spearmanr(merge_all[g].values, merge_all["ETE_peak"].values)
        if np.isnan(r):
            continue
        if abs(r) > abs(best_rho):
            best_gene, best_rho, best_p = g, float(r), float(p)
    if best_gene is None:
        raise RuntimeError("No gene produced a valid correlation.")

    # Plot
    x = merge_all[best_gene].values
    y = merge_all["ETE_peak"].values
    coef = np.polyfit(x, y, 1)
    xline = np.linspace(x.min(), x.max(), 200)
    yline = np.polyval(coef, xline)

    out_name = out_name or f"ETE_vs_{best_gene}.png"
    out_path = base_dir / out_name
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = merge_all["label"].unique()
    palette = {"normal": "#E64A19", "tumor": "#2E7D32"}
    for lab in labels:
        sel = merge_all["label"] == lab
        ax.scatter(merge_all.loc[sel, best_gene], merge_all.loc[sel, "ETE_peak"],
                   s=18, alpha=0.75, label=lab, c=palette.get(lab, "#616161"))
    ax.plot(xline, yline, lw=2, color="black")
    ax.set_xlabel(f"{best_gene} expression")
    ax.set_ylabel("ETE_peak")
    ax.set_title(f"ETE vs {best_gene} (ρ={best_rho:.2f}, p={best_p:.2e})")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return best_gene, best_rho, best_p, out_path


def network_gene_correlation(expr_df, df_metrics, gene_names, base_dir=".", out_name="network_gene_ETE_correlation.png", max_genes=12, show=False):
    """
    Compute Spearman(ETE, expression) for up to max_genes and draw a network graph with node colors.
    Returns out_path.
    """
    base_dir = Path(base_dir)
    safe_mkdir(base_dir)
    # pick preferred mitochondrial-like genes if available, else top variables
    preferred_prefix = ("NDUF", "UQCR", "COX", "ATP", "CYC", "SDHB", "MT-", "CYCS")
    avail = [g for g in gene_names if g.startswith(preferred_prefix)]
    if len(avail) < max_genes:
        # fill with next genes
        avail = list(dict.fromkeys(avail + list(gene_names)))[:max_genes]
    genes_for_graph = avail[:max_genes]

    # create simple chain + chords depending on length
    edges = [(i, i + 1) for i in range(len(genes_for_graph) - 1)]
    if len(genes_for_graph) >= 6:
        edges += [(0, 3), (2, 5)]
    if len(genes_for_graph) >= 9:
        edges += [(6, 8)]

    # build graph
    G = nx.Graph()
    for i, g in enumerate(genes_for_graph):
        G.add_node(i, gene=g)
    G.add_edges_from(edges)

    # compute correlations
    expr_sub = expr_df[["sample_id"] + genes_for_graph].copy()
    m = pd.merge(df_metrics[["sample_id", "ETE_peak"]], expr_sub, on="sample_id", how="inner").dropna()
    node_vals = {}
    for i, g in enumerate(genes_for_graph):
        r, p = spearmanr(m[g].values, m["ETE_peak"].values)
        node_vals[i] = 0.0 if np.isnan(r) else float(r)

    vals = np.array(list(node_vals.values()))
    vmin, vmax = -1.0, 1.0
    normed = (vals - vmin) / (vmax - vmin + 1e-12)

    pos = nx.spring_layout(G, seed=42)
    fig, ax = plt.subplots(figsize=(7, 5))
    nodes = nx.draw_networkx_nodes(G, pos, node_size=600, node_color=normed, cmap="coolwarm", ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.6, alpha=0.7, ax=ax)
    nx.draw_networkx_labels(G, pos, labels={i: G.nodes[i]["gene"] for i in G.nodes()}, font_size=9, ax=ax)
    cbar = fig.colorbar(nodes, ax=ax, shrink=0.8)
    cbar.set_label("Spearman(ETE, expression)")
    ax.set_title("Mitochondrial subnetwork: gene–ETE correlation")
    ax.axis("off")
    out_path = base_dir / out_name
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out_path


def tsne_qmetrics(df_metrics, base_dir=".", out_name="tsne_qmetrics_3d.png", feat_cols=None, show=False):
    """
    Compute 3D t-SNE embedding on a small set of quantum metrics and save a 3D scatter plot PNG.
    feat_cols default: ['ETE_peak','gamma_star','QLS_star']
    """
    base_dir = Path(base_dir)
    safe_mkdir(base_dir)
    if feat_cols is None:
        feat_cols = ["ETE_peak", "gamma_star", "QLS_star"]
    assert set(feat_cols).issubset(df_metrics.columns), f"Missing required feat cols: {feat_cols}"

    X = df_metrics[feat_cols].values
    y = df_metrics["label"].astype(str).values
    n = len(df_metrics)
    perp = max(5, min(30, max(5, n // 10)))
    Xn = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=3, perplexity=perp, n_iter=1500, learning_rate="auto", init="pca", random_state=42)
    Z = tsne.fit_transform(Xn)

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    palette = {"normal": "#E64A19", "tumor": "#2E7D32"}
    for lab in np.unique(y):
        sel = y == lab
        ax.scatter(Z[sel, 0], Z[sel, 1], Z[sel, 2], s=18, alpha=0.8, label=lab, c=palette.get(lab, "#616161"))
    ax.set_title("3D t-SNE of quantum metrics")
    ax.set_xlabel("tSNE-1"); ax.set_ylabel("tSNE-2"); ax.set_zlabel("tSNE-3")
    ax.legend()
    out_path = base_dir / out_name
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return out_path
