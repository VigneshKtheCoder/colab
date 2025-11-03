# frontend/app.py
import io, json, time, numpy as np, pandas as pd, requests, streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Quantum Diagnostics Platform", layout="wide")
st.title("🧬 Quantum Diagnostics Platform")
st.caption("Mechanistic, first-principles diagnostics — quantum transport modeled from patient omics.")

API_URL = st.secrets.get("API_URL", "http://localhost:8000")

# ---- Sidebar (params) ----
st.sidebar.header("⚙️ Simulation Controls")
gamma_min = st.sidebar.number_input("γ min", 0.0, 0.2, 0.00, 0.001)
gamma_max = st.sidebar.number_input("γ max", 0.0, 0.2, 0.05, 0.001)
gamma_steps = st.sidebar.number_input("γ steps", 5, 501, 21, 1)
T_final = st.sidebar.number_input("T (final)", 10.0, 1000.0, 150.0, 1.0)
dt = st.sidebar.number_input("dt", 0.01, 5.0, 0.5, 0.01)
k_sink = st.sidebar.number_input("k_sink", 0.0, 2.0, 0.10, 0.01)
k_loss = st.sidebar.number_input("k_loss", 0.0, 2.0, 0.01, 0.01)
n_core = st.sidebar.number_input("n_core", 2, 50, 7, 1)
static_sigma = st.sidebar.number_input("static σ", 0.0, 0.5, 0.03, 0.01)
seed = st.sidebar.number_input("seed", 0, 1000000, 42, 1)

uploaded = st.file_uploader("Upload any tabular file (CSV/TSV/XLSX/JSON/Parquet). Must include `sample_id` (or will be created) and gene columns; `label` is optional.", type=None)
st.write("**Tip:** column names like `sample_id`, `label`, `COX4I1`, `NDUFS2`, etc.")

tabs = st.tabs(["▶️ Simulation", "📊 Cohort Results", "🧬 Gene Correlations", "🔗 Network View"])

@st.cache_data(show_spinner=False)
def ingest_preview(file_bytes, name):
    files = {"file": (name, file_bytes)}
    r = requests.post(f"{API_URL}/ingest", files=files, timeout=120)
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=False)
def run_sim(file_bytes, name, params):
    files = {"file": (name, file_bytes)}
    data = {"params": json.dumps(params)}
    r = requests.post(f"{API_URL}/simulate", files=files, data=data, timeout=900)
    r.raise_for_status()
    return r.json()

if uploaded:
    preview = ingest_preview(uploaded.getvalue(), uploaded.name)
    with st.expander("Preview / schema"):
        st.json(preview)

    params = dict(
        gamma_min=gamma_min, gamma_max=gamma_max, gamma_steps=int(gamma_steps),
        T=T_final, dt=dt, k_sink=k_sink, k_loss=k_loss, n_core=int(n_core),
        static_sigma=static_sigma, seed=int(seed)
    )

    with tabs[0]:
        st.subheader("Run cohort simulation")
        if st.button("Run", type="primary"):
            with st.spinner("Simulating quantum transport across cohort…"):
                res = run_sim(uploaded.getvalue(), uploaded.name, params)
            if "error" in res:
                st.error(res["error"])
            else:
                df = pd.DataFrame(res["metrics"])
                st.success(f"Completed: {len(df)} samples")
                st.dataframe(df.head(20), use_container_width=True)
                st.session_state["metrics_df"] = df

                # Animated ENAQT feel: show ETE_peak distribution building up
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df["ETE_peak"], nbinsx=30))
                fig.update_layout(title="Distribution of ETE_peak", xaxis_title="ETE_peak", yaxis_title="count")
                st.plotly_chart(fig, use_container_width=True)

    with tabs[1]:
        st.subheader("Cohort Results")
        df = st.session_state.get("metrics_df")
        if df is None:
            st.info("Run a simulation first")
        else:
            cols = st.columns(2)
            with cols[0]:
                # Box plot by label if present
                if "label" in df.columns and df["label"].notna().any():
                    fig = go.Figure()
                    for lab in df["label"].dropna().unique():
                        fig.add_trace(go.Box(y=df[df["label"]==lab]["ETE_peak"], name=f"{lab}"))
                    fig.update_layout(title="ETE_peak by cohort")
                    st.plotly_chart(fig, use_container_width=True)
            with cols[1]:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=df["gamma_star"], y=df["ETE_peak"], mode="markers", name="samples"))
                fig2.update_layout(title="ETE_peak vs gamma* (per sample)", xaxis_title="gamma*", yaxis_title="ETE_peak")
                st.plotly_chart(fig2, use_container_width=True)
            st.download_button("Download metrics CSV", df.to_csv(index=False).encode(), "cohort_metrics.csv", "text/csv")

    with tabs[2]:
        st.subheader("Gene ↔ ETE correlations (quick look)")
        st.info("Upload the same file to this tab if you want ad-hoc correlations with your metrics CSV later.")
        dfm = st.session_state.get("metrics_df")
        if dfm is None:
            st.caption("Run a simulation to populate metrics.")
        else:
            st.caption("Because parsing of raw omics can be heavy, correlations are kept simple here; deeper analysis can be added if you share large cohorts.")
            st.write("Pick a gene column name to correlate against `ETE_peak` after joining on `sample_id`.")
            # lightweight helper: re-open uploaded file and try to merge
            raw = ingest_preview(uploaded.getvalue(), uploaded.name)  # already cached
            gene_guess = st.text_input("Gene column (e.g., COX4I1, NDUFS2, ATP5F1A):", "")
            if st.button("Compute correlation"):
                try:
                    import pandas as pd
                    # read again fully
                    from io import BytesIO
                    import pyarrow  # ensure parquet ok if used
                    def read_generic(bts, name):
                        name = name.lower()
                        bio = BytesIO(bts)
                        if name.endswith(".parquet"): return pd.read_parquet(bio)
                        if name.endswith(".json"): return pd.read_json(bio)
                        if name.endswith(".xlsx") or name.endswith(".xls"): return pd.read_excel(bio)
                        return pd.read_csv(bio)
                    raw_df = read_generic(uploaded.getvalue(), uploaded.name)
                    if "sample_id" not in raw_df.columns: raw_df.insert(0,"sample_id", raw_df.index.astype(str))
                    m = dfm.merge(raw_df[["sample_id", gene_guess]], on="sample_id", how="inner").dropna()
                    if m.empty: st.warning("No overlap or gene not found."); 
                    else:
                        r = float(np.corrcoef(m["ETE_peak"].values, m[gene_guess].values)[0,1])
                        st.metric(f"Pearson r (ETE_peak, {gene_guess})", f"{r:.3f}")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=m[gene_guess], y=m["ETE_peak"], mode="markers"))
                        fig.update_layout(xaxis_title=f"{gene_guess} expression", yaxis_title="ETE_peak")
                        st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")

    with tabs[3]:
        st.subheader("Network View (illustrative)")
        st.caption("Quick visualization of core chain + chords; edge colors reflect hypothetical sensitivity.")
        import networkx as nx
        n = int(n_core)
        edges = [(i, i+1) for i in range(n-1)]
        if n>=6: edges += [(0,3),(2,5)]
        if n>=9: edges += [(6,8)]
        G = nx.Graph(); [G.add_node(i) for i in range(n)]; G.add_edges_from(edges)
        pos = nx.spring_layout(G, seed=42)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6,4))
        nx.draw(G, pos, with_labels=True, node_size=500, ax=ax)
        st.pyplot(fig, use_container_width=True)
