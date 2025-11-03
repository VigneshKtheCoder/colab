# backend/core.py
import math, numpy as np, numpy.linalg as npl, scipy.linalg as spla
from scipy.integrate import solve_ivp
from dataclasses import dataclass
from typing import List, Tuple, Dict

np.set_printoptions(precision=4, suppress=True)

def cplx(a): return np.asarray(a, dtype=np.complex128)
def dagger(M): return M.conj().T
def is_hermitian(M, tol=1e-9): return npl.norm(M - M.conj().T) <= tol

def vec(rho): return rho.reshape((-1,), order='F')
def unvec(v, d): return v.reshape((d, d), order='F')

def liouvillian(H, collapses):
    H = cplx(H)
    d = H.shape[0]
    I = np.eye(d, dtype=np.complex128)
    LH = -1j * (np.kron(I, H) - np.kron(H.T, I))
    LD = np.zeros((d*d, d*d), dtype=np.complex128)
    for Lk in collapses:
        Lk = cplx(Lk)
        LdL = Lk.conj().T @ Lk
        LD += (np.kron(Lk, Lk.conj()) -
               0.5*(np.kron(np.eye(d), LdL.T) + np.kron(LdL, np.eye(d))))
    return LH + LD

def line_topology(n): return [(i, i+1) for i in range(n-1)]

def index_layout(n_core, have_sink=True, have_loss=True):
    sink_idx = n_core if have_sink else None
    loss_idx = n_core + 1 if have_sink and have_loss else (n_core if (not have_sink and have_loss) else None)
    d = n_core + (1 if have_sink else 0) + (1 if have_loss else 0)
    return sink_idx, loss_idx, d

def hamiltonian_from_params(n_core, eps, J_edges, static_sigma=0.0, seed=42, add_sink=True, add_loss=True):
    rng = np.random.default_rng(seed)
    eps = np.array(eps, dtype=float)
    if static_sigma > 0: eps = eps + rng.normal(0.0, static_sigma, size=eps.shape)
    d = n_core + (1 if add_sink else 0) + (1 if add_loss else 0)
    H = np.zeros((d, d), dtype=np.complex128)
    for i in range(n_core): H[i, i] = eps[i]
    for (i, j, Jij) in J_edges: H[i, j] = H[j, i] = Jij
    return H

def collapse_ops(n_core, d, gamma, sink_idx, sink_target_idx, k_sink, k_loss):
    cols = []
    if gamma > 0:
        for j in range(n_core):
            Lj = np.zeros((d, d), dtype=np.complex128); Lj[j, j] = np.sqrt(gamma); cols.append(Lj)
    if sink_idx is not None and sink_target_idx is not None and k_sink > 0:
        Ls = np.zeros((d, d), dtype=np.complex128); Ls[sink_idx, sink_target_idx] = np.sqrt(k_sink); cols.append(Ls)
    if k_loss > 0:
        loss_idx = d-1
        for j in range(n_core):
            Lj = np.zeros((d, d), dtype=np.complex128); Lj[loss_idx, j] = np.sqrt(k_loss); cols.append(Lj)
    return cols

def evolve_with_propagator(H, Ls, rho0, T, dt):
    d = H.shape[0]; L = liouvillian(H, Ls); P = spla.expm(L*dt)
    steps = int(round(T/dt))+1; tgrid = np.linspace(0.0, T, steps)
    v = rho0.reshape((-1,), order='F'); rho_last = None
    for _ in tgrid: rho_last = v.reshape((d,d), order='F'); v = P @ v
    return tgrid, rho_last

def simulate_enaqt_curve(n_core, eps, J_edges, gamma_grid, params):
    sink_idx, _, D = index_layout(n_core, True, True)
    H = hamiltonian_from_params(n_core, eps, J_edges,
                                static_sigma=params.get("static_sigma", 0.03),
                                seed=params.get("seed",42),
                                add_sink=True, add_loss=True)
    psi0 = np.zeros((D,1), dtype=np.complex128); psi0[0,0]=1.0
    rho0 = psi0 @ psi0.conj().T
    T = params.get("T", 150.0); dt = params.get("dt", 0.5)
    k_sink = params.get("k_sink", 0.1); k_loss = params.get("k_loss", 0.01)
    sink_target = n_core-1
    etes=[]
    for g in gamma_grid:
        Ls = collapse_ops(n_core, D, g, sink_idx, sink_target, k_sink, k_loss)
        _, rho_last = evolve_with_propagator(H, Ls, rho0, T, dt)
        etes.append(float(np.real(rho_last[sink_idx, sink_idx])))
    return np.array(etes), int(np.argmax(etes))

def zscore_cols(df, cols):
    X = df[cols].to_numpy(dtype=np.float64); mu = np.nanmean(X,0)
    sd = np.nanstd(X,0); sd[sd==0]=1.0
    Z = (X-mu)/sd
    return Z, mu, sd

EPS0, ALPHA, J0, J_MAX = 0.0, 0.50, 0.30, 0.30
def map_expr_to_eps_J(row_dict, genes, alpha=ALPHA, j0=J0, jmax=J_MAX):
    # row_dict: {gene: value}
    z = np.array([row_dict[g] for g in genes], dtype=np.float64)
    pos = 1.0/(1.0+np.exp(-z))
    eps = EPS0 - alpha * z
    J_edges = [(u, v, float(min(j0*math.sqrt(pos[u]*pos[v]), jmax))) for u, v in line_topology(len(genes))]
    return eps, J_edges
