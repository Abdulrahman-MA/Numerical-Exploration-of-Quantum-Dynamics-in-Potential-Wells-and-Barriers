import numpy as np
from scipy.sparse.linalg import eigsh
import importlib
import sys
import os
from pathlib import Path
from Functions import *

# ==============================
# PATHS (dynamic, relative to notebook)
# ==============================
base_dir   = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
utils_path = base_dir / "Utils"
data_path  = base_dir / "Data"

# ==============================
# PARAMETERS
# ==============================
xmin, xmax = -1.0, 1.0
N = 400
L = xmax - xmin
dt = 0.01          # time step
T_total = 5.0      # simulate 5 seconds per state
n_steps = int(T_total / dt)
n_states = 3       # number of eigenstates to evolve

# ==============================
# GRID + OPERATORS
# ==============================
x, dx = make_grid(xmin, xmax, N)
D2 = second_derivative_matrix(N, dx)
V = infinite_well_potential(x, L=L)
H = build_hamiltonian(D2, V)

# ==============================
# EIGENSTATES
# ==============================
vals, vecs = eigsh(H, k=n_states, which="SM")
order = np.argsort(vals)
vals = vals[order]
vecs = vecs[:, order]

print("Eigenenergies:", vals[:n_states])

# ==============================
# TIME EVOLUTION
# ==============================
psi_real_all, psi_imag_all, psi_prob_all = [], [], []
times = np.linspace(0, T_total, n_steps)

for n in range(n_states):
    E = vals[n]
    psi0 = vecs[:, n]

    psi_real = np.zeros((n_steps, N))
    psi_imag = np.zeros((n_steps, N))
    psi_prob = np.zeros((n_steps, N))

    for i, t in enumerate(times):
        phase = np.exp(-1j * E * t)
        psi_t = psi0 * phase
        psi_real[i, :] = np.real(psi_t)
        psi_imag[i, :] = np.imag(psi_t)
        psi_prob[i, :] = np.abs(psi_t)**2

    psi_real_all.append(psi_real)
    psi_imag_all.append(psi_imag)
    psi_prob_all.append(psi_prob)

# convert to numpy arrays
psi_real_all = np.array(psi_real_all)
psi_imag_all = np.array(psi_imag_all)
psi_prob_all = np.array(psi_prob_all)

# ==============================
# SAVE DATA
# ==============================
data_path.mkdir(exist_ok=True)

np.save(data_path / "psi_real.npy", psi_real_all)
np.save(data_path / "psi_imag.npy", psi_imag_all)
np.save(data_path / "psi_prob.npy", psi_prob_all)
np.save(data_path / "x.npy", x)         # save grid
np.save(data_path / "times.npy", times) # save time points

print("Data saved in:", data_path)
