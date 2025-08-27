import os

import h5py
from scipy.sparse.linalg import eigsh
from pathlib import Path
from Functions import *
from tqdm import tqdm
import numpy as np

# ==============================
# PATHS (dynamic, relative to notebook)
# ==============================
base_dir   = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
data_path  = base_dir / "Data"

os.makedirs(data_path / "Numerical", exist_ok=True)
os.makedirs(data_path / "Analytical", exist_ok=True)

# ==============================
# PARAMETERS
# ==============================
xmin, xmax = -1.0, 1.0
N = 1000
L = xmax - xmin
dt = 0.01          # time step
T_total = 5.0      # simulate 5 seconds per state
n_steps = int(T_total / dt)
n_states = 5       # number of eigenstates to evolve

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
# TIME EVOLUTION + SAVE
# ==============================
data_path.mkdir(exist_ok=True)
h5_file = data_path / "Numerical/simulation_data.h5"

with h5py.File(h5_file, "w") as f_master:

    # Create master datasets
    f_master.create_dataset("x", data=x)
    f_master.create_dataset("times", data=np.linspace(0, T_total, n_steps))
    f_master.create_dataset("energies", data=vals[:n_states])

    psi_real_all = f_master.create_dataset(
        "psi_real", shape=(n_states, n_steps, N), dtype=np.float64, compression="gzip"
    )
    psi_imag_all = f_master.create_dataset(
        "psi_imag", shape=(n_states, n_steps, N), dtype=np.float64, compression="gzip"
    )
    psi_prob_all = f_master.create_dataset(
        "psi_prob", shape=(n_states, n_steps, N), dtype=np.float64, compression="gzip"
    )

    times = f_master["times"][:]

    # Progress bar for states
    for n in tqdm(range(n_states), desc="States", unit="state"):
        c = np.zeros(n_states, dtype=complex)
        c[0] = 1 / np.sqrt(2)
        c[1] = 1 / np.sqrt(2)

        psi_real = np.zeros((n_steps, N))
        psi_imag = np.zeros((n_steps, N))
        psi_prob = np.zeros((n_steps, N))

        for i, t in enumerate(tqdm(times, desc=f"Time evolution (state {n})", unit="step", leave=False)):
            psi_t = np.zeros(N, dtype=complex)
            for k in range(n_states):
                psi_t += c[k] * vecs[:, k] * np.exp(-1j * vals[k] * t)

            psi_real[i, :] = np.real(psi_t)
            psi_imag[i, :] = np.imag(psi_t)
            psi_prob[i, :] = np.abs(psi_t) ** 2

        # Save into master file
        psi_real_all[n, :, :] = psi_real
        psi_imag_all[n, :, :] = psi_imag
        psi_prob_all[n, :, :] = psi_prob

        # Save also as separate file for convenience
        state_file = data_path / f"Numerical/state_{n}.h5"
        with h5py.File(state_file, "w") as f_state:
            f_state.create_dataset("psi_real", data=psi_real, compression="gzip")
            f_state.create_dataset("psi_imag", data=psi_imag, compression="gzip")
            f_state.create_dataset("psi_prob", data=psi_prob, compression="gzip")
            f_state.create_dataset("x", data=x)
            f_state.create_dataset("times", data=times)
            f_state.create_dataset("energy", data=vals[n])


print(f"\n Data saved in master file: {h5_file}")
print(f" Separate files per state saved in: {data_path}")
