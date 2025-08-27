import os
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ==============================
# PATHS (dynamic, relative to notebook)
# ==============================
base_dir   = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
data_path  = base_dir / "Data"

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
# GRID
# ==============================
x = np.linspace(xmin, xmax, N)

def psi_n_analytic(n, x, L):
    """Analytical eigenfunction for infinite square well centered at 0"""
    if n % 2 == 1:  # odd -> cosine
        return np.sqrt(2/L) * np.cos(n * np.pi * x / L)
    else:           # even -> sine
        return np.sqrt(2/L) * np.sin(n * np.pi * x / L)

def E_n_analytic(n, L):
    """Analytical energy"""
    return (n**2 * np.pi**2) / (2 * L**2)

# ==============================
# TIME EVOLUTION + SAVE
# ==============================
h5_file = data_path / "Analytical/simulation_data.h5"

with h5py.File(h5_file, "w") as f_master:

    # Create master datasets
    f_master.create_dataset("x", data=x)
    f_master.create_dataset("times", data=np.linspace(0, T_total, n_steps))
    energies = np.array([E_n_analytic(n+1, L) for n in range(n_states)])
    f_master.create_dataset("energies", data=energies)

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
    for n in tqdm(range(1, n_states+1), desc="States", unit="state"):
        E = E_n_analytic(n, L)
        psi0 = psi_n_analytic(n, x, L)

        # Preallocate
        psi_real = np.zeros((n_steps, N))
        psi_imag = np.zeros((n_steps, N))
        psi_prob = np.zeros((n_steps, N))

        # Progress bar for time evolution
        for i, t in enumerate(tqdm(times, desc=f"Time evolution (state {n})", unit="step", leave=False)):
            phase = np.exp(-1j * E * t)
            psi_t = psi0 * phase
            psi_real[i, :] = np.real(psi_t)
            psi_imag[i, :] = np.imag(psi_t)
            psi_prob[i, :] = np.abs(psi_t)**2

        # Save into master file
        psi_real_all[n-1, :, :] = psi_real
        psi_imag_all[n-1, :, :] = psi_imag
        psi_prob_all[n-1, :, :] = psi_prob

        # Save also as separate file for convenience
        state_file = data_path / f"Analytical/state_{n}.h5"
        with h5py.File(state_file, "w") as f_state:
            f_state.create_dataset("psi_real", data=psi_real, compression="gzip")
            f_state.create_dataset("psi_imag", data=psi_imag, compression="gzip")
            f_state.create_dataset("psi_prob", data=psi_prob, compression="gzip")
            f_state.create_dataset("x", data=x)
            f_state.create_dataset("times", data=times)
            f_state.create_dataset("energy", data=E)

print(f"\n Data saved in master file: {h5_file}")
print(f" Separate files per state saved in: {data_path}")
