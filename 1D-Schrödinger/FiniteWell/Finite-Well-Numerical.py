"""
Quantum Well Eigenstate Simulator
=================================

This module simulates the eigenstates and time evolution of a quantum particle
in 1D wells (infinite, finite, harmonic). Results are saved to HDF5 and can
be visualized with PyQtGraph in real time.

Features:
- Finite square well (barrier outside well region)
- Eigenstate solver (sparse eigensolver)
- Time evolution of states
- HDF5 results (per-state + master file)
- Real-time PyQtGraph plotting

Author: Abdelrahman Mohamed
Date: 2025-09
"""

import os
import sys

import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
import pyqtgraph as pg


# =============================================================================
# Potential definitions
# =============================================================================

def finite_well(x: np.ndarray, L: float, V0: float = 1.0) -> np.ndarray:
    """Finite well: zero inside, barrier V0 outside."""
    V = np.full_like(x, V0)
    V[np.abs(x) < L / 2] = 0.0
    return V

# =============================================================================
# Helper functions
# =============================================================================
def make_grid(xmin: float, xmax: float, N: int):
    x = np.linspace(xmin, xmax, N)
    dx = x[1] - x[0]
    return x, dx


def second_derivative_matrix(N: int, dx: float):
    main_diag = -2.0 * np.ones(N) / dx**2
    off_diag = 1.0 * np.ones(N - 1) / dx**2
    return diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr")


# =============================================================================
# Simulator Class
# =============================================================================
class QuantumWellSimulator:
    def __init__(self, N=500, xmin=-5.0, xmax=5.0,
                 dt=0.01, T_total=10.0, n_states=5,
                 potential="finite", pot_params=None,
                 base_dir: Path | str = "Data"):
        """
        Initialize simulator with dynamic potential choice.
        """
        self.N, self.xmin, self.xmax = N, xmin, xmax
        self.dt, self.T_total, self.n_states = dt, T_total, n_states
        self.base_dir = Path(base_dir)

        # Grid
        self.x, self.dx = make_grid(xmin, xmax, N)

        if potential == "finite":
            self.V = finite_well(self.x, **pot_params)

        else:
            raise ValueError(f"Unknown potential: {potential}")

        # Hamiltonian
        D2 = second_derivative_matrix(N, self.dx)
        self.H = (-0.5) * D2 + diags(self.V, 0, format="csr")

        # Storage
        self.eigvals = None
        self.eigvecs = None
        self.times = np.linspace(0, T_total, int(T_total / dt) + 1)

        # Output directory
<<<<<<< HEAD
        self.num_dir = self.base_dir / "Numerical"
=======
        self.num_dir = self.base_dir / potential.capitalize()
>>>>>>> 6c370e0dffd62b60b9177e799c8272aed7d44cc3
        os.makedirs(self.num_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    def solve_eigenstates(self):
        eigvals, eigvecs = eigsh(self.H, k=self.n_states, which="SM")
        for k in range(self.n_states):
            norm = np.sqrt(np.sum(np.abs(eigvecs[:, k])**2) * self.dx)
            eigvecs[:, k] /= norm
        order = np.argsort(eigvals)
        self.eigvals = eigvals[order]
        self.eigvecs = eigvecs[:, order]
        print("Eigenenergies:", self.eigvals)
        return self.eigvals, self.eigvecs

    # -------------------------------------------------------------------------
    def evolve_state(self, n: int):
        coeffs = np.zeros(self.n_states, dtype=complex)
        coeffs[n] = 1.0
        psi_real = np.zeros((len(self.times), self.N))
        psi_imag = np.zeros((len(self.times), self.N))
        psi_prob = np.zeros((len(self.times), self.N))
        for i, t in enumerate(self.times):
            psi_t = np.zeros(self.N, dtype=complex)
            for k in range(self.n_states):
                psi_t += coeffs[k] * self.eigvecs[:, k] * np.exp(-1j * self.eigvals[k] * t)
            psi_real[i, :] = np.real(psi_t)
            psi_imag[i, :] = np.imag(psi_t)
            psi_prob[i, :] = np.abs(psi_t) ** 2
        return psi_real, psi_imag, psi_prob

    # -------------------------------------------------------------------------
    def run(self):
        h5_file = self.num_dir / "simulation_data.h5"
        with h5py.File(h5_file, "w") as f:
            f.create_dataset("x", data=self.x)
            f.create_dataset("times", data=self.times)
            f.create_dataset("energies", data=self.eigvals)
            psi_real_all = f.create_dataset(
                "psi_real", shape=(self.n_states, len(self.times), self.N),
                dtype=np.float64, compression="gzip"
            )
            psi_imag_all = f.create_dataset(
                "psi_imag", shape=(self.n_states, len(self.times), self.N),
                dtype=np.float64, compression="gzip"
            )
            psi_prob_all = f.create_dataset(
                "psi_prob", shape=(self.n_states, len(self.times), self.N),
                dtype=np.float64, compression="gzip"
            )
            for n in tqdm(range(self.n_states), desc="Evolving states"):
                psi_real, psi_imag, psi_prob = self.evolve_state(n)
                psi_real_all[n, :, :] = psi_real
                psi_imag_all[n, :, :] = psi_imag
                psi_prob_all[n, :, :] = psi_prob
        print(f"\nMaster file saved at: {h5_file}")
        return h5_file

    # -------------------------------------------------------------------------
    def run_viewer(self, state_idx, x, times, psi_real, psi_imag, psi_prob):
        """Launch a PyQtGraph viewer for one eigenstate."""
        import pyqtgraph as pg  # local import so workers can pickle this
        app = pg.mkQApp(f"Quantum Well Viewer - State {state_idx}")
        win = pg.GraphicsLayoutWidget(show=True, title=f"Quantum Well: State {state_idx}")
        win.resize(800, 600)

        # Real part
        p1 = win.addPlot(title="Wavefunction ℜ ψ(x)")
        curve_real = p1.plot(x, psi_real[state_idx, 0], pen="b")

        # Imag part
        p2 = win.addPlot(title="Wavefunction ℑ ψ(x)")
        curve_imag = p2.plot(x, psi_imag[state_idx, 0], pen="r")

        win.nextRow()

        # Probability
        p3 = win.addPlot(title="Probability Density |ψ|²")
        curve_prob = p3.plot(x, psi_prob[state_idx, 0], pen="g")

        Vscale = np.max(np.abs(self.V))
        if Vscale > 0:
            p3.plot(x, self.V / Vscale * psi_prob[state_idx].max() * 0.9, pen="y")

        # Animation
        frame = 0
        def update():
            nonlocal frame
            curve_real.setData(x, psi_real[state_idx, frame])
            curve_imag.setData(x, psi_imag[state_idx, frame])
            curve_prob.setData(x, psi_prob[state_idx, frame])
            frame = (frame + 1) % len(times)

        timer = pg.QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start(30)

        app.exec()


# =============================================================================
# Script entry point
# =============================================================================
if __name__ == "__main__":
    # === Input parameters (clean block) ===
    N, xmin, xmax = 1000, -5.0, 5.0
    dt, T_total = 0.01, 10.0
    n_states = 5
    pot_params = {"L": 2.0, "V0": 1.0}  # only used for finite or harmonic
<<<<<<< HEAD
    base_dir = "Data"
=======
    base_dir = "WellSim/Data"
>>>>>>> 6c370e0dffd62b60b9177e799c8272aed7d44cc3

    # === Run simulation ===
    sim = QuantumWellSimulator(N=N, xmin=xmin, xmax=xmax,
                               dt=dt, T_total=T_total, n_states=n_states,
                                pot_params=pot_params, base_dir=base_dir)
    sim.solve_eigenstates()
    h5file = sim.run()

    import h5py
    import numpy as np
    import pyqtgraph as pg
    import sys
    from multiprocessing import Process


    def run_viewer(state_idx, x, times, psi_real, psi_imag, psi_prob, V):
        """Launch a PyQtGraph viewer for one state."""
        app = pg.mkQApp(f"Quantum Well Viewer - State {state_idx}")
        win = pg.GraphicsLayoutWidget(show=True, title=f"Quantum Well: State {state_idx}")
        win.resize(800, 600)

        # Real
        p1 = win.addPlot(title="Wavefunction ℜ ψ(x)")
        curve_real = p1.plot(x, psi_real[state_idx, 0], pen="b")

        # Imag
        p2 = win.addPlot(title="Wavefunction ℑ ψ(x)")
        curve_imag = p2.plot(x, psi_imag[state_idx, 0], pen="r")

        win.nextRow()

        # Probability
        p3 = win.addPlot(title="Probability Density |ψ|²")
        curve_prob = p3.plot(x, psi_prob[state_idx, 0], pen="g")

        Vscale = np.max(np.abs(V))
        if Vscale > 0:
            p3.plot(x, V / Vscale * psi_prob[state_idx].max() * 0.9, pen="y")

        # Animate
        frame = 0

        def update():
            nonlocal frame
            curve_real.setData(x, psi_real[state_idx, frame])
            curve_imag.setData(x, psi_imag[state_idx, frame])
            curve_prob.setData(x, psi_prob[state_idx, frame])
            frame = (frame + 1) % len(times)

        timer = pg.QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start(30)

        app.exec()

    import h5py
    from multiprocessing import Process

    # Load results once
    with h5py.File(h5file, "r") as f:
        x = f["x"][:]
        times = f["times"][:]
        psi_real = f["psi_real"][:]
        psi_imag = f["psi_imag"][:]
        psi_prob = f["psi_prob"][:]

    n_states = psi_real.shape[0]

    # Launch one viewer per state (each in its own process)
    processes = []
    for state in range(n_states):
        p = Process(target=sim.run_viewer, args=(state, x, times, psi_real, psi_imag, psi_prob))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()