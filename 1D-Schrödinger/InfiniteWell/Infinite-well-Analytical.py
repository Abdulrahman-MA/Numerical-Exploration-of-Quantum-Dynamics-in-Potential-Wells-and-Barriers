"""
Quantum Infinite Well Analytical Simulator
=========================================

This module computes the analytical eigenfunctions and time evolution
of a quantum particle in a 1D infinite square well centered at 0.
Results are saved to HDF5 for consistency with numerical simulations.

Features:
- Analytical eigenfunctions (sine/cosine)
- Analytical energy levels
- Time evolution with closed-form solution
- Automatic directory and file management

Author: Abdelrahman Mohamed
Date: 2025-8
"""

import os
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path


# =============================================================================
# Helper functions
# =============================================================================
def psi_n_analytic(n: int, x: np.ndarray, L: float) -> np.ndarray:
    """Analytical eigenfunction for infinite square well centered at 0."""
    if n % 2 == 1:  # odd index → cosine
        return np.sqrt(2 / L) * np.cos(n * np.pi * x / L)
    else:           # even index → sine
        return np.sqrt(2 / L) * np.sin(n * np.pi * x / L)


def E_n_analytic(n: int, L: float) -> float:
    """Analytical energy eigenvalue for state n."""
    return (n**2 * np.pi**2) / (2 * L**2)


# =============================================================================
# Simulator Class
# =============================================================================
class AnalyticalWellSimulator:
    """
    Analytical Infinite Square Well Simulator.

    Attributes
    ----------
    x : np.ndarray
        Spatial grid.
    times : np.ndarray
        Time points.
    n_states : int
        Number of eigenstates evolved.
    L : float
        Well width.
    """

    def __init__(self, xmin=-1.0, xmax=1.0, N=1000, dt=0.01, T_total=5.0, n_states=5,
                 base_dir: Path | str = "Data"):
        """
        Initialize the simulator.

        Parameters
        ----------
        xmin, xmax : float
            Spatial domain boundaries.
        N : int
            Number of grid points.
        dt : float
            Time step.
        T_total : float
            Total simulation time.
        n_states : int
            Number of eigenstates to compute.
        base_dir : str or Path
            Base directory for saving results.
        """
        self.xmin, self.xmax, self.N = xmin, xmax, N
        self.dt, self.T_total, self.n_states = dt, T_total, n_states

        self.x = np.linspace(xmin, xmax, N)
        self.L = xmax - xmin
        self.times = np.linspace(0, T_total, int(T_total / dt))

        # paths
        self.base_dir = Path(base_dir)
        self.ana_dir = self.base_dir / "Analytical"
        os.makedirs(self.ana_dir, exist_ok=True)

        # storage
        self.energies = np.array([E_n_analytic(n + 1, self.L) for n in range(n_states)])

    # -------------------------------------------------------------------------
    def evolve_state(self, n: int):
        """
        Time evolve the n-th analytical eigenstate.

        Parameters
        ----------
        n : int
            State index (0-based, maps to n+1).

        Returns
        -------
        psi_real, psi_imag, psi_prob : np.ndarray
            Arrays of shape (n_steps, N).
        """
        n_eff = n + 1  # shift (state index starts from 1 in formulas)
        E = self.energies[n]
        psi0 = psi_n_analytic(n_eff, self.x, self.L)

        psi_real = np.zeros((len(self.times), self.N))
        psi_imag = np.zeros((len(self.times), self.N))
        psi_prob = np.zeros((len(self.times), self.N))

        for i, t in enumerate(self.times):
            phase = np.exp(-1j * E * t)
            psi_t = psi0 * phase
            psi_real[i, :] = np.real(psi_t)
            psi_imag[i, :] = np.imag(psi_t)
            psi_prob[i, :] = np.abs(psi_t)**2

        return psi_real, psi_imag, psi_prob

    # -------------------------------------------------------------------------
    def run(self):
        """Run full simulation and save results to HDF5."""
        h5_file = self.ana_dir / "simulation_data.h5"

        with h5py.File(h5_file, "w") as f_master:
            # master datasets
            f_master.create_dataset("x", data=self.x)
            f_master.create_dataset("times", data=self.times)
            f_master.create_dataset("energies", data=self.energies)

            psi_real_all = f_master.create_dataset(
                "psi_real", shape=(self.n_states, len(self.times), self.N),
                dtype=np.float64, compression="gzip"
            )
            psi_imag_all = f_master.create_dataset(
                "psi_imag", shape=(self.n_states, len(self.times), self.N),
                dtype=np.float64, compression="gzip"
            )
            psi_prob_all = f_master.create_dataset(
                "psi_prob", shape=(self.n_states, len(self.times), self.N),
                dtype=np.float64, compression="gzip"
            )

            # loop over states
            for n in tqdm(range(self.n_states), desc="Analytical states", unit="state"):
                psi_real, psi_imag, psi_prob = self.evolve_state(n)

                psi_real_all[n, :, :] = psi_real
                psi_imag_all[n, :, :] = psi_imag
                psi_prob_all[n, :, :] = psi_prob

                # individual file
                state_file = self.ana_dir / f"state_{n}.h5"
                with h5py.File(state_file, "w") as f_state:
                    f_state.create_dataset("psi_real", data=psi_real, compression="gzip")
                    f_state.create_dataset("psi_imag", data=psi_imag, compression="gzip")
                    f_state.create_dataset("psi_prob", data=psi_prob, compression="gzip")
                    f_state.create_dataset("x", data=self.x)
                    f_state.create_dataset("times", data=self.times)
                    f_state.create_dataset("energy", data=self.energies[n])

        print(f"\nMaster file saved at: {h5_file}")
        print(f"Per-state files saved in: {self.ana_dir}")


# =============================================================================
# Script entry point
# =============================================================================
if __name__ == "__main__":
    sim = AnalyticalWellSimulator()
    sim.run()
