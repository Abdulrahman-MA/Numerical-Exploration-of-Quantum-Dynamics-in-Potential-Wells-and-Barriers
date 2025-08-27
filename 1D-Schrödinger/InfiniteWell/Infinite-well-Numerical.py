"""
Quantum Infinite Well Simulator
===============================

This module simulates the time evolution of a quantum particle in a 1D potential
well using finite-difference Hamiltonian discretization. Results are saved to HDF5.

Features:
- Flexible grid and Hamiltonian builder
- Supports custom potentials (infinite well, harmonic oscillator, etc.)
- Time evolution of eigenstates
- Automatic directory and file management

Author: Abdelrahman Mohamed
Date: 2025-8
"""

import os
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


# =============================================================================
# Helper functions
# =============================================================================
def make_grid(xmin: float, xmax: float, N: int):
    """Create a uniform 1D spatial grid."""
    x = np.linspace(xmin, xmax, N)
    dx = x[1] - x[0]
    return x, dx


def second_derivative_matrix(N: int, dx: float):
    """Construct sparse Laplacian operator (central 3-point finite difference)."""
    main_diag = -2.0 * np.ones(N) / dx**2
    off_diag = 1.0 * np.ones(N - 1) / dx**2
    return diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr")


def infinite_well_potential(x: np.ndarray, L: float):
    """Infinite square well potential: zero inside, large outside."""
    V = np.zeros_like(x)
    V[np.abs(x) >= L / 2] = 1e6  # effectively infinite walls
    return V


# =============================================================================
# Simulator Class
# =============================================================================
class QuantumWellSimulator:
    """
    Quantum 1D Well Simulator.

    Attributes
    ----------
    x : np.ndarray
        Spatial grid.
    dx : float
        Grid spacing.
    H : scipy.sparse.csr_matrix
        Hamiltonian matrix.
    eigvals : np.ndarray
        Eigenenergies.
    eigvecs : np.ndarray
        Eigenstates (columns).
    """

    def __init__(self, xmin=-1.0, xmax=1.0, N=1000, dt=0.01, T_total=5.0, n_states=5,
                 base_dir: Path | str = "Data"):
        """
        Initialize the simulator.

        Parameters
        ----------
        xmin, xmax : float
            Grid boundaries.
        N : int
            Number of grid points.
        dt : float
            Time step.
        T_total : float
            Total simulation time.
        n_states : int
            Number of eigenstates to evolve.
        base_dir : str or Path
            Base directory for saving results.
        """
        self.xmin, self.xmax, self.N = xmin, xmax, N
        self.dt, self.T_total, self.n_states = dt, T_total, n_states

        # grid
        self.x, self.dx = make_grid(xmin, xmax, N)

        # potential
        self.L = xmax - xmin
        self.V = infinite_well_potential(self.x, self.L)

        # Hamiltonian
        D2 = second_derivative_matrix(N, self.dx)
        self.H = (-0.5) * D2 + diags(self.V, 0, format="csr")

        # paths
        self.base_dir = Path(base_dir)
        self.num_dir = self.base_dir / "Numerical"
        self.ana_dir = self.base_dir / "Analytical"

        # ensure directories exist
        os.makedirs(self.num_dir, exist_ok=True)
        os.makedirs(self.ana_dir, exist_ok=True)

        # storage for results
        self.eigvals = None
        self.eigvecs = None
        self.times = np.linspace(0, T_total, int(T_total / dt))

    # -------------------------------------------------------------------------
    def solve_eigenstates(self):
        """Solve for the lowest-energy eigenstates of the Hamiltonian."""
        eigvals, eigvecs = eigsh(self.H, k=self.n_states, which="SM")

        # normalize
        for k in range(self.n_states):
            norm = np.sqrt(np.sum(np.abs(eigvecs[:, k])**2) * self.dx)
            eigvecs[:, k] /= norm

        # sort
        order = np.argsort(eigvals)
        self.eigvals = eigvals[order]
        self.eigvecs = eigvecs[:, order]

        print("Eigenenergies:", self.eigvals)
        return self.eigvals, self.eigvecs

    # -------------------------------------------------------------------------
    def evolve_state(self, n: int):
        """
        Time-evolve the n-th eigenstate.

        Parameters
        ----------
        n : int
            Index of eigenstate to evolve.

        Returns
        -------
        psi_real, psi_imag, psi_prob : np.ndarray
            Arrays of shape (n_steps, N).
        """
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
            psi_prob[i, :] = np.abs(psi_t)**2

        return psi_real, psi_imag, psi_prob

    # -------------------------------------------------------------------------
    def run(self):
        """Run the full simulation and save results into HDF5 files."""
        h5_file = self.num_dir / "simulation_data.h5"

        with h5py.File(h5_file, "w") as f_master:
            # datasets
            f_master.create_dataset("x", data=self.x)
            f_master.create_dataset("times", data=self.times)
            f_master.create_dataset("energies", data=self.eigvals)

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

            # loop over eigenstates
            for n in tqdm(range(self.n_states), desc="Evolving states", unit="state"):
                psi_real, psi_imag, psi_prob = self.evolve_state(n)

                # save to master file
                psi_real_all[n, :, :] = psi_real
                psi_imag_all[n, :, :] = psi_imag
                psi_prob_all[n, :, :] = psi_prob

                # save individual state file
                state_file = self.num_dir / f"state_{n}.h5"
                with h5py.File(state_file, "w") as f_state:
                    f_state.create_dataset("psi_real", data=psi_real, compression="gzip")
                    f_state.create_dataset("psi_imag", data=psi_imag, compression="gzip")
                    f_state.create_dataset("psi_prob", data=psi_prob, compression="gzip")
                    f_state.create_dataset("x", data=self.x)
                    f_state.create_dataset("times", data=self.times)
                    f_state.create_dataset("energy", data=self.eigvals[n])

        print(f"\nMaster file saved at: {h5_file}")
        print(f"Per-state files saved in: {self.num_dir}")


# =============================================================================
# Script entry point
# =============================================================================
if __name__ == "__main__":
    sim = QuantumWellSimulator()
    sim.solve_eigenstates()
    sim.run()
