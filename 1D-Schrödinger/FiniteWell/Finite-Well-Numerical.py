"""
Finite Potential Well Simulator
===============================

This module simulates the bound eigenstates of a particle in a 1D finite
potential well using finite-difference Hamiltonian discretization. Results
(time evolution of eigenstates) are saved to HDF5.

Features:
- Flexible grid and Hamiltonian builder
- Custom finite-well potential (inside/outside values)
- Eigenstate solver + phase-based time evolution
- Automatic directory and file management

Author: Abdelrahman Mohamed
Date: 2025-08
"""

import os
import h5py
import numpy as np
from tqdm import tqdm
import pyqtgraph as pg
from pathlib import Path
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
    """Construct sparse Laplacian operator (3-point finite difference)."""
    main_diag = -2.0 * np.ones(N) / dx**2
    off_diag = 1.0 * np.ones(N - 1) / dx**2
    return diags([off_diag, main_diag, off_diag], offsets=[-1, 0, 1], format="csr")


def finite_well_potential(x: np.ndarray, L: float,
                          V_inside: float = 0.0, V_outside: float = 50.0):
    """Finite square well potential: inside=V_inside, outside=V_outside."""
    V = np.ones_like(x) * V_outside
    V[np.abs(x) < L / 2] = V_inside
    return V


# =============================================================================
# Simulator Class
# =============================================================================
class FiniteWellSimulator:
    """
    Finite 1D Potential Well Simulator.

    Attributes
    ----------
    x : np.ndarray
        Spatial grid.
    dx : float
        Grid spacing.
    V : np.ndarray
        Potential profile.
    H : scipy.sparse.csr_matrix
        Hamiltonian matrix.
    eigvals : np.ndarray
        Eigenenergies.
    eigvecs : np.ndarray
        Eigenstates (columns).
    """

    def __init__(self, xmin=-30.0, xmax=30.0, N=2048, dt=0.001, T_total=8.0,
                 n_states=3, L=10.0, V_inside=0.0, V_outside=50.0,
                 base_dir: Path | str = "Data"):
        """
        Initialize the finite well simulator.

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
        L : float
            Width of the well.
        V_inside : float
            Potential inside the well.
        V_outside : float
            Potential outside the well.
        base_dir : str or Path
            Base directory for saving results.
        """
        self.xmin, self.xmax, self.N = xmin, xmax, N
        self.dt, self.T_total, self.n_states = dt, T_total, n_states
        self.L, self.V_inside, self.V_outside = L, V_inside, V_outside

        # grid
        self.x, self.dx = make_grid(xmin, xmax, N)

        # potential
        self.V = finite_well_potential(self.x, L, V_inside, V_outside)

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
        h5_file = self.num_dir / "finite_well_simulation.h5"

        with h5py.File(h5_file, "w") as f_master:
            # datasets
            f_master.create_dataset("x", data=self.x)
            f_master.create_dataset("times", data=self.times)
            f_master.create_dataset("energies", data=self.eigvals)
            f_master.create_dataset("V", data=self.V)

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
                state_file = self.num_dir / f"finite_state_{n}.h5"
                with h5py.File(state_file, "w") as f_state:
                    f_state.create_dataset("psi_real", data=psi_real, compression="gzip")
                    f_state.create_dataset("psi_imag", data=psi_imag, compression="gzip")
                    f_state.create_dataset("psi_prob", data=psi_prob, compression="gzip")
                    f_state.create_dataset("x", data=self.x)
                    f_state.create_dataset("times", data=self.times)
                    f_state.create_dataset("energy", data=self.eigvals[n])
                    f_state.create_dataset("V", data=self.V)

        print(f"\nMaster file saved at: {h5_file}")
        print(f"Per-state files saved in: {self.num_dir}")

    @staticmethod
    def plot_states(h5file: Path = "Data/Numerical/finite_well_simulation.h5",
                          n_states: list[int] = [0], interval: int = 30) -> None:
            """
            Animate stored eigenstates from an HDF5 simulation file (no recalculation).

            Parameters
            ----------
            h5file : Path
                Path to the HDF5 file containing the simulation results.
                Must include datasets: "x", "psi_real", "psi_imag", "psi_prob", and optionally "V".
            n_states : list[int], optional
                List of eigenstate indices to animate (default: [0]).
            interval : int, optional
                Frame update interval in milliseconds (default: 30 ms).

            Notes
            -----
            - Each eigenstate is plotted in three rows:
                1. Real part of the wavefunction ℜψ(x, t)
                2. Imaginary part ℑψ(x, t)
                3. Probability density |ψ(x, t)|² with scaled potential overlay
            - The animation stops automatically at the final time step (no looping).
            - Multiple eigenstates are stacked vertically for comparison.
            """
            # --- Load data ---
            with h5py.File(h5file, "r") as f:
                x = f["x"][:]
                psi_real_all = f["psi_real"][:]   # shape: (n_states, n_times, N)
                psi_imag_all = f["psi_imag"][:]
                psi_prob_all = f["psi_prob"][:]
                V = f["V"][:] if "V" in f else np.zeros_like(x)

            n_times = psi_real_all.shape[1]
            frame = 0

            # --- Setup Qt app and window ---
            app = pg.mkQApp("Finite Well Animation")
            win = pg.GraphicsLayoutWidget(show=True, title="Finite Well Quantum States")
            win.resize(1000, 300 * len(n_states))  # scale window by number of states

            # --- Prepare plots for each requested eigenstate ---
            curves = []  # store curve handles for updating

            for idx, n in enumerate(n_states):
                psi_real = psi_real_all[n]
                psi_imag = psi_imag_all[n]
                psi_prob = psi_prob_all[n]

                # Row 1: Real part
                p1 = win.addPlot(title=f"Eigenstate {n}: ℜ ψ(x)")
                curve_real = p1.plot(x, psi_real[0], pen="b")

                # Row 2: Imag part
                p2 = win.addPlot(title=f"Eigenstate {n}: ℑ ψ(x)")
                curve_imag = p2.plot(x, psi_imag[0], pen="r")

                win.nextRow()

                # Row 3: Probability + Potential
                p3 = win.addPlot(title=f"Eigenstate {n}: |ψ(x)|²")
                curve_prob = p3.plot(x, psi_prob[0], pen="g")
                Vscale = np.max(np.abs(V))
                if Vscale > 0:
                    p3.plot(x, V / Vscale * psi_prob.max() * 0.9, pen="y")

                # Store handles + data for update loop
                curves.append((curve_real, curve_imag, curve_prob, psi_real, psi_imag, psi_prob))

            # --- Update function ---
            def update():
                nonlocal frame
                for curve_real, curve_imag, curve_prob, psi_real, psi_imag, psi_prob in curves:
                    curve_real.setData(x, psi_real[frame])
                    curve_imag.setData(x, psi_imag[frame])
                    curve_prob.setData(x, psi_prob[frame])
                if frame < n_times - 1:
                    frame += 1

            # --- Timer setup ---
            timer = pg.QtCore.QTimer()
            timer.timeout.connect(update)
            timer.start(interval)

            # --- Run app ---
            app.exec()


# =============================================================================
# Script entry point
# =============================================================================
if __name__ == "__main__":

    # grid parameters
    xmin, xmax, N = -30, 30, 2048

    # time parameters
    dt, T_total = 0.001, 8.0

    # number of states to be calculated
    n_states = 1

    # potential parameters
    L, V_inside, V_outside = 10.0, 0.0, 50.0

    # plot stats
    n_plot = [0]
    '''
    sim = FiniteWellSimulator(
        xmin=xmin, xmax=xmax, N=N,
        dt=dt, T_total=T_total,
        n_states=n_states,
        L=L, V_inside=V_inside, V_outside=V_outside,
        base_dir="Data"
    )
    sim.solve_eigenstates()
    sim.run()
    '''
    FiniteWellSimulator.plot_states(n_states=n_plot, interval=int(round(T_total)))