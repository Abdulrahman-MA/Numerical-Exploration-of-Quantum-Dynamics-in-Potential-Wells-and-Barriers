"""
Quantum Scattering Simulator (Split-Operator Method)
====================================================

This module implements a 1D quantum wave packet simulator using the
split-operator Fourier method. It allows the study of scattering,
tunneling, bound states, and free propagation in different potentials.

Features:
- Infinite square well
- Finite well / barrier
- Harmonic oscillator
- Gaussian wave packet initialization
- Absorbing boundary layers
- HDF5 output for real, imaginary, and probability densities
- Pre-check function to classify scenarios before running

Author: Abdelrahman Mohamed
Date: 2025-08
"""

import os
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pyqtgraph as pg


# =============================================================================
# Scenario Pre-check
# =============================================================================
def precheck_scenario(L: float, V_inside: float, V_outside: float,
                      k0: float, x_start: float) -> tuple[float, str]:
    """
    Quick classification of the quantum scattering scenario.
    """
    E0 = 0.5 * k0**2  # kinetic energy in natural units

    if np.isclose(V_inside, V_outside):
        if np.isclose(E0, V_inside):
            mode = "Free particle (at threshold energy)"
        else:
            mode = "Free particle (flat potential)"
    elif V_outside < V_inside:
        if E0 < V_outside:
            mode = "Bound state (localized inside well)"
        elif E0 < V_inside:
            mode = "Resonant tunneling (partially trapped)"
        else:
            mode = "Free / scattering out of the well"
    else:
        if E0 < V_outside:
            if x_start < -L/2 or x_start > L/2:
                mode = "Reflection with possible tunneling"
            else:
                mode = "Metastable trapped (inside, can tunnel out)"
        elif np.isclose(E0, V_outside):
            mode = "Threshold scattering"
        else:
            mode = "Normal scattering (transmission + reflection)"

    print("=== Scenario Pre-check ===")
    print(f"Inside potential : {V_inside}")
    print(f"Outside potential: {V_outside}")
    print(f"Packet energy    : {E0:.4f}")
    print(f"Result           : {mode}")


# =============================================================================
# Potential Definitions
# =============================================================================
def finite_well(x: np.ndarray, L: float,
                V_inside: float = 0.0, V_outside: float = 1.0) -> np.ndarray:
    V = np.ones_like(x) * V_outside
    V[np.abs(x) < L / 2] = V_inside
    return V


# =============================================================================
# Absorber
# =============================================================================
def make_edge_absorber(x: np.ndarray, xmin: float, xmax: float,
                       thickness: float = 1.0,
                       power: float = 4.0,
                       strength: float = 8.0) -> np.ndarray:
    if thickness is None or thickness <= 0:
        return np.ones_like(x)

    left_start = xmin + thickness
    right_start = xmax - thickness
    M = np.ones_like(x, dtype=float)

    mask_left = x <= left_start
    if np.any(mask_left):
        d = (left_start - x[mask_left]) / thickness
        d = np.clip(d, 0.0, 1.0)
        M[mask_left] = np.exp(- strength * (d ** power))

    mask_right = x >= right_start
    if np.any(mask_right):
        d = (x[mask_right] - right_start) / thickness
        d = np.clip(d, 0.0, 1.0)
        M[mask_right] = np.exp(- strength * (d ** power))

    return M


# =============================================================================
# Quantum Simulator
# =============================================================================
class QuantumSimulator:
    def __init__(self, N=1024, xmin=-12.0, xmax=12.0,
                 dt=0.005, T_total=10.0, save_every=10,
                 potential="finite", pot_params=None,
                 base_dir: str | Path = "Data",
                 absorber_thickness=0.5):
        """
        Initialize simulation.
        """
        self.N, self.xmin, self.xmax = N, xmin, xmax
        self.dt, self.T_total, self.save_every = dt, T_total, save_every
        self.steps = int(T_total / dt)
        self.base_dir = Path(base_dir)

        # Spatial grid
        self.x = np.linspace(xmin, xmax, N, endpoint=False)
        self.dx = self.x[1] - self.x[0]

        # Potential
        if potential == "finite":
            self.V = finite_well(self.x, **pot_params)
        else:
            raise ValueError(f"Unknown potential: {potential}")

        # Wavefunction
        self.psi = np.zeros(N, dtype=complex)

        # Momentum grid and kinetic operator
        self.k = 2.0 * np.pi * np.fft.fftfreq(N, d=self.dx)
        self.Tk = 0.5 * self.k**2

        # Split-operator propagators
        self.expV_half = np.exp(-0.5j * dt * self.V)
        self.expT_full = np.exp(-1.0j * self.Tk * dt)

        # Absorber
        self.abs_mask = make_edge_absorber(self.x, self.x[0], self.x[-1],
                                           thickness=absorber_thickness,
                                           power=12.0)

    def set_initial_packet(self, x0=-8.0, k0=6.0, sigma=0.6):
        psi0 = np.exp(-(self.x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * self.x)
        psi0 /= np.sqrt(np.trapezoid(np.abs(psi0)**2, self.x))
        self.psi = psi0

    def step(self):
        self.psi *= self.expV_half
        psi_k = np.fft.fft(self.psi)
        psi_k *= self.expT_full
        self.psi = np.fft.ifft(psi_k)
        self.psi *= self.expV_half
        self.psi *= self.abs_mask

    def run(self):
        out_file = self.base_dir / "scattering_run.h5"
        os.makedirs(self.base_dir, exist_ok=True)

        dx = (self.xmax - self.xmin) / self.N
        self.absorber_thickness = absorber_thickness
        trim = int(self.absorber_thickness / dx)

        x_save = self.x[trim: self.N - trim]
        V_save = self.V[trim: self.N - trim]

        with h5py.File(out_file, "w") as f:
            f.create_dataset("x", data=x_save)
            f.create_dataset("V", data=V_save)

            psi_real = f.create_dataset("psi_real", (self.steps // self.save_every + 1, len(x_save)), dtype=np.float64)
            psi_imag = f.create_dataset("psi_imag", (self.steps // self.save_every + 1, len(x_save)), dtype=np.float64)
            psi_prob = f.create_dataset("psi_prob", (self.steps // self.save_every + 1, len(x_save)), dtype=np.float64)
            times = f.create_dataset("times", (self.steps // self.save_every + 1,), dtype=np.float64)

            psi_real[0] = np.abs(self.psi[trim: self.N - trim].real)
            psi_imag[0] = np.abs(self.psi[trim: self.N - trim].imag)
            psi_prob[0] = np.abs(self.psi[trim: self.N - trim]) ** 2
            times[0] = 0.0

            idx = 1
            for t in tqdm(range(1, self.steps + 1), desc="Propagating"):
                self.step()
                if t % self.save_every == 0:
                    psi_real[idx] = self.psi[trim: self.N - trim].real
                    psi_imag[idx] = self.psi[trim: self.N - trim].imag
                    psi_prob[idx] = np.abs(self.psi[trim: self.N - trim]) ** 2
                    times[idx] = t * self.dt
                    idx += 1

        print(f"\nResults saved to: {out_file}")


# =============================================================================
# Script Entry Point
# =============================================================================
if __name__ == "__main__":
    N, xmin, xmax = 2048, -30.0, 30.0
    L, V_in, V_out = 5.0, 0.0, 1.9
    k0, x0 = 2, -8.0
    dt, T_total, save_every = 0.001, 20.0, 50
    absorber_thickness = 10

    precheck_scenario(L, V_in, V_out, k0, x0)

    sim = QuantumSimulator(
        N=N, xmin=xmin, xmax=xmax,
        dt=dt, T_total=T_total, save_every=save_every,
        potential="finite", pot_params={"L": L, "V_inside": V_in, "V_outside": V_out},
        base_dir="FiniteWell/Data/Scattering", absorber_thickness=absorber_thickness
    )
    """Paths : FiniteWell/Data/Scattering, InfiniteWell/Data/Scattering"""
    sim.set_initial_packet(x0=x0, k0=k0, sigma=1.0)
    sim.run()

    h5file = "FiniteWell/Data/Scattering/scattering_run.h5"
    with h5py.File(h5file, "r") as f:
        x = f["x"][:]
        psi_real = f["psi_real"][:]
        psi_imag = f["psi_imag"][:]
        psi_prob = f["psi_prob"][:]
        V = f["V"][:]
        times = f["times"][:]
        n_times, N = psi_real.shape

        app = pg.mkQApp("Wave Packet Viewer")
        win = pg.GraphicsLayoutWidget(show=True, title="Quantum Scattering")
        win.resize(800, 600)

        p1 = win.addPlot(title="Wavefunction ℜ ψ(x)")
        curve_real = p1.plot(x, psi_real[0], pen="b")

        p2 = win.addPlot(title="Wavefunction ℑ ψ(x)")
        curve_imag = p2.plot(x, psi_imag[0], pen="r")

        win.nextRow()
        p3 = win.addPlot(title="Probability Density |ψ|²")
        curve_prob = p3.plot(x, psi_prob[0], pen="g")

        Vscale = np.max(np.abs(V))
        if Vscale > 0:
            p3.plot(x, V / Vscale * psi_prob.max() * 0.9, pen="y")
        else:
            p3.plot(x, V / np.max(V) * psi_prob.max() * 0.9, pen="y")

        frame = 0
        def update():
            global frame
            curve_real.setData(x, psi_real[frame])
            curve_imag.setData(x, psi_imag[frame])
            curve_prob.setData(x, psi_prob[frame])
            frame = (frame + 1) % n_times

        timer = pg.QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start(30)

        app.exec()
