"""
Wave Packet Scattering Simulator (Numerical, Split-Operator FFT)
================================================================

- Uniform 1D grid with periodic FFT propagation and edge absorbers.
- Gaussian wave packet initialized at x0 with mean momentum k0.
- Potential: square barrier (configurable width/height/center), or custom.
- Outputs time series of psi (real/imag), |psi|^2, and R/T vs time.

Units: ħ = 1, m = 1.

Author: Abdelrahman Mohamed
Date: 2025-08
"""

from __future__ import annotations
import os
from pathlib import Path

import h5py
import numpy as np
from numpy.fft import fft, ifft, fftfreq
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Potentials
# -----------------------------------------------------------------------------
def square_barrier_potential(x: np.ndarray, center: float = 0.0,
                             width: float = 0.2, height: float = 10.0) -> np.ndarray:
    """Square barrier of given width/height centered at `center`."""
    half = width / 2.0
    V = np.zeros_like(x)
    V[(x >= center - half) & (x <= center + half)] = height
    return V


# -----------------------------------------------------------------------------
# Absorber (mask) to avoid spurious reflections from box edges
# -----------------------------------------------------------------------------
def edge_absorber(x: np.ndarray, xmin: float, xmax: float,
                  thickness: float = 0.2, power: float = 4.0) -> np.ndarray:
    """
    Smooth multiplicative mask M(x) in [0,1] applied each step near edges.
    - thickness: physical thickness of each absorbing layer at both ends
    - power: controls steepness (>=2 typical). Larger -> steeper absorption.
    """
    L = xmax - xmin
    left_start = xmin + thickness
    right_start = xmax - thickness

    M = np.ones_like(x)

    # Left edge region
    left_mask = x < left_start
    if np.any(left_mask):
        # normalized distance to left boundary in [0,1]
        d = (x[left_mask] - xmin) / thickness
        # go to 0 smoothly at boundary (x=xmin)
        M[left_mask] = np.exp(-((1 - d) ** power))

    # Right edge region
    right_mask = x > right_start
    if np.any(right_mask):
        d = (xmax - x[right_mask]) / thickness
        M[right_mask] = np.exp(-((1 - d) ** power))

    return M


# -----------------------------------------------------------------------------
# Simulator
# -----------------------------------------------------------------------------
class WavePacketScatteringSimulator:
    """
    Numerical 1D scattering using split-operator method.

    Grid/Time
    ---------
    xmin, xmax : domain limits
    N         : number of grid points (use power-of-two-ish for FFT speed)
    dt        : time step
    steps     : number of time steps to propagate

    Initial packet
    --------------
    x0     : initial center
    sigma  : spatial width
    k0     : mean wave number (controls kinetic energy E = k0^2 / 2)

    Potential
    ---------
    Pass a callable V_func(x) -> V(x), or use the default square barrier via
    barrier_* parameters.

    Absorber
    --------
    absorber_thickness, absorber_power : tuned to reduce edge reflections.

    Outputs
    -------
    - HDF5 with x, times, V(x), psi_real(t,x), psi_imag(t,x), psi_prob(t,x),
      R_t, T_t, and final R, T.
    """

    def __init__(
        self,
        xmin: float = -10.0,
        xmax: float = 10.0,
        N: int = 4096,
        dt: float = 0.002,
        steps: int = 6000,
        # initial packet
        x0: float = -5.0,
        sigma: float = 0.5,
        k0: float = 8.0,
        # potential (square barrier by default)
        V_center: float = 0.0,
        V_width: float = 0.5,
        V_height: float = 10.0,
        V_func=None,  # optional custom potential: def V_func(x)->array
        # absorber at edges
        absorber_thickness: float = 0.0,
        absorber_power: float = 4.0,
        # saving
        base_dir: str | Path = "Data/Scattering",
        save_every: int = 10,   # downsample in time when storing
    ):
        self.xmin, self.xmax, self.N = xmin, xmax, N
        self.dt, self.steps = dt, steps
        self.x = np.linspace(xmin, xmax, N, endpoint=False)  # periodic grid for FFT
        self.dx = self.x[1] - self.x[0]
        self.times = np.arange(steps) * dt

        # k-grid for kinetic operator (FFT)
        L = xmax - xmin
        self.k = 2.0 * np.pi * fftfreq(N, d=self.dx)  # assumes ħ=1
        self.Tk = 0.5 * (self.k ** 2)                # kinetic energy operator

        # potential
        if V_func is None:
            self.V = square_barrier_potential(self.x, center=V_center,
                                              width=V_width, height=V_height)
        else:
            self.V = V_func(self.x)

        self.abs_thickness = absorber_thickness  # store it for diagnostics/R/T
        # absorber
        self.abs_mask = np.ones_like(self.x)


        # initial packet
        self.x0 = x0
        self.sigma = sigma
        self.k0 = k0
        self.psi0 = self._gaussian_packet(self.x, x0, sigma, k0)

        # precompute split-operator exponentials
        self._build_split_operators()

        # saving
        self.base_dir = Path(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)
        self.save_every = max(1, int(save_every))

        # where to split space for R/T (relative to barrier center by default)
        self.split_x = V_center

    # -------------------------------------------------------------------------
    @staticmethod
    def _gaussian_packet(x, x0, sigma, k0):
        """Normalized Gaussian wave packet with mean momentum k0."""
        psi = np.exp(-(x - x0) ** 2 / (4.0 * sigma ** 2)) * np.exp(1j * k0 * x)
        # normalize
        norm = np.sqrt(np.trapezoid(np.abs(psi) ** 2, x))
        return psi / norm

    def _build_split_operators(self):
        """Precompute exp(-i V dt/2) in x-space and exp(-i T dt) in k-space."""
        self.expV_half = np.exp(-0.5j * self.V * self.dt)
        self.expT_full = np.exp(-1j * self.Tk * self.dt)

    # -------------------------------------------------------------------------
    def run(self, hdf5_path: str | Path = None):
        """Propagate and save HDF5 with fields and R/T curves."""
        if hdf5_path is None:
            hdf5_path = self.base_dir / "scattering_run.h5"

        # allocate storage (downsampled in time)
        keep_idxs = np.arange(0, self.steps, self.save_every, dtype=int)
        n_keep = len(keep_idxs)

        psi_real_save = np.zeros((n_keep, self.N), dtype=np.float64)
        psi_imag_save = np.zeros((n_keep, self.N), dtype=np.float64)
        psi_prob_save = np.zeros((n_keep, self.N), dtype=np.float64)
        times_save = self.times[keep_idxs]
        R_t = np.zeros(n_keep, dtype=np.float64)
        T_t = np.zeros(n_keep, dtype=np.float64)

        # set initial state
        psi = self.psi0.copy()

        keep_ptr = 0


        for n in tqdm(range(self.steps), desc="Propagating", unit="step"):
            # Split-operator step: V/2 -> T -> V/2
            psi *= self.expV_half
            psi_k = fft(psi)
            psi_k *= self.expT_full
            psi = ifft(psi_k)
            psi *= self.expV_half

            # absorber near edges
            psi *= self.abs_mask

            # Save downsampled snapshots and R/T
            if n % self.save_every == 0:
                prob = np.abs(psi) ** 2
                psi_real_save[keep_ptr, :] = np.real(psi)
                psi_imag_save[keep_ptr, :] = np.imag(psi)
                psi_prob_save[keep_ptr, :] = prob

                # --- compute safe measurement windows away from barrier & absorber ---

                left_mask = self.x < self.split_x
                right_mask = self.x >= self.split_x

                # fallback: if masks are empty (too aggressive margin), relax margin
                if not np.any(left_mask):
                    left_mask = (self.x < (self.split_x - 0.1)) & (self.x > self.xmin + self.abs_thickness + 1e-9)
                if not np.any(right_mask):
                    right_mask = (self.x > (self.split_x + 0.1)) & (self.x < self.xmax - self.abs_thickness - 1e-9)

                # final fallback: if still empty, use simple left/right half (last resort)
                if not np.any(left_mask):
                    left_mask = self.x < self.split_x
                if not np.any(right_mask):
                    right_mask = self.x > self.split_x

                # reflection/transmission (integrate in safe regions)
                R_t[keep_ptr] = np.trapezoid(prob[left_mask], self.x[left_mask])
                T_t[keep_ptr] = np.trapezoid(prob[right_mask], self.x[right_mask])

                keep_ptr += 1

        # final R/T (use last saved point)
        R_final = R_t[-1]
        T_final = T_t[-1]

        # write HDF5
        with h5py.File(hdf5_path, "w") as f:
            f.create_dataset("x", data=self.x)
            f.create_dataset("times", data=times_save)
            f.create_dataset("V", data=self.V)
            f.create_dataset("psi_real", data=psi_real_save, compression="gzip")
            f.create_dataset("psi_imag", data=psi_imag_save, compression="gzip")
            f.create_dataset("psi_prob", data=psi_prob_save, compression="gzip")
            f.create_dataset("R_t", data=R_t)
            f.create_dataset("T_t", data=T_t)
            f.create_dataset("R_final", data=np.array([R_final]))
            f.create_dataset("T_final", data=np.array([T_final]))

            # metadata
            f.attrs["dt"] = self.dt
            f.attrs["steps"] = self.steps
            f.attrs["save_every"] = self.save_every
            f.attrs["x0"] = self.x0
            f.attrs["sigma"] = self.sigma
            f.attrs["k0"] = self.k0
            f.attrs["split_x"] = self.split_x
            f.attrs["abs_thickness"] = self.abs_thickness

        print(f"\nSaved scattering results to: {hdf5_path}")
        print(f"Final R ≈ {R_final:.4f}, T ≈ {T_final:.4f}, R+T ≈ {R_final+T_final:.4f}")


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    sim = WavePacketScatteringSimulator(
        xmin=-12.0, xmax=12.0, N=4096,
        dt=0.01, steps=8000,
        x0=-6.0, sigma=0.6, k0=6.0,
        V_center=0.0, V_width=0.6, V_height=8.0,
        absorber_thickness=0.0, absorber_power=4.0,
        base_dir="Data/Scattering",
        save_every=1,
    )
    sim.run()
