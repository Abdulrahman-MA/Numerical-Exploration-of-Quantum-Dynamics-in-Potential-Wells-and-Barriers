from manim import *
import h5py
import numpy as np

class WavePacketScattering(Scene):
    """
    Animates the 1D wave packet scattering simulation:
      - ψ(x,t) real and imaginary parts
      - Probability density |ψ(x,t)|^2
      - Optional square barrier visualization
    """
    def construct(self):
        # ----------------------------
        # 0. Load simulation data
        # ----------------------------
        h5file = "Data/Scattering/scattering_run.h5"
        with h5py.File(h5file, "r") as f:
            x = f["x"][:]
            psi_real = f["psi_real"][:]   # shape (n_times, N)
            psi_imag = f["psi_imag"][:]
            psi_prob = f["psi_prob"][:]
            V = f["V"][:]
            times = f["times"][:]

        n_times, N = psi_real.shape

        # ----------------------------
        # 1. Downsample for performance
        # ----------------------------
        skip_x = max(1, N // 500)
        skip_t = max(1, n_times // 300)
        x = x[::skip_x]
        psi_real = psi_real[::skip_t, ::skip_x]
        psi_imag = psi_imag[::skip_t, ::skip_x]
        psi_prob = psi_prob[::skip_t, ::skip_x]
        V = V[::skip_x]
        n_times, N = psi_real.shape

        # ----------------------------
        # 2. Axes setup
        # ----------------------------
        # Left panel: ψ(x,t)
        axes_psi = Axes(
            x_range=[x.min(), x.max(), (x.max()-x.min())/8],
            y_range=[-1.2, 1.2, 0.5],
            axis_config={"include_numbers": False},
            tips=False,
        ).scale(0.5).to_edge(LEFT).shift(DOWN*0.5)
        labels_psi = axes_psi.get_axis_labels(MathTex("x"), MathTex(r"\psi(x,t)"))

        # Right panel: |ψ|²
        axes_prob = Axes(
            x_range=[x.min(), x.max(), (x.max()-x.min())/8],
            y_range=[0, psi_prob.max()*1.1, 0.1],
            axis_config={"include_numbers": False},
            tips=False,
        ).scale(0.5).to_edge(RIGHT).shift(DOWN*0.5)
        labels_prob = axes_prob.get_axis_labels(MathTex("x"), MathTex(r"|\psi(x,t)|^2"))

        # Barrier plot (overlay on |ψ|² for reference)
        barrier = V / V.max() * axes_prob.y_range[1] * 0.9
        barrier_curve = VMobject(color=YELLOW)
        barrier_curve.set_points_smoothly([axes_prob.c2p(x[i], barrier[i]) for i in range(N)])

        # ----------------------------
        # 3. Curve factory
        # ----------------------------
        def make_curve(yvals, axes, color=BLUE):
            points = [axes.c2p(x[i], yvals[i]) for i in range(len(x))]
            mob = VMobject(color=color, stroke_width=2)
            mob.set_points_smoothly(points)
            return mob

        # Initial curves
        curve_real = make_curve(psi_real[0], axes_psi, BLUE)
        curve_imag = make_curve(psi_imag[0], axes_psi, RED)
        curve_prob = make_curve(psi_prob[0], axes_prob, GREEN)

        # Labels
        label_time = Tex(f"t = {times[0]:.2f}").to_edge(UP)

        # ----------------------------
        # 4. Draw axes, curves, barrier, label
        # ----------------------------
        self.play(
            Write(axes_psi), Write(labels_psi),
            Write(axes_prob), Write(labels_prob),
            Create(curve_real), Create(curve_imag),
            Create(curve_prob), Create(barrier_curve),
            Write(label_time),
            run_time=3
        )

        # ----------------------------
        # 5. Animate time evolution
        # ----------------------------
        for t_idx in range(1, n_times):
            new_real = make_curve(psi_real[t_idx], axes_psi, BLUE)
            new_imag = make_curve(psi_imag[t_idx], axes_psi, RED)
            new_prob = make_curve(psi_prob[t_idx], axes_prob, GREEN)
            new_label = Tex(f"t = {times[t_idx]:.2f}").to_edge(UP)

            self.play(
                Transform(curve_real, new_real),
                Transform(curve_imag, new_imag),
                Transform(curve_prob, new_prob),
                Transform(label_time, new_label),
                run_time=0.03,  # adjust speed
                rate_func=linear
            )

        # ----------------------------
        # 6. Outro
        # ----------------------------
        self.wait(2)
        self.play(
            Uncreate(curve_real),
            Uncreate(curve_imag),
            Uncreate(curve_prob),
            Uncreate(barrier_curve),
            Uncreate(label_time),
            Uncreate(axes_psi), Uncreate(labels_psi),
            Uncreate(axes_prob), Uncreate(labels_prob),
            run_time=3
        )
