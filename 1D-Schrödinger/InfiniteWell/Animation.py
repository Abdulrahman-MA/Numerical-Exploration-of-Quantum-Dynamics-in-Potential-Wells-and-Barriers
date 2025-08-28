from manim import *
import h5py
import numpy as np


class QuantumPsiWithProb(Scene):
    def construct(self):
        # ----------------------------
        # 0. Load data
        # ----------------------------
        with h5py.File("Data/Numerical/simulation_data.h5", "r") as f:
            x = f["x"][:]
            times = f["times"][:]
            psi_real = f["psi_real"][:]  # (n_states, n_times, N)
            psi_imag = f["psi_imag"][:]
            psi_prob = f["psi_prob"][:]  # (n_states, n_times, N)
            energies = f["energies"][:]

        n_states, n_times, N = psi_real.shape

        # Downsample
        skip_x = 3
        skip_t = 5
        x = x[::skip_x]
        psi_real = psi_real[:, :, ::skip_x]
        psi_imag = psi_imag[:, :, ::skip_x]
        psi_prob = psi_prob[:, 0, ::skip_x]   # time-independent slice
        N = len(x)

        # ----------------------------
        # 1. Title
        # ----------------------------

        title = Tex("Time Evolution of $\\psi(x,t)$ and $|\\psi(x)|^2$").move_to(ORIGIN)
        self.play(Write(title), run_time=2)
        self.wait(2)
        self.play(Uncreate(title))

        # ----------------------------
        # 2. Axes
        # ----------------------------
        # Left: ψ(x,t)
        axes_psi = Axes(
            x_range=[x.min(), x.max(), 0.5],
            y_range=[-1.2, 1.2, 0.5],
            axis_config={"include_numbers": False},
            tips=False,
        ).scale(0.5).to_edge(LEFT).shift(DOWN*0.5)

        labels_psi = axes_psi.get_axis_labels(MathTex("x"), MathTex(r"\psi(x,t)"))

        # Right: |ψ|²
        axes_prob = Axes(
            x_range=[x.min(), x.max(), 0.5],
            y_range=[0, psi_prob.max()*1.1, 0.1],
            axis_config={"include_numbers": False},
            tips=False,
        ).scale(0.5).to_edge(RIGHT).shift(DOWN*0.5)

        labels_prob = axes_prob.get_axis_labels(MathTex("x"), MathTex(r"|\psi(x)|^2"))

        self.play(
            Write(axes_psi), Write(labels_psi),
            Write(axes_prob), Write(labels_prob),
            run_time=2
        )

        # ----------------------------
        # 3. Curve factories
        # ----------------------------
        def make_curve(yvals, axes, color=BLUE):
            points = [axes.c2p(x[i], yvals[i]) for i in range(len(x))]
            mob = VMobject(color=color, stroke_width=2)
            mob.set_points_smoothly(points)
            return mob

        # ----------------------------
        # 4. Initial curves
        # ----------------------------
        curve_real = make_curve(psi_real[0, 0, :], axes_psi, color=BLUE)
        curve_imag = make_curve(psi_imag[0, 0, :], axes_psi, color=RED)
        curve_prob = make_curve(psi_prob[0, :], axes_prob, color=GREEN)

        self.play(Create(curve_real), Create(curve_imag), Create(curve_prob), run_time=2)

        label = Tex(f"State 0, E = {energies[0]:.3f}").to_edge(UP)
        self.play(Write(label))

        # ----------------------------
        # 5. Animate
        # ----------------------------
        seconds_per_timeblock = 6  # duration of evolution for each state
        transition_duration = 2    # smooth transition between states
        seconds_per_state_pause = 1

        for s in range(n_states):
            # time evolution of ψ(x,t)
            for t in range(1, n_times, skip_t):
                new_real = make_curve(psi_real[s, t, :], axes_psi, color=BLUE)
                new_imag = make_curve(psi_imag[s, t, :], axes_psi, color=RED)

                self.play(
                    Transform(curve_real, new_real),
                    Transform(curve_imag, new_imag),
                    run_time=seconds_per_timeblock / (n_times/skip_t),
                    rate_func=linear
                )

            # transition to next state
            if s < n_states - 1:
                next_real = make_curve(psi_real[s+1, 0, :], axes_psi, color=BLUE)
                next_imag = make_curve(psi_imag[s+1, 0, :], axes_psi, color=RED)
                next_prob = make_curve(psi_prob[s+1, :], axes_prob, color=GREEN)
                next_label = Tex(f"State {s+1}, E = {energies[s+1]:.3f}").to_edge(UP)

                self.play(
                    Transform(curve_real, next_real),
                    Transform(curve_imag, next_imag),
                    Transform(curve_prob, next_prob),
                    Transform(label, next_label),
                    run_time=transition_duration,
                    rate_func=linear
                )
                self.wait(seconds_per_state_pause)

        self.wait(2)
        self.play(
            Uncreate(curve_real),
            Uncreate(curve_imag),
            Uncreate(curve_prob),
            Uncreate(label),
            Uncreate(axes_psi),
            Uncreate(labels_psi),
            Uncreate(axes_prob),
            Uncreate(labels_prob),
            run_time=3
        )
        self.wait(1)
