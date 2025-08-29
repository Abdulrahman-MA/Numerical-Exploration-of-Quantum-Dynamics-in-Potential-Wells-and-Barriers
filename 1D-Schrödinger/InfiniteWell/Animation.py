from manim import *
import h5py
import numpy as np


class QuantumPsiWithProb(Scene):
    """
    A Manim scene that visualizes the quantum wavefunction ψ(x,t)
    (real and imaginary parts) alongside the probability density |ψ(x)|²
    for different eigenstates.

    Data is precomputed and loaded from an HDF5 file. The animation shows:
      - Time evolution of ψ(x,t) (real and imaginary parts).
      - The corresponding static probability density |ψ(x)|².
      - Smooth transitions between quantum eigenstates.

    Side-by-side layout:
      Left panel  : ψ(x,t) (time-dependent).
      Right panel : |ψ(x)|² (time-independent for each eigenstate).

    Notes
    -----
    - ψ_real, ψ_imag, and ψ_prob are assumed to be stored in the HDF5 file.
    - Frames are downsampled for performance.
    - The probability plot is synchronized with the ψ(x,t) animation.

    """

    def construct(self):
        # ----------------------------
        # 0. Load simulation data
        # ----------------------------
        with h5py.File("Data/Numerical/simulation_data.h5", "r") as f:
            x = f["x"][:]                       # spatial grid
            times = f["times"][:]               # time grid
            psi_real = f["psi_real"][:]         # shape: (n_states, n_times, N)
            psi_imag = f["psi_imag"][:]
            psi_prob = f["psi_prob"][:]         # probability density
            energies = f["energies"][:]         # eigenenergies

        n_states, n_times, N = psi_real.shape

        # ----------------------------
        # 1. Downsample for performance
        # ----------------------------
        skip_x = 3       # spatial resolution reduction
        skip_t = 5       # temporal resolution reduction
        x = x[::skip_x]
        psi_real = psi_real[:, :, ::skip_x]
        psi_imag = psi_imag[:, :, ::skip_x]
        psi_prob = psi_prob[:, 0, ::skip_x]     # time-independent slice
        N = len(x)

        # ----------------------------
        # 2. Title screen
        # ----------------------------
        title = Tex("Time Evolution of $\\psi(x,t)$ and $|\\psi(x)|^2$").move_to(ORIGIN)
        self.play(Write(title), run_time=2)
        self.wait(2)
        self.play(Uncreate(title))

        # ----------------------------
        # 3. Axes setup
        # ----------------------------
        # Left panel: ψ(x,t)
        axes_psi = Axes(
            x_range=[x.min(), x.max(), 0.5],
            y_range=[-1.2, 1.2, 0.5],
            axis_config={"include_numbers": False},
            tips=False,
        ).scale(0.5).to_edge(LEFT).shift(DOWN * 0.5)

        labels_psi = axes_psi.get_axis_labels(MathTex("x"), MathTex(r"\psi(x,t)"))

        # Right panel: |ψ|²
        axes_prob = Axes(
            x_range=[x.min(), x.max(), 0.5],
            y_range=[0, psi_prob.max() * 1.1, 0.1],
            axis_config={"include_numbers": False},
            tips=False,
        ).scale(0.5).to_edge(RIGHT).shift(DOWN * 0.5)

        labels_prob = axes_prob.get_axis_labels(MathTex("x"), MathTex(r"|\psi(x)|^2"))

        # ----------------------------
        # 4. Curve factory helper
        # ----------------------------
        def make_curve(yvals, axes, color=BLUE):
            """Return a smooth VMobject curve for given y-values in given axes."""
            points = [axes.c2p(x[i], yvals[i]) for i in range(len(x))]
            mob = VMobject(color=color, stroke_width=2)
            mob.set_points_smoothly(points)
            return mob

        # ----------------------------
        # 5. Initial curves (State 0)
        # ----------------------------
        curve_real = make_curve(psi_real[0, 0, :], axes_psi, color=BLUE)
        curve_imag = make_curve(psi_imag[0, 0, :], axes_psi, color=RED)
        curve_prob = make_curve(psi_prob[0, :], axes_prob, color=GREEN)


        label = Tex(f"State 0, E = {energies[0]:.3f}").to_edge(UP)
        

        # ----------------------------
        # 6. Legend for curves
        # ----------------------------
        def make_legend_item(color, text):
            marker = Square(side_length=0.2, fill_color=color, fill_opacity=1, stroke_width=0)
            label = MathTex(text).scale(0.6).next_to(marker, RIGHT, buff=0.2)
            return VGroup(marker, label)

        items = VGroup(
            make_legend_item(BLUE, r"\text{Re}(\psi)"),
            make_legend_item(RED, r"\text{Im}(\psi)"),
            make_legend_item(GREEN, r"|\psi|^2")
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).scale(0.9)

        # ----------------------------
        # Rectangle background
        # ----------------------------
        rect = Rectangle(
            width=items.width + 0.5,
            height=items.height + 0.5,
            stroke_color=WHITE,
            stroke_width=1.5,
            fill_color=BLACK,
            fill_opacity=0.15
        )
        rect.surround(items, buff=0.3)

        legend = VGroup(rect, items).to_corner(UL)

        self.play(
            Write(legend), Write(label),
            Write(axes_psi), Write(labels_psi),
            Write(axes_prob), Write(labels_prob),
            run_time=2
            )
        
        self.play(
            Create(curve_real), Create(curve_imag), 
            Create(curve_prob), run_time=2
            )
        # ----------------------------
        # 7. Animation parameters
        # ----------------------------
        seconds_per_timeblock = 6   # duration of ψ(x,t) evolution per state
        transition_duration = 2     # smooth transition to next state
        seconds_per_state_pause = 1 # pause after each state

        # ----------------------------
        # 8. Animate time evolution
        # ----------------------------
        for s in range(n_states):
            # Animate ψ(x,t) over time for current state
            for t in range(1, n_times, skip_t):
                new_real = make_curve(psi_real[s, t, :], axes_psi, color=BLUE)
                new_imag = make_curve(psi_imag[s, t, :], axes_psi, color=RED)

                self.play(
                    Transform(curve_real, new_real),
                    Transform(curve_imag, new_imag),
                    run_time=seconds_per_timeblock / (n_times / skip_t),
                    rate_func=linear
                )

            # Transition to next state (ψ and |ψ|²)
            if s < n_states - 1:
                next_real = make_curve(psi_real[s + 1, 0, :], axes_psi, color=BLUE)
                next_imag = make_curve(psi_imag[s + 1, 0, :], axes_psi, color=RED)
                next_prob = make_curve(psi_prob[s + 1, :], axes_prob, color=GREEN)
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

        # ----------------------------
        # 9. Outro cleanup
        # ----------------------------
        self.wait(2)
        self.play(
            Uncreate(curve_real),
            Uncreate(curve_imag),
            Uncreate(curve_prob),
            Uncreate(label),
            Uncreate(axes_psi), Uncreate(labels_psi),
            Uncreate(axes_prob), Uncreate(labels_prob),
            Uncreate(legend),
            run_time=3
        )
        self.wait(1)
