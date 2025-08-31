from manim import *
import h5py
import numpy as np

class WavePacketScattering(Scene):
    """
    Quantum Scattering Simulator (Split-Operator Method)

    Visualizes the time evolution of a Gaussian wave packet ψ(x,t) scattering
    from a finite potential barrier/well.

    Features:
    - Real and imaginary parts of ψ(x,t) in the top row (separate plots).
    - Probability density |ψ|² spanning the full width below.
    - Potential barrier overlay on |ψ|² plot.
    - Title card intro, legend, and professional outro.
    """

    def construct(self):
        # ----------------------------
        # 0. Load precomputed data
        # ----------------------------
        h5file = "Data/Scattering/scattering_run.h5"
        with h5py.File(h5file, "r") as f:
            x = f["x"][:]
            psi_real = f["psi_real"][:]
            psi_imag = f["psi_imag"][:]
            psi_prob = f["psi_prob"][:]
            V = f["V"][:]
            times = f["times"][:]

        n_times, N = psi_real.shape

        # Downsample for performance
        skip_x = max(1, N // 500)
        skip_t = max(1, n_times // 300)
        x = x[::skip_x]
        psi_real = psi_real[::skip_t, ::skip_x]
        psi_imag = psi_imag[::skip_t, ::skip_x]
        psi_prob = psi_prob[::skip_t, ::skip_x]
        V = V[::skip_x]
        n_times, N = psi_real.shape

        # ----------------------------
        # 1. Title card intro
        # ----------------------------
        title = VGroup(
            Tex("Quantum Scattering Simulator"),
            Tex("Split-Operator Fourier Method").scale(0.7)
        ).arrange(DOWN).move_to(ORIGIN)

        self.play(Write(title), run_time=2)
        self.wait(2)
        self.play(FadeOut(title))

        # ----------------------------
        # 2. Axes setup (grid layout)
        # ----------------------------
        # Top row: Re(ψ), Im(ψ)
        axes_real = Axes(
            x_range=[x.min(), x.max(), (x.max()-x.min())/8],
            y_range=[-1.2, 1.2, 0.5],
            tips=False
        ).scale(0.45).to_edge(LEFT).shift(UP*1.5)

        axes_imag = axes_real.copy().next_to(axes_real, RIGHT, buff=1.0)

        label_real = axes_real.get_axis_labels(MathTex("x"), MathTex(r"\Re(\psi)"))
        label_imag = axes_imag.get_axis_labels(MathTex("x"), MathTex(r"\Im(\psi)"))

        # Bottom row: |ψ|² full width
        axes_prob = Axes(
            x_range=[x.min(), x.max(), (x.max()-x.min())/8],
            y_range=[0, psi_prob.max()*1.1, 0.1],
            tips=False
        ).scale(0.6).to_edge(DOWN)

        label_prob = axes_prob.get_axis_labels(MathTex("x"), MathTex(r"|\psi(x,t)|^2"))

        # ----------------------------
        # Barrier overlay (step-like, aligned edges)
        # ----------------------------
        def make_barrier(L, V_inside, V_outside, axes):
            # Explicit edges
            x_edges = [-L/2, L/2]

            x_plot = np.array([
                x.min(),
                x_edges[0], x_edges[0],
                x_edges[1], x_edges[1],
                x.max()
            ])
            V_plot = np.array([
                V_outside,
                V_outside, V_inside,
                V_inside, V_outside,
                V_outside
            ])

            # Scale into probability plot range
            V_scaled = V_plot / V_plot.max() * axes.y_range[1] * 0.9
            barrier_curve = VMobject(color=YELLOW, stroke_width=2)
            barrier_curve.set_points_as_corners([axes.c2p(xx, yy) for xx, yy in zip(x_plot, V_scaled)])
            return barrier_curve

        # Replace old barrier
        L = 10.0  # or pull from metadata / parameters in HDF5
        V_inside, V_outside = 0.0, 10.0
        barrier_curve = make_barrier(L, V_inside, V_outside, axes_prob)


        # ----------------------------
        # 3. Helper to make curves
        # ----------------------------
        def make_curve(yvals, axes, color=BLUE):
            points = [axes.c2p(x[i], yvals[i]) for i in range(len(x))]
            mob = VMobject(color=color, stroke_width=2)
            mob.set_points_smoothly(points)
            return mob

        # Initial curves
        curve_real = make_curve(psi_real[0], axes_real, GREEN)
        curve_imag = make_curve(psi_imag[0], axes_imag, RED)
        curve_prob = make_curve(psi_prob[0], axes_prob, BLUE )

        # ----------------------------
        # 4. Legend
        # ----------------------------
        def make_legend_item(color, text):
            marker = Square(side_length=0.2, fill_color=color, fill_opacity=1, stroke_width=0)
            label = MathTex(text).scale(0.6).next_to(marker, RIGHT, buff=0.2)
            return VGroup(marker, label)

        legend = VGroup(
            make_legend_item(GREEN , r"\Re(\psi)"),
            make_legend_item(RED, r"\Im(\psi)"),
            make_legend_item(BLUE, r"|\psi(x,t)|^2"),
            make_legend_item(YELLOW, "V(x)")
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3).scale(0.9)

        legend_rect = Rectangle(
            width=legend.width + 0.5,
            height=legend.height + 0.5,
            stroke_color=WHITE,
            fill_color=BLACK, fill_opacity=0.15
        )
        legend_rect.surround(legend)
        legend_group = VGroup(legend_rect, legend).to_corner(DL)

        # ----------------------------
        # Time label (updated dynamically)
        # ----------------------------
        time_label = MathTex(r"t = 0").scale(0.7).to_corner(UR).shift(LEFT*0.5 + DOWN*0.5)

        # ----------------------------
        # 5. Draw everything
        # ----------------------------
        self.play(
            Write(axes_real), Write(label_real),
            Write(axes_imag), Write(label_imag),
            Write(axes_prob), Write(label_prob),
            Create(barrier_curve),
            Write(legend_group),
            Create(curve_real), Create(curve_imag), Create(curve_prob),
            run_time=3
        )

        # ----------------------------
        # 6. Animate evolution
        # ----------------------------
        seconds_per_timeblock = 1   # Time Control Parameter


        for t in range(1, n_times):
            new_real = make_curve(psi_real[t], axes_real, GREEN)
            new_imag = make_curve(psi_imag[t], axes_imag, RED)
            new_prob = make_curve(psi_prob[t], axes_prob, BLUE)

            new_time_label = MathTex(
                fr"t = {times[t]:.2f}"
            ).scale(0.7).to_corner(UR).shift(LEFT*0.5 + DOWN*0.5)

            self.play(
                Transform(curve_real, new_real),
                Transform(curve_imag, new_imag),
                Transform(curve_prob, new_prob),
                Transform(time_label, new_time_label),
                run_time=seconds_per_timeblock / n_times,
                rate_func=linear
            )

        # ----------------------------
        # 7. Outro
        # ----------------------------
        self.wait(2)
        self.play(
            *[FadeOut(mob) for mob in
              [curve_real, curve_imag, curve_prob, barrier_curve,
               axes_real, axes_imag, axes_prob, label_real, label_imag, label_prob,
               legend_group]],
            run_time=3
        )
        self.wait(1)
