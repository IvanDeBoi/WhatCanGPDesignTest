"""Figure: structural and prior-based detection.

(a) A schematic of the split of a departure delta into the part that the model
    can represent, col(Phi), and the part that violates consistency relations,
    null(Phi^T) = col(G).
(b) The noncentrality eta = delta^T (K + sigma0^2 I)^{-1} delta of the test of all
    observations against a unit curvature delta(x) = x^2, under the straight-line
    kernel k(x, x') = 1 + x x', as a function of the noise standard deviation
    sigma0, for the two designs of experiment E1(c): 20 inputs at each end of
    [-1, 1], and 18 at each end with 4 at the centre. Computed exactly; no
    simulation is involved.

Writes output/figures/fig_schematic.pdf.
Run from any directory:  python code/fig_schematic.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "..", "output", "figures")

NOISE_SD = 0.05  # the noise standard deviation of experiment E1
DESIGNS = {
    "two ends": np.repeat([-1.0, 1.0], 20),
    "two ends + centre": np.concatenate([np.repeat([-1.0, 1.0], 18), np.zeros(4)]),
}
COLOURS = {"two ends": "#c0392b", "two ends + centre": "#17a589"}

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})


def noncentrality(x, sd):
    k = 1.0 + np.outer(x, x)
    delta = x ** 2
    return float(delta @ np.linalg.solve(k + sd ** 2 * np.eye(len(x)), delta))


def split(x, sd):
    """Structural and prior parts of eta, along the null space and the range of K."""
    k = 1.0 + np.outer(x, x)
    lam, u = np.linalg.eigh(k)
    z2 = (u.T @ (x ** 2)) ** 2
    zero = lam < 1e-9 * lam.max()
    lam = np.where(zero, 0.0, lam)
    parts = z2 / (lam + sd ** 2)
    return float(parts[zero].sum()), float(parts[~zero].sum())


def schematic(ax):
    ax.set_xlim(-0.15, 1.35)
    ax.set_ylim(-0.12, 1.12)
    ax.set_aspect("equal")
    ax.axis("off")
    arrow = dict(arrowstyle="-|>", lw=1.0, color="0.25", shrinkA=0, shrinkB=0)
    ax.annotate("", xy=(1.25, 0), xytext=(0, 0), arrowprops=arrow)
    ax.annotate("", xy=(0, 1.05), xytext=(0, 0), arrowprops=arrow)
    ax.text(1.25, -0.07, r"$\mathrm{col}(\Phi)$: representable", ha="right", va="top", fontsize=8)
    ax.text(0.03, 1.05, r"$\mathrm{null}(\Phi^\top)=\mathrm{col}(G)$" "\n" "consistency relations",
            ha="left", va="top", fontsize=8)
    d = np.array([0.95, 0.6])
    ax.annotate("", xy=d, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", lw=1.6, color="k"))
    ax.text(d[0] + 0.03, d[1] + 0.02, r"$\delta$", fontsize=10)
    ax.plot([d[0], d[0]], [0, d[1]], color="0.5", lw=0.7, ls=":")
    ax.plot([0, d[0]], [d[1], d[1]], color="0.5", lw=0.7, ls=":")
    ax.annotate("", xy=(d[0], 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#c0392b"))
    ax.annotate("", xy=(0, d[1]), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", lw=2.2, color="#17a589"))
    ax.text(d[0] / 2, 0.04, "prior-based\nbounded as " r"$\sigma_0\to0$", color="#c0392b",
            ha="center", va="bottom", fontsize=7.5)
    ax.text(-0.03, d[1] / 2, "structural\n" r"$\propto1/\sigma_0^2$", color="#17a589",
            ha="right", va="center", fontsize=7.5, rotation=90)
    ax.set_title("(a) two parts of a departure", fontsize=9)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw={"width_ratios": [1, 1.15]})
    schematic(axes[0])
    ax = axes[1]
    sds = np.logspace(-3, 0, 121)
    for name, x in DESIGNS.items():
        eta = [noncentrality(x, s) for s in sds]
        ax.loglog(sds, eta, color=COLOURS[name], lw=1.2, label=name)
        struct, prior = split(x, NOISE_SD)
        print(f"{name}: eta at sigma0={NOISE_SD}: structural {struct:.4g}, prior {prior:.4g}")
    ax.axvline(NOISE_SD, color="0.6", lw=0.7, ls=":")
    ax.text(NOISE_SD * 1.1, 2e5, r"$\sigma_0$ of E1", fontsize=7, color="0.4")
    ax.set_xlabel(r"noise standard deviation $\sigma_0$")
    ax.set_ylabel(r"noncentrality $\eta$ for $\delta=x^2$")
    ax.set_title("(b) straight line against curvature", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(os.path.join(FIGURES, "fig_schematic.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
