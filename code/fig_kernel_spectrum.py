"""Figure: hard structural relations and soft prior restrictions of a kernel.

On n = 40 equally spaced inputs in [-1, 1] this computes
  (a) the eigenvalues of the kernel matrix for an RBF kernel and for the
      polynomial kernel (1 + x x')^3, against the noise variance,
  (b) a leading and a small-eigenvalue eigenvector of the RBF kernel matrix,
  (c) the row of the residual matrix S = sigma^2 (K + sigma^2 I)^{-1}
      at a central input, next to the straight-line relation (-1/2, 1, -1/2),
and writes output/figures/fig_kernel_spectrum.pdf.

Run from any directory:  python code/fig_kernel_spectrum.py
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from gpcore import rbf

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES = os.path.join(HERE, "..", "output", "figures")

N = 40
LENGTHSCALE = 0.3
NOISE = 0.0025  # the true noise variance of the experiments
DEGREE = 3
CENTRE = N // 2

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})


def main():
    x = np.linspace(-1.0, 1.0, N)
    k_rbf = rbf(x, x, LENGTHSCALE, 1.0)
    k_poly = (1.0 + np.outer(x, x)) ** DEGREE

    lam_rbf, u_rbf = np.linalg.eigh(k_rbf)
    lam_rbf, u_rbf = lam_rbf[::-1], u_rbf[:, ::-1]
    lam_poly = np.linalg.eigvalsh(k_poly)[::-1]
    # Clip round-off at the level of machine precision relative to the largest eigenvalue.
    floor_rbf = np.finfo(float).eps * lam_rbf[0]
    floor_poly = np.finfo(float).eps * lam_poly[0]
    first_soft = int(np.argmax(lam_rbf < NOISE))  # first eigenvalue below the noise variance

    s_mat = NOISE * np.linalg.inv(k_rbf + NOISE * np.eye(N))
    row = s_mat[CENTRE] / s_mat[CENTRE, CENTRE]

    print(f"RBF: {np.sum(lam_rbf >= NOISE)} eigenvalues >= noise, first soft index {first_soft + 1}")
    print(f"polynomial: numerical rank {np.sum(lam_poly > 1e-10 * lam_poly[0])}")
    print("S row around centre:", np.round(row[CENTRE - 4:CENTRE + 5], 3))

    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.4))
    j = np.arange(1, N + 1)
    ax = axes[0]
    ax.semilogy(j, np.maximum(lam_rbf, floor_rbf), "o", ms=2.5, color="#2471a3",
                label=rf"RBF, $\ell={LENGTHSCALE}$")
    ax.semilogy(j, np.maximum(lam_poly, floor_poly), "s", ms=2.5, mfc="none", color="#b9770e",
                label="polynomial\n" + rf"$q={DEGREE}$")
    ax.axhline(NOISE, color="0.4", lw=0.8, ls="--")
    ax.text(N, NOISE * 3, r"$\sigma^2$", ha="right", va="bottom", fontsize=8, color="0.3")
    ax.set_xlabel("index $j$")
    ax.set_ylabel(r"eigenvalue $\lambda_j$")
    ax.set_title("(a) spectrum of $K$", fontsize=9)
    ax.text(17, 1e-5, rf"RBF, $\ell={LENGTHSCALE}$", fontsize=7, color="#2471a3")
    ax.text(1, 2e-12, "polynomial\n" + rf"$q={DEGREE}$", fontsize=7, color="#b9770e", va="bottom")

    ax = axes[1]
    ax.plot(x, u_rbf[:, 0], color="0.5", lw=1.0, label=r"$u_1$")
    ax.plot(x, u_rbf[:, first_soft], color="#2471a3", lw=1.0, marker="o", ms=2,
            label=rf"$u_{{{first_soft + 1}}}$")
    ax.axhline(0.0, color="0.8", lw=0.6)
    ax.set_xlabel("$x$")
    ax.set_title("(b) RBF eigenvectors", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2)

    ax = axes[2]
    offsets = np.arange(N) - CENTRE
    window = np.abs(offsets) <= 8
    ax.axhline(0.0, color="0.8", lw=0.6)
    ax.plot(offsets[window], row[window], color="#2471a3", lw=1.0, marker="o", ms=2.5,
            label=r"row of $S$")
    ax.plot([-1, 0, 1], [-0.5, 1.0, -0.5], "s", ms=4, mfc="none", color="#c0392b",
            label=r"$(-\frac{1}{2},1,-\frac{1}{2})$")
    ax.set_xlabel(r"offset from $x_i$ (inputs)")
    ax.set_title("(c) a soft relation", fontsize=9)
    ax.text(1.0, 0.9, r"row of $S$", fontsize=7, color="#2471a3")
    ax.text(1.6, -0.5, r"$(-\frac{1}{2},1,-\frac{1}{2})$", fontsize=7, color="#c0392b", va="center")

    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    out = os.path.join(FIGURES, "fig_kernel_spectrum.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", os.path.normpath(out))


if __name__ == "__main__":
    main()
