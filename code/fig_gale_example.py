"""Figure: Gale vectors of a planar model on the unit square plus its centre.

Computes the residual projector P_N = I - Phi (Phi^T Phi)^{-1} Phi^T, an
orthonormal Gale basis G of null(Phi^T), the Gale vectors g_i (rows of G)
and the leverages h_ii, checks the identities used in the paper, and writes
a TikZ figure that is compiled to output/figures/fig_gale_example.pdf.

Standard library only. Run from any directory:
    python code/fig_gale_example.py
"""

import math
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "output", "figures")
NAME = "fig_gale_example"

# Design: corners 1-4 of the unit square and the centre 5.
X = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.5, 0.5)]
PHI = [[1.0, x1, x2] for x1, x2 in X]  # affine feature map (1, x1, x2)

# The two consistency relations stated in the paper.
RELATIONS = [[1, -1, 1, -1, 0], [1, 1, 1, 1, -4]]


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
            for i in range(len(a))]


def transpose(a):
    return [list(r) for r in zip(*a)]


def inv3(m):
    (a, b, c), (d, e, f), (g, h, i) = m
    det = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    adj = [[e * i - f * h, c * h - b * i, b * f - c * e],
           [f * g - d * i, a * i - c * g, c * d - a * f],
           [d * h - e * g, b * g - a * h, a * e - b * d]]
    return [[v / det for v in row] for row in adj]


def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = [float(x) for x in v]
        for q in basis:
            proj = sum(wi * qi for wi, qi in zip(w, q))
            w = [wi - proj * qi for wi, qi in zip(w, q)]
        norm = math.sqrt(sum(wi * wi for wi in w))
        basis.append([wi / norm for wi in w])
    return basis


def close(a, b, tol=1e-12):
    return all(abs(x - y) < tol for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def compute():
    n = len(PHI)
    phit = transpose(PHI)
    hat = matmul(matmul(PHI, inv3(matmul(phit, PHI))), phit)
    p_n = [[(1.0 if i == j else 0.0) - hat[i][j] for j in range(n)] for i in range(n)]

    cols = gram_schmidt(RELATIONS)          # orthonormal basis of null(Phi^T)
    g = transpose(cols)                      # n x (n - r), rows are Gale vectors

    # Checks: relations annihilate the features, and G G^T equals P_N.
    assert close(matmul(phit, g), [[0.0, 0.0]] * 3)
    assert close(matmul(g, transpose(g)), p_n)
    leverage = [hat[i][i] for i in range(n)]
    return g, leverage, p_n


def tikz(g, leverage):
    labels = ["1", "2", "3", "4", "5"]
    s = 4.0  # cm per unit in the design panel
    t = 3.2  # cm per unit in the Gale panel
    lines = [
        r"\documentclass[tikz,border=2pt]{standalone}",
        r"\usepackage{lmodern,amssymb}",
        r"\begin{document}",
        r"\begin{tikzpicture}[font=\small]",
        # Panel (a): design and leverages.
        rf"\draw[gray!60] (0,0) rectangle ({s},{s});",
    ]
    anchors = ["north east", "north west", "south west", "south east", "north"]
    for (x1, x2), lab, h, anc in zip(X, labels, leverage, anchors):
        lines.append(rf"\fill ({s * x1:.4f},{s * x2:.4f}) circle (2.2pt);")
        lines.append(rf"\node[anchor={anc}, align=center] at ({s * x1:.4f},{s * x2:.4f}) "
                     rf"{{${lab}$ \scriptsize\color{{gray}}($h={h:.1f}$)}};")
    lines.append(rf"\node at ({s / 2},-0.9) {{(a) design in $\mathbb{{R}}^2$}};")

    # Panel (b): Gale vectors in R^2.
    ox = s + 5.0
    oy = s - 0.6
    lines.append(rf"\draw[gray!50, ->] ({ox - 2.0},{oy}) -- ({ox + 2.0},{oy});")
    lines.append(rf"\draw[gray!50, ->] ({ox},{oy - 3.1}) -- ({ox},{oy + 1.2});")
    seen = {}
    for lab, gi in zip(labels, g):
        key = (round(gi[0], 9), round(gi[1], 9))
        seen.setdefault(key, []).append(lab)
    for (a, b), labs in seen.items():
        x, y = ox + t * a, oy + t * b
        lines.append(rf"\draw[->, thick] ({ox},{oy}) -- ({x:.4f},{y:.4f});")
        norm2 = a * a + b * b
        anchor = "south" if b > 0 else "west"
        lines.append(rf"\node[anchor={anchor}] at ({x:.4f},{y:.4f}) "
                     rf"{{$g_{{{','.join(labs)}}}$, $\|g\|^2={norm2:.1f}$}};")
    lines.append(rf"\node at ({ox},-0.9) {{(b) Gale vectors in $\mathbb{{R}}^{{2}}$}};")
    lines += [r"\end{tikzpicture}", r"\end{document}"]
    return "\n".join(lines) + "\n"


def main():
    g, leverage, _ = compute()
    for i, (gi, h) in enumerate(zip(g, leverage), 1):
        print(f"point {i}: g = ({gi[0]: .4f}, {gi[1]: .4f}), |g|^2 = {gi[0]**2 + gi[1]**2:.3f}, h = {h:.3f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    build = os.path.join(HERE, "build")
    os.makedirs(build, exist_ok=True)
    tex = os.path.join(build, NAME + ".tex")
    with open(tex, "w", encoding="utf-8") as f:
        f.write(tikz(g, leverage))
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", NAME + ".tex"],
                   cwd=build, check=True, stdout=subprocess.DEVNULL)
    shutil.copy(os.path.join(build, NAME + ".pdf"), os.path.join(OUT_DIR, NAME + ".pdf"))
    print("wrote", os.path.normpath(os.path.join(OUT_DIR, NAME + ".pdf")))


if __name__ == "__main__":
    main()
