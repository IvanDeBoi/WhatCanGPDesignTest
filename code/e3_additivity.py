"""Experiment E3: testing additivity in two dimensions.

Null model: an additive GP on [-1, 1]^2 with kernel
    k(x, x') = k1(x_1, x_1') + k2(x_2, x_2'),
where k1 and k2 are RBF kernels with lengthscale 0.5 and unit output scale, zero
mean, and noise variance 0.0025. Alternative: an interaction delta(x) = c x_1 x_2.

Four kinds of design with n = 16 inputs are compared:
  grid      the 4 x 4 grid with coordinates -1, -1/3, 1/3, 1
  LHS       Latin hypercube samples (50 draws)
  random    uniformly random inputs (50 draws)
  power     chosen greedily, one input at a time, from a 21 x 21 candidate grid,
            each time maximising the noncentrality of the GP test against x_1 x_2
For each design this computes
  * the number of exact consistency relations, n - rank(K),
  * the power of the GP test (C = all observations, noncentral chi-square power) and of the
    exact residual test on null(K), against c x_1 x_2 for a range of c,
  * the structural and prior parts of the noncentrality,
and checks the predicted power on the grid and on one LHS design by simulation.

Writes code/results/E3.json. Run:  python code/e3_additivity.py
"""

import json
import os

import numpy as np
from scipy.stats import chi2, ncx2

from gpcore import rbf

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

N = 16
LENGTHSCALE = 0.5
NOISE = 0.0025
ALPHA = 0.05
DRAWS = 50
CURVATURES = np.linspace(0.0, 0.3, 31)
SIM_C = [0.1, 0.2]
SIMS = 4000
RANK_TOL = 1e-12


def kernel(xa, xb):
    return rbf(xa[:, 0], xb[:, 0], LENGTHSCALE, 1.0) + rbf(xa[:, 1], xb[:, 1], LENGTHSCALE, 1.0)


def interaction(x):
    return x[:, 0] * x[:, 1]


def analyse(x):
    """Exact relations, noncentralities per unit c^2, and their split."""
    k = kernel(x, x)
    lam, u = np.linalg.eigh(k)
    zero = lam < RANK_TOL * lam.max()
    lam = np.where(zero, 0.0, lam)
    z2 = (u.T @ interaction(x)) ** 2
    parts = z2 / (lam + NOISE)
    return {"relations": int(zero.sum()), "eta_gp": float(parts.sum()),
            "eta_structural": float(parts[zero].sum()), "eta_prior": float(parts[~zero].sum())}


def powers(info, n):
    crit_gp = chi2.ppf(1 - ALPHA, n)
    gp = [float(ncx2.sf(crit_gp, n, c ** 2 * info["eta_gp"])) for c in CURVATURES]
    r = info["relations"]
    if r == 0:
        res = [None] * len(CURVATURES)
    else:
        crit_res = chi2.ppf(1 - ALPHA, r)
        res = [float(ncx2.sf(crit_res, r, c ** 2 * info["eta_structural"])) for c in CURVATURES]
    return gp, res


def grid_design():
    g = np.linspace(-1.0, 1.0, 4)
    return np.array([(a, b) for a in g for b in g])


def lhs_design(rng):
    cols = [(rng.permutation(N) + rng.uniform(size=N)) / N for _ in range(2)]
    return 2.0 * np.column_stack(cols) - 1.0


def power_design():
    g = np.linspace(-1.0, 1.0, 21)
    cand = np.array([(a, b) for a in g for b in g])
    chosen = []
    for _ in range(N):
        best, best_eta = None, -1.0
        for j in range(len(cand)):
            if j in chosen:
                continue
            x = cand[chosen + [j]]
            k = kernel(x, x) + NOISE * np.eye(len(x))
            d = interaction(x)
            eta = float(d @ np.linalg.solve(k, d))
            if eta > best_eta + 1e-12:
                best, best_eta = j, eta
        chosen.append(best)
    return cand[chosen]


def simulate(x, rng):
    """Rejection rates of the GP test and the residual test under y ~ N(c delta, A)."""
    k = kernel(x, x)
    a = k + NOISE * np.eye(N)
    q = np.linalg.inv(a)
    lam, u = np.linalg.eigh(k)
    g = u[:, lam < RANK_TOL * lam.max()]
    chol = np.linalg.cholesky(a)
    out = {}
    for c in SIM_C:
        y = c * interaction(x)[None, :] + rng.standard_normal((SIMS, N)) @ chol.T
        t_gp = np.einsum("si,ij,sj->s", y, q, y)
        rej = {"gp": float(np.mean(t_gp > chi2.ppf(1 - ALPHA, N)))}
        if g.shape[1] > 0:
            t_res = np.sum((y @ g) ** 2, axis=1) / NOISE
            rej["residual"] = float(np.mean(t_res > chi2.ppf(1 - ALPHA, g.shape[1])))
        out[str(c)] = rej
    return out


def main():
    rng = np.random.default_rng(2026)
    designs = {"grid": [grid_design()], "LHS": [lhs_design(rng) for _ in range(DRAWS)],
               "random": [rng.uniform(-1.0, 1.0, size=(N, 2)) for _ in range(DRAWS)],
               "power": [power_design()]}
    out = {"settings": {"n": N, "lengthscale": LENGTHSCALE, "noise": NOISE, "alpha": ALPHA,
                        "draws": DRAWS, "curvatures": CURVATURES.tolist(), "sims": SIMS},
           "designs": {}}
    for name, xs in designs.items():
        infos = [analyse(x) for x in xs]
        pw = [powers(i, N) for i in infos]
        gp = np.mean([p[0] for p in pw], axis=0).tolist()
        res = None if any(p[1][0] is None for p in pw) else np.mean([p[1] for p in pw], axis=0).tolist()
        entry = {"x": xs[0].tolist(), "relations": [i["relations"] for i in infos],
                 "eta_gp": [i["eta_gp"] for i in infos],
                 "eta_structural": [i["eta_structural"] for i in infos],
                 "eta_prior": [i["eta_prior"] for i in infos],
                 "gp_power": gp, "res_power": res,
                 "first_gp_power": pw[0][0], "first_res_power": pw[0][1]}
        if name in ("grid", "LHS", "power"):
            entry["simulated"] = simulate(xs[0], rng)
        out["designs"][name] = entry
        print(f"{name:7s} relations {np.mean(entry['relations']):5.1f}  "
              f"eta/c^2 gp {np.mean(entry['eta_gp']):9.1f}  structural {np.mean(entry['eta_structural']):9.1f}  "
              f"prior {np.mean(entry['eta_prior']):7.1f}  power at c=0.1: {gp[10]:.3f}  c=0.2: {gp[20]:.3f}",
              flush=True)
        if "simulated" in entry:
            print("        simulated:", entry["simulated"], " predicted for this design at c=0.1, 0.2:",
                  [round(pw[0][0][i], 3) for i in (10, 20)], flush=True)
    print("power design:", np.round(designs["power"][0], 2).tolist())
    ratios = [np.linalg.eigvalsh(kernel(x, x)) for x in designs["random"] + designs["LHS"]]
    print("smallest eigenvalue / largest, LHS and random draws:", min(r.min() / r.max() for r in ratios))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "E3.json"), "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
