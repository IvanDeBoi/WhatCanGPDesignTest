"""Experiment E1: the power of a group test depends on the design.

Fully specified GP null: RBF kernel with lengthscale 0.3 and unit output
scale, noise variance 0.0025, zero mean. The group test T_C is
then exactly chi-square under the null, and noncentral chi-square with
noncentrality (Q delta)_C^T Q_CC^{-1} (Q delta)_C under a discrepancy delta
(noncentral chi-square power of the group test). For delta supported on C this is
delta_C^T Q_CC delta_C.

Part (a): one design of 40 inputs (36 uniform, 4 in a tight cluster). A
common offset c is planted on four kinds of group: the observation at the
right boundary, an interior observation, one member of the cluster, and the
whole cluster. Predicted power is compared with Monte Carlo rejection rates.

Part (b): a localised bump of height c and width w at a specified location
x0, tested with the group of observations within r of x0. Each candidate new
input is scored by the predicted power after adding it, and by the current
predictive variance. The two choices are compared for one design in detail,
and over 20 designs and 19 bump locations each.

Run:  python code/e1_power.py                   (everything)
      python code/e1_power.py --examples-only   (only the illustration)
      python code/e1_power.py --part-c-only     (only part (c))
Writes code/results/E1.json.
"""

import json
import os
import time

import numpy as np
from scipy.stats import chi2, ncx2

from gpcore import rbf

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

N = 40
NOISE_VAR = 0.0025
LENGTHSCALE = 0.3
SCALE = 1.0
ALPHA = 0.05
MC_DRAWS = 4000

OFFSETS = np.linspace(0.0, 0.3, 13)          # part (a)
BUMP_HEIGHT = 0.2                            # part (b)
BUMP_WIDTH = 0.04
WINDOW = 0.1
EXAMPLE_X0 = 0.10
CANDIDATES = np.linspace(-1.0, 1.0, 401)
SWEEP_X0 = np.linspace(-0.9, 0.9, 19)
SWEEP_DESIGNS = range(100, 120)


def make_design(seed):
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.uniform(-1, 1, N - 4), 0.2 + rng.uniform(-0.01, 0.01, 4)])
    return np.sort(x)


def precision(x):
    return np.linalg.inv(rbf(x, x, LENGTHSCALE, SCALE) + NOISE_VAR * np.eye(len(x)))


def noncentrality(q, group, delta):
    """Noncentrality of T_C for a discrepancy delta on all observations (noncentral chi-square power of the group test)."""
    w = (q @ delta)[group]
    return float(w @ np.linalg.solve(q[np.ix_(group, group)], w))


def power(eta, k):
    if k == 0:
        return 0.0
    return float(1.0 - ncx2.cdf(chi2.ppf(1.0 - ALPHA, k), k, eta)) if eta > 0 else ALPHA


def monte_carlo_power(x, group, delta_full, rng):
    """Empirical rejection rate of T_C for y ~ N(delta, A)."""
    a = rbf(x, x, LENGTHSCALE, SCALE) + NOISE_VAR * np.eye(len(x))
    chol = np.linalg.cholesky(a)
    q = np.linalg.inv(a)
    y = delta_full[:, None] + chol @ rng.standard_normal((len(x), MC_DRAWS))
    alpha_c = (q @ y)[group]
    t = np.einsum("ij,ij->j", alpha_c, np.linalg.solve(q[np.ix_(group, group)], alpha_c))
    return float(np.mean(t > chi2.ppf(1.0 - ALPHA, len(group))))


def bump(t, x0):
    return BUMP_HEIGHT * np.exp(-0.5 * ((t - x0) / BUMP_WIDTH) ** 2)


def bump_power(x, x0):
    group = np.flatnonzero(np.abs(x - x0) <= WINDOW)
    if len(group) == 0:
        return 0.0, 0.0
    eta = noncentrality(precision(x), group, bump(x, x0))
    return eta, power(eta, len(group))


def predictive_variance(x, t):
    a = rbf(x, x, LENGTHSCALE, SCALE) + NOISE_VAR * np.eye(len(x))
    v = np.linalg.solve(np.linalg.cholesky(a), rbf(x, t, LENGTHSCALE, SCALE))
    return SCALE - np.sum(v ** 2, axis=0)


def score_candidates(x, x0):
    powers = np.array([bump_power(np.append(x, c), x0)[1] for c in CANDIDATES])
    return powers, predictive_variance(x, CANDIDATES)


def part_a(rng):
    x = make_design(3)
    q = precision(x)
    cluster = list(np.flatnonzero(np.abs(x - 0.2) <= 0.011))
    groups = {
        "boundary": [int(np.argmax(x))],
        "interior": [int(np.argmin(np.abs(x + 0.4)))],
        "one in cluster": [cluster[0]],
        "whole cluster": cluster,
    }
    out = {"x": x.tolist(), "offsets": OFFSETS.tolist(), "groups": {}}
    for name, g in groups.items():
        g = [int(i) for i in g]
        pred, emp = [], []
        for c in OFFSETS:
            delta = np.zeros(N)
            delta[g] = c
            pred.append(power(noncentrality(q, g, delta), len(g)))
            emp.append(monte_carlo_power(x, g, delta, rng))
        out["groups"][name] = {"index": g, "soft_gale_sq": [float(NOISE_VAR * q[i, i]) for i in g],
                               "predicted": pred, "empirical": emp}
        print(f"E1(a) {name}: power at c={OFFSETS[-5]:.3f} predicted {pred[-5]:.3f}, "
              f"empirical {emp[-5]:.3f}", flush=True)
    return out


def part_b_example(rng):
    x = make_design(3)
    powers, var = score_candidates(x, EXAMPLE_X0)
    choices = {"current": None, "max variance": float(CANDIDATES[np.argmax(var)]),
               "max power": float(CANDIDATES[np.argmax(powers)])}
    summary = {}
    for name, c in choices.items():
        xa = x if c is None else np.append(x, c)
        group = np.flatnonzero(np.abs(xa - EXAMPLE_X0) <= WINDOW)
        delta = bump(xa, EXAMPLE_X0)
        eta, pw = bump_power(xa, EXAMPLE_X0)
        summary[name] = {"candidate": c, "eta": eta, "predicted": pw,
                         "empirical": monte_carlo_power(xa, list(group), delta, rng)}
        print(f"E1(b) example, {name}: candidate {c}, power predicted {pw:.3f}, "
              f"empirical {summary[name]['empirical']:.3f}", flush=True)
    return {"x": x.tolist(), "x0": EXAMPLE_X0, "candidates": CANDIDATES.tolist(),
            "power": powers.tolist(), "variance": var.tolist(), "choices": summary}


def part_b_sweep(rng):
    rows = []
    start = time.perf_counter()
    for d, seed in enumerate(SWEEP_DESIGNS):
        x = make_design(seed)
        for x0 in SWEEP_X0:
            powers, var = score_candidates(x, x0)
            rows.append({
                "seed": seed, "x0": float(x0),
                "current": bump_power(x, x0)[1],
                "random": float(np.mean(powers)),
                "max variance": float(powers[np.argmax(var)]),
                "max power": float(powers.max()),
                "distance": float(abs(CANDIDATES[np.argmax(var)] - CANDIDATES[np.argmax(powers)])),
            })
        print(f"E1(b) sweep: design {d + 1}/{len(SWEEP_DESIGNS)} done "
              f"({time.perf_counter() - start:.0f} s)", flush=True)
    return rows


CURVATURES = np.linspace(0.0, 0.25, 11)       # part (c)
STRUCTURE_DESIGNS = {
    "two ends": np.repeat([-1.0, 1.0], 20),
    "two ends + centre": np.concatenate([np.repeat([-1.0, 1.0], 18), np.zeros(4)]),
    "uniform": np.linspace(-1.0, 1.0, 40),
}


LARGE_CURVATURES = np.linspace(0.0, 10.0, 41)  # part (c), end-point design only


def _structure_tests(x, curvatures, rng):
    """Power of the GP test (C = all) and of the exact residual test against c x^2.

    GP test: T = y^T Q y, chi-square with n degrees of freedom under the null,
    noncentrality delta^T Q delta (noncentral chi-square power of the group test).
    Residual test of the linear model: ||G^T y||^2 / sigma^2, chi-square with
    n - r degrees of freedom, noncentrality ||G^T delta||^2 / sigma^2. It does not
    depend on the coefficients, so it is the same under any prior on them.
    """
    n = len(x)
    k_mat = 1.0 + np.outer(x, x)
    a = k_mat + NOISE_VAR * np.eye(n)
    q = np.linalg.inv(a)
    chol = np.linalg.cholesky(a)
    lam, u = np.linalg.eigh(k_mat)
    g = u[:, lam < 1e-9 * lam.max()]
    dof_res = g.shape[1]
    crit_gp, crit_res = chi2.ppf(1.0 - ALPHA, n), chi2.ppf(1.0 - ALPHA, dof_res)
    out = {k: [] for k in ["gp_predicted", "gp_empirical", "res_predicted", "res_empirical"]}
    for c in curvatures:
        delta = c * x ** 2
        out["gp_predicted"].append(power(float(delta @ q @ delta), n))
        out["res_predicted"].append(power(float(np.sum((g.T @ delta) ** 2) / NOISE_VAR), dof_res))
        y = delta[:, None] + chol @ rng.standard_normal((n, MC_DRAWS))
        out["gp_empirical"].append(float(np.mean(np.einsum("ij,ij->j", y, q @ y) > crit_gp)))
        r = g.T @ y
        out["res_empirical"].append(float(np.mean(np.sum(r ** 2, axis=0) / NOISE_VAR > crit_res)))
    visible = float(np.sum((g.T @ x ** 2) ** 2) / np.sum(x ** 4))
    return out, dof_res, visible


def part_c(rng):
    """Straight-line hypothesis, k(x, x') = 1 + x x', against a curvature c x^2."""
    out = {"curvatures": CURVATURES.tolist(), "designs": {},
           "large_curvatures": LARGE_CURVATURES.tolist()}
    for name, x in STRUCTURE_DESIGNS.items():
        res, dof_res, visible = _structure_tests(x, CURVATURES, rng)
        out["designs"][name] = dict(x=x.tolist(), predicted=res["gp_predicted"],
                                    empirical=res["gp_empirical"], res_predicted=res["res_predicted"],
                                    res_empirical=res["res_empirical"], visible_fraction=visible,
                                    relations=dof_res)
        print(f"E1(c) {name}: {dof_res} relations, at c={CURVATURES[5]:.3f} GP power "
              f"{res['gp_predicted'][5]:.3f}, residual power {res['res_predicted'][5]:.3f}", flush=True)
    res, _, _ = _structure_tests(STRUCTURE_DESIGNS["two ends"], LARGE_CURVATURES, rng)
    out["two_ends_large"] = res
    print("E1(c) two ends, large c: GP power at c=5 and 10:",
          round(res["gp_predicted"][20], 4), round(res["gp_predicted"][40], 6),
          "residual power:", round(res["res_predicted"][40], 4), flush=True)
    return out


def update_part_c():
    """Recompute only part (c) in an existing E1.json."""
    path = os.path.join(RESULTS, "E1.json")
    with open(path) as fh:
        out = json.load(fh)
    out["c"] = part_c(np.random.default_rng(2027))
    with open(path, "w") as fh:
        json.dump(out, fh)
    print("updated part (c) in", path)


EXAMPLE_OFFSET = 0.2


def part_examples(rng):
    """One simulated data set for illustration, on the design of parts (a) and (b).

    A latent function is drawn from the GP prior on a fine grid and at the design,
    and noisy responses are added. The planted discrepancies are stored separately,
    so the figure can show the clean data, the boundary offset and the bump.
    """
    x = make_design(3)
    grid = np.linspace(-1.0, 1.0, 400)
    pts = np.concatenate([grid, x])
    cov = rbf(pts, pts, LENGTHSCALE, SCALE) + 1e-10 * np.eye(len(pts))
    f_all = np.linalg.cholesky(cov) @ rng.standard_normal(len(pts))
    f_grid, f_x = f_all[:len(grid)], f_all[len(grid):]
    y = f_x + rng.normal(0.0, np.sqrt(NOISE_VAR), N)
    cluster = [int(i) for i in np.flatnonzero(np.abs(x - 0.2) <= 0.011)]
    return {"x": x.tolist(), "grid": grid.tolist(), "f_grid": f_grid.tolist(), "y": y.tolist(),
            "boundary": int(np.argmax(x)), "interior": int(np.argmin(np.abs(x + 0.4))),
            "cluster": cluster, "offset": EXAMPLE_OFFSET, "x0": EXAMPLE_X0,
            "bump_grid": bump(grid, EXAMPLE_X0).tolist(), "bump_x": bump(x, EXAMPLE_X0).tolist()}


def add_examples():
    """Add the illustration to an existing E1.json without recomputing the other parts."""
    path = os.path.join(RESULTS, "E1.json")
    with open(path) as fh:
        out = json.load(fh)
    out["examples"] = part_examples(np.random.default_rng(2029))
    with open(path, "w") as fh:
        json.dump(out, fh)
    print("added examples to", path)


def main():
    rng = np.random.default_rng(2026)
    out = {"settings": {"n": N, "noise": NOISE_VAR, "lengthscale": LENGTHSCALE, "scale": SCALE,
                        "alpha": ALPHA, "mc_draws": MC_DRAWS, "bump_height": BUMP_HEIGHT,
                        "bump_width": BUMP_WIDTH, "window": WINDOW},
           "a": part_a(rng), "b_example": part_b_example(rng), "b_sweep": part_b_sweep(rng),
           "c": part_c(np.random.default_rng(2027)),
           "examples": part_examples(np.random.default_rng(2029))}
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "E1.json"), "w") as fh:
        json.dump(out, fh)
    print("wrote", os.path.join(RESULTS, "E1.json"))


if __name__ == "__main__":
    import sys

    if "--examples-only" in sys.argv:
        add_examples()
    elif "--part-c-only" in sys.argv:
        update_part_c()
    else:
        main()
