"""Experiment E2: choosing observations against a discrepancy of unknown location.

Fully specified GP null as in E1 (RBF, lengthscale 0.3, unit scale, noise
variance 0.0025). The alternative is a family of bumps of height 0.2 and
width 0.04 whose centre x0 is unknown. It lies on a grid in the region
[0.3, 0.9] with prior weights proportional to a normal density centred at 0.6
with standard deviation 0.15. The test is fixed in advance: the group test
T_C with C the observations in the region, including added ones. Its power
for each x0 follows from the noncentral chi-square power of the group test, which allows the bump to be non-zero
outside C.

Starting from each of 20 designs of 40 inputs, five observations are added one
at a time, from 61 candidate inputs in the region, by six rules:
  average power   maximise the prior-weighted mean power over the family
  worst-case      maximise the minimum power over the family
  variance        maximise the current predictive variance in the region
  amplitude       maximise the prior-weighted mean squared bump height at the
                  candidate, ignoring the covariance (always the same input)
  space-filling   equally spaced inputs in the region
  random          uniform inputs in the region (mean over 10 draws)
No response is needed to choose, because power depends only on the design.

Run:  python code/e2_design.py
Writes code/results/E2.json.
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
ALPHA = 0.05
REGION = (0.3, 0.9)
HEIGHT = 0.2
WIDTH = 0.04
LOCATIONS = np.linspace(*REGION, 31)
WEIGHTS = np.exp(-0.5 * ((LOCATIONS - 0.6) / 0.15) ** 2)
WEIGHTS /= WEIGHTS.sum()
CANDIDATES = np.linspace(*REGION, 61)
BUDGET = 5
DESIGNS = range(100, 120)
RANDOM_DRAWS = 10
MC_DRAWS = 4000
RULES = ["average power", "worst-case", "variance", "amplitude", "space-filling", "random"]


def make_design(seed):
    rng = np.random.default_rng(seed)
    x = np.concatenate([rng.uniform(-1, 1, N - 4), 0.2 + rng.uniform(-0.01, 0.01, 4)])
    return np.sort(x)


def bumps(x):
    """Matrix of discrepancies, one column per bump location."""
    return HEIGHT * np.exp(-0.5 * ((x[:, None] - LOCATIONS[None, :]) / WIDTH) ** 2)


def family_power(x):
    """Power of the region test against every bump location (noncentral chi-square power of the group test)."""
    group = np.flatnonzero((x >= REGION[0]) & (x <= REGION[1]))
    if len(group) == 0:
        return np.zeros(len(LOCATIONS))
    q = np.linalg.inv(rbf(x, x, LENGTHSCALE, 1.0) + NOISE_VAR * np.eye(len(x)))
    w = (q @ bumps(x))[group]
    eta = np.einsum("ij,ij->j", w, np.linalg.solve(q[np.ix_(group, group)], w))
    return 1.0 - ncx2.cdf(chi2.ppf(1.0 - ALPHA, len(group)), len(group), np.maximum(eta, 1e-12))


def predictive_variance(x, t):
    a = rbf(x, x, LENGTHSCALE, 1.0) + NOISE_VAR * np.eye(len(x))
    v = np.linalg.solve(np.linalg.cholesky(a), rbf(x, t, LENGTHSCALE, 1.0))
    return 1.0 - np.sum(v ** 2, axis=0)


def choose(rule, x, added, rng):
    xa = np.append(x, added)
    if rule in ("average power", "worst-case"):
        scores = []
        for c in CANDIDATES:
            p = family_power(np.append(xa, c))
            scores.append(WEIGHTS @ p if rule == "average power" else p.min())
        return float(CANDIDATES[int(np.argmax(scores))])
    if rule == "variance":
        return float(CANDIDATES[int(np.argmax(predictive_variance(xa, CANDIDATES)))])
    if rule == "amplitude":
        return float(CANDIDATES[int(np.argmax(bumps(CANDIDATES) ** 2 @ WEIGHTS))])
    return float(rng.uniform(*REGION))


def trajectory(rule, x, rng):
    """Mean and worst-case power after 0, 1, ..., BUDGET added observations."""
    if rule == "space-filling":
        means, worsts, chosen = [], [], []
        for m in range(BUDGET + 1):
            pts = np.linspace(*REGION, m + 2)[1:-1]
            p = family_power(np.append(x, pts))
            means.append(float(WEIGHTS @ p))
            worsts.append(float(p.min()))
            chosen = pts.tolist()
        return means, worsts, chosen
    added = []
    p = family_power(x)
    means, worsts = [float(WEIGHTS @ p)], [float(p.min())]
    for _ in range(BUDGET):
        added.append(choose(rule, x, added, rng))
        p = family_power(np.append(x, added))
        means.append(float(WEIGHTS @ p))
        worsts.append(float(p.min()))
    return means, worsts, added


def monte_carlo_check(x, rng):
    """Simulated rejection rates of the region test at five bump locations."""
    group = np.flatnonzero((x >= REGION[0]) & (x <= REGION[1]))
    a = rbf(x, x, LENGTHSCALE, 1.0) + NOISE_VAR * np.eye(len(x))
    chol, q = np.linalg.cholesky(a), np.linalg.inv(a)
    crit = chi2.ppf(1.0 - ALPHA, len(group))
    predicted = family_power(x)
    rows = []
    for j in range(0, len(LOCATIONS), 7):
        y = bumps(x)[:, [j]] + chol @ rng.standard_normal((len(x), MC_DRAWS))
        alpha_c = (q @ y)[group]
        t = np.einsum("ij,ij->j", alpha_c, np.linalg.solve(q[np.ix_(group, group)], alpha_c))
        rows.append({"x0": float(LOCATIONS[j]), "predicted": float(predicted[j]),
                     "empirical": float(np.mean(t > crit))})
    return rows


def main():
    rng = np.random.default_rng(2028)
    out = {"settings": {"region": REGION, "height": HEIGHT, "width": WIDTH, "budget": BUDGET,
                        "locations": LOCATIONS.tolist(), "weights": WEIGHTS.tolist()},
           "designs": []}
    start = time.perf_counter()
    for d, seed in enumerate(DESIGNS):
        x = make_design(seed)
        res = {"seed": seed}
        for rule in RULES:
            if rule == "random":
                runs = [trajectory(rule, x, rng) for _ in range(RANDOM_DRAWS)]
                res[rule] = {"mean": np.mean([r[0] for r in runs], axis=0).tolist(),
                             "worst": np.mean([r[1] for r in runs], axis=0).tolist()}
            else:
                means, worsts, chosen = trajectory(rule, x, rng)
                res[rule] = {"mean": means, "worst": worsts, "chosen": chosen}
        if d == 0:
            final = np.append(x, res["average power"]["chosen"])
            res["mc_check"] = monte_carlo_check(final, rng)
        out["designs"].append(res)
        print(f"E2: design {d + 1}/{len(DESIGNS)} done ({time.perf_counter() - start:.0f} s), "
              + ", ".join(f"{r} {res[r]['mean'][-1]:.2f}" for r in RULES), flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "E2.json"), "w") as fh:
        json.dump(out, fh)
    print("wrote", os.path.join(RESULTS, "E2.json"))


if __name__ == "__main__":
    main()
