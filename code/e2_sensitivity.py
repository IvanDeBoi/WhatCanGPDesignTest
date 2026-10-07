"""Experiment E2, paired comparison and sensitivity to a misspecified alternative.

The designs chosen in E2 (code/results/E2.json) were selected for bumps of
height 0.2, width 0.04 and a normal location prior centred at 0.6. Here the
same final designs are evaluated when the true discrepancy differs from that
assumption: a narrower or wider bump, a lower or higher bump, or a uniform
location prior. The designs are not re-selected, so the rows measure what
happens when the alternative used for design is wrong. Random designs are
redrawn (10 per starting design, as in E2).

For each scenario and each rule, the script reports the prior-weighted mean
power after five additions, and the paired difference to the average-power
rule with its standard error over the 20 starting designs.

Run:  python code/e2_sensitivity.py
Writes code/results/E2_sensitivity.json.
"""

import json
import os

import numpy as np
from scipy.stats import chi2, ncx2

import e2_design as e2
from gpcore import rbf

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

UNIFORM = np.full(len(e2.LOCATIONS), 1.0 / len(e2.LOCATIONS))
SCENARIOS = {
    "as assumed": dict(height=e2.HEIGHT, width=e2.WIDTH, weights=e2.WEIGHTS),
    "narrower bump (width 0.02)": dict(height=e2.HEIGHT, width=0.02, weights=e2.WEIGHTS),
    "wider bump (width 0.08)": dict(height=e2.HEIGHT, width=0.08, weights=e2.WEIGHTS),
    "lower bump (height 0.1)": dict(height=0.1, width=e2.WIDTH, weights=e2.WEIGHTS),
    "higher bump (height 0.3)": dict(height=0.3, width=e2.WIDTH, weights=e2.WEIGHTS),
    "uniform location prior": dict(height=e2.HEIGHT, width=e2.WIDTH, weights=UNIFORM),
}
RULES = ["average power", "worst-case", "variance", "amplitude", "space-filling", "random"]


def mean_power(x, height, width, weights):
    group = np.flatnonzero((x >= e2.REGION[0]) & (x <= e2.REGION[1]))
    q = np.linalg.inv(rbf(x, x, e2.LENGTHSCALE, 1.0) + e2.NOISE_VAR * np.eye(len(x)))
    d = height * np.exp(-0.5 * ((x[:, None] - e2.LOCATIONS[None, :]) / width) ** 2)
    w = (q @ d)[group]
    eta = np.einsum("ij,ij->j", w, np.linalg.solve(q[np.ix_(group, group)], w))
    k = len(group)
    p = 1.0 - ncx2.cdf(chi2.ppf(1.0 - e2.ALPHA, k), k, np.maximum(eta, 1e-12))
    return float(weights @ p)


def main():
    stored = json.load(open(os.path.join(RESULTS, "E2.json")))
    rng = np.random.default_rng(2030)
    out = {}
    for name, sc in SCENARIOS.items():
        per_rule = {r: [] for r in RULES}
        for res in stored["designs"]:
            x = e2.make_design(res["seed"])
            for r in RULES:
                if r == "random":
                    vals = [mean_power(np.append(x, rng.uniform(*e2.REGION, e2.BUDGET)), **sc)
                            for _ in range(e2.RANDOM_DRAWS)]
                    per_rule[r].append(float(np.mean(vals)))
                elif r == "space-filling":
                    pts = np.linspace(*e2.REGION, e2.BUDGET + 2)[1:-1]
                    per_rule[r].append(mean_power(np.append(x, pts), **sc))
                else:
                    per_rule[r].append(mean_power(np.append(x, res[r]["chosen"]), **sc))
        ref = np.array(per_rule["average power"])
        out[name] = {r: {"mean": float(np.mean(v)),
                         "diff": float(np.mean(ref - np.array(v))),
                         "se": float(np.std(ref - np.array(v), ddof=1) / np.sqrt(len(v)))}
                     for r, v in per_rule.items()}
        print(name, {r: round(out[name][r]["mean"], 3) for r in RULES}, flush=True)
    with open(os.path.join(RESULTS, "E2_sensitivity.json"), "w") as fh:
        json.dump(out, fh)
    print("wrote", os.path.join(RESULTS, "E2_sensitivity.json"))


if __name__ == "__main__":
    main()
