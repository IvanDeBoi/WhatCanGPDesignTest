"""Discrimination baseline and hyperparameter mismatch for experiments E1 and E2.

KL discrimination. Under the fully specified GP, the response vector is
N(0, A) under the null and N(delta, A) under the alternative, so the
Kullback-Leibler divergence between them is delta^T Q delta / 2 (half the
noncentrality of the test of all observations). The KL rule chooses the
candidate that maximises this divergence on the augmented design; for a
finite-rank kernel and small noise it reduces to the T-optimality criterion.
Every choice is evaluated by the power of the same group test as in E1 and E2.

Mismatch (E1(b) only). The rules choose under assumed hyperparameters
(lengthscale x {0.5, 1, 2}, or noise standard deviation x {0.5, 1, 2}), and the
chosen design is evaluated under the true model (lengthscale 0.3, noise
variance 0.0025). This isolates the robustness of the design criterion. The
test itself is evaluated with the true hyperparameters.

Run:  python code/e1e2_baselines.py
Writes code/results/E1_baselines.json and code/results/E2_kl.json.
"""

import json
import os
import time

import numpy as np
from scipy.stats import chi2, ncx2

import e1_power as e1
import e2_design as e2
from gpcore import rbf

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

TRUE = (e1.LENGTHSCALE, e1.NOISE_VAR)
SETTINGS = {
    "true": TRUE,
    "lengthscale x0.5": (0.5 * e1.LENGTHSCALE, e1.NOISE_VAR),
    "lengthscale x2": (2.0 * e1.LENGTHSCALE, e1.NOISE_VAR),
    "noise sd x0.5": (e1.LENGTHSCALE, 0.25 * e1.NOISE_VAR),
    "noise sd x2": (e1.LENGTHSCALE, 4.0 * e1.NOISE_VAR),
}


def precision(x, ls, noise):
    return np.linalg.inv(rbf(x, x, ls, 1.0) + noise * np.eye(len(x)))


def group_power(x, x0, ls, noise):
    group = np.flatnonzero(np.abs(x - x0) <= e1.WINDOW)
    if len(group) == 0:
        return 0.0
    q = precision(x, ls, noise)
    w = (q @ e1.bump(x, x0))[group]
    eta = float(w @ np.linalg.solve(q[np.ix_(group, group)], w))
    return float(ncx2.sf(chi2.ppf(1 - e1.ALPHA, len(group)), len(group), max(eta, 1e-12)))


def kl(x, x0, ls, noise):
    d = e1.bump(x, x0)
    return 0.5 * float(d @ precision(x, ls, noise) @ d)


def variance(x, t, ls, noise):
    a = rbf(x, x, ls, 1.0) + noise * np.eye(len(x))
    v = np.linalg.solve(np.linalg.cholesky(a), rbf(x, t, ls, 1.0))
    return 1.0 - np.sum(v ** 2, axis=0)


def e1_part():
    cands = e1.CANDIDATES
    rows = []
    start = time.perf_counter()
    for d, seed in enumerate(e1.SWEEP_DESIGNS):
        x = e1.make_design(seed)
        for x0 in e1.SWEEP_X0:
            true_power = np.array([group_power(np.append(x, c), x0, *TRUE) for c in cands])
            row = {"seed": seed, "x0": float(x0), "current": group_power(x, x0, *TRUE), "settings": {}}
            for name, (ls, noise) in SETTINGS.items():
                if name == "true":
                    p_assumed = true_power
                else:
                    p_assumed = np.array([group_power(np.append(x, c), x0, ls, noise) for c in cands])
                k_assumed = np.array([kl(np.append(x, c), x0, ls, noise) for c in cands])
                v_assumed = variance(x, cands, ls, noise)
                picks = {"power": int(np.argmax(p_assumed)), "KL": int(np.argmax(k_assumed)),
                         "variance": int(np.argmax(v_assumed))}
                row["settings"][name] = {r: float(true_power[i]) for r, i in picks.items()}
                if name == "true":
                    row["kl_same_choice"] = bool(abs(cands[picks["KL"]] - cands[picks["power"]]) < 1e-9)
                    row["kl_distance"] = float(abs(cands[picks["KL"]] - cands[picks["power"]]))
            rows.append(row)
        print(f"E1 baselines: design {d + 1}/{len(e1.SWEEP_DESIGNS)} done "
              f"({time.perf_counter() - start:.0f} s)", flush=True)
    summary = {}
    for name in SETTINGS:
        summary[name] = {r: float(np.mean([row["settings"][name][r] for row in rows]))
                         for r in ("power", "KL", "variance")}
    summary["current"] = float(np.mean([r["current"] for r in rows]))
    summary["kl_same_choice"] = float(np.mean([r["kl_same_choice"] for r in rows]))
    print(json.dumps(summary, indent=1), flush=True)
    return {"rows": rows, "summary": summary}


def e2_kl_choice(xa):
    scores = []
    for c in e2.CANDIDATES:
        xc = np.append(xa, c)
        q = np.linalg.inv(rbf(xc, xc, e2.LENGTHSCALE, 1.0) + e2.NOISE_VAR * np.eye(len(xc)))
        b = e2.bumps(xc)
        scores.append(e2.WEIGHTS @ (0.5 * np.einsum("ij,ik,kj->j", b, q, b)))
    return float(e2.CANDIDATES[int(np.argmax(scores))])


def e2_part():
    cached = json.load(open(os.path.join(RESULTS, "E2.json")))
    out = {"designs": []}
    start = time.perf_counter()
    for d, seed in enumerate(e2.DESIGNS):
        x = e2.make_design(seed)
        added = []
        p = e2.family_power(x)
        means, worsts = [float(e2.WEIGHTS @ p)], [float(p.min())]
        for _ in range(e2.BUDGET):
            added.append(e2_kl_choice(np.append(x, added)))
            p = e2.family_power(np.append(x, added))
            means.append(float(e2.WEIGHTS @ p))
            worsts.append(float(p.min()))
        ref = cached["designs"][d]
        assert ref["seed"] == seed
        out["designs"].append({"seed": seed, "mean": means, "worst": worsts, "chosen": added,
                               "average_power_mean": ref["average power"]["mean"]})
        print(f"E2 KL: design {d + 1}/{len(e2.DESIGNS)} ({time.perf_counter() - start:.0f} s): "
              f"KL {means[-1]:.3f}, average power {ref['average power']['mean'][-1]:.3f}", flush=True)
    kl_final = np.array([r["mean"][-1] for r in out["designs"]])
    ap_final = np.array([r["average_power_mean"][-1] for r in out["designs"]])
    diff = ap_final - kl_final
    out["summary"] = {"kl_mean": float(kl_final.mean()), "average_power_mean": float(ap_final.mean()),
                      "paired_diff": float(diff.mean()), "paired_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
                      "kl_worst": float(np.mean([r["worst"][-1] for r in out["designs"]]))}
    print(json.dumps(out["summary"], indent=1), flush=True)
    return out


def main():
    os.makedirs(RESULTS, exist_ok=True)
    res2 = e2_part()
    with open(os.path.join(RESULTS, "E2_kl.json"), "w") as fh:
        json.dump(res2, fh)
    res1 = e1_part()
    with open(os.path.join(RESULTS, "E1_baselines.json"), "w") as fh:
        json.dump(res1, fh)


if __name__ == "__main__":
    main()
