"""Calibration of the test-based deletion pipeline under a true GP null.

Unlike calibration.py, which uses a fixed cubic, the data here are drawn from
the GP itself: RBF kernel with lengthscale 0.3 and unit output scale, noise
variance 0.0025, zero mean. Any deletion is a false rejection. Each data set
is analysed twice:
  fixed      the true kernel and the true noise variance are used, so every
             single group test is exact
  estimated  the pipeline of S1-S3: MAP fit of all hyperparameters on all data
             and a reference noise fitted on the observations in the target region
The difference between the two isolates the effect of estimating the
hyperparameters. The greedy search and Bonferroni stopping rule are the same.

Configurations as in calibration.py: 40 uniform inputs with groups up to size 3,
and 36 uniform inputs plus 4 at 0.2 +- 0.01 with groups up to size 5.

Run:  python code/calibration_gp.py
Writes code/results/calibration_gp.json.
"""

import json
import os
import time

import numpy as np

import methods
from experiments import NOISE_VAR, TARGET
from gpcore import GP, fit_hyperparameters, rbf

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

N = 40
LENGTHSCALE = 0.3
SCALE = 1.0
SEEDS = range(7000, 8000)
TESTS = ["single", "group", "gale"]


def make_inputs(name, rng):
    if name == "null_uniform":
        return rng.uniform(-1, 1, N), 3
    return np.concatenate([rng.uniform(-1, 1, N - 4), 0.2 + rng.uniform(-0.01, 0.01, 4)]), 5


def run_one(name, seed):
    rng = np.random.default_rng(seed)
    x, max_group = make_inputs(name, rng)
    k = rbf(x, x, LENGTHSCALE, SCALE) + 1e-10 * np.eye(N)
    y = np.linalg.cholesky(k) @ rng.standard_normal(N) + rng.normal(0, np.sqrt(NOISE_VAR), N)
    groups = methods.candidate_groups(x, max_group)
    out = {"seed": seed}
    true_gp = GP(LENGTHSCALE, SCALE)
    for m in TESTS:
        keep = getattr(methods, m)(gp=true_gp, x=x, y=y, ref_noise=NOISE_VAR, groups=groups)
        out[f"fixed_{m}"] = N - len(keep)
    lengthscale, scale, _ = fit_hyperparameters(x, y, rng)
    gp = GP(lengthscale, scale)
    in_target = (x >= TARGET[0]) & (x <= TARGET[1])
    ref_noise = methods.reference_noise(gp, x, y, in_target)
    for m in TESTS:
        keep = getattr(methods, m)(gp=gp, x=x, y=y, ref_noise=ref_noise, groups=groups)
        out[f"estimated_{m}"] = N - len(keep)
    out["lengthscale"], out["ref_noise"] = lengthscale, ref_noise
    return out


def main():
    results = {}
    for name in ["null_uniform", "null_cluster"]:
        runs = []
        start = time.perf_counter()
        for k, seed in enumerate(SEEDS):
            runs.append(run_one(name, seed))
            if (k + 1) % 100 == 0:
                rates = {key: round(float(np.mean([r[key] > 0 for r in runs])), 3)
                         for key in runs[0] if key.startswith(("fixed", "estimated"))}
                print(f"{name}: {k + 1}/{len(SEEDS)} seeds ({time.perf_counter() - start:.0f} s), "
                      f"any-deletion rates so far {rates}", flush=True)
        results[name] = runs
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "calibration_gp.json"), "w") as fh:
        json.dump(results, fh)
    print("wrote", os.path.join(RESULTS, "calibration_gp.json"))


if __name__ == "__main__":
    main()
