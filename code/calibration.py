"""Calibration of the complete test-based deletion pipeline under a clean null.

The data contain no misfit observations: the latent function is the cubic of
S1 and S3, observed with noise variance 0.0025. The pipeline is exactly the one
of S1-S3 (MAP fit of all hyperparameters on all data, reference noise from the
observations in the target region, greedy search with a Bonferroni stopping
rule). Any deletion is therefore a false rejection.

Configurations
  null_uniform   40 inputs drawn uniformly, candidate groups up to size 3 (as S1)
  null_cluster   36 uniform inputs plus 4 at 0.2 +- 0.01, groups up to size 5 (as S3)

Run:  python code/calibration.py
Writes code/results/calibration.json.
"""

import json
import os
import time

import numpy as np

import methods
from experiments import NOISE_VAR, TARGET, cubic
from gpcore import GP, fit_hyperparameters

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

N = 40
SEEDS = range(5000, 6000)
TESTS = ["single", "group", "gale"]


def make_inputs(name, rng):
    if name == "null_uniform":
        return rng.uniform(-1, 1, N), 3
    return np.concatenate([rng.uniform(-1, 1, N - 4), 0.2 + rng.uniform(-0.01, 0.01, 4)]), 5


def run_one(name, seed):
    rng = np.random.default_rng(seed)
    x, max_group = make_inputs(name, rng)
    y = cubic(x) + rng.normal(0, np.sqrt(NOISE_VAR), N)
    lengthscale, scale, _ = fit_hyperparameters(x, y, rng)
    gp = GP(lengthscale, scale)
    in_target = (x >= TARGET[0]) & (x <= TARGET[1])
    ref_noise = methods.reference_noise(gp, x, y, in_target)
    groups = methods.candidate_groups(x, max_group)
    out = {"seed": seed, "ref_noise": ref_noise, "lengthscale": lengthscale}
    for m in TESTS:
        keep = getattr(methods, m)(gp=gp, x=x, y=y, ref_noise=ref_noise, groups=groups)
        out[m] = N - len(keep)
    return out


def main():
    results = {}
    for name in ["null_uniform", "null_cluster"]:
        runs = []
        start = time.perf_counter()
        for k, seed in enumerate(SEEDS):
            runs.append(run_one(name, seed))
            if (k + 1) % 100 == 0:
                rates = {m: np.mean([r[m] > 0 for r in runs]) for m in TESTS}
                print(f"{name}: {k + 1}/{len(SEEDS)} seeds ({time.perf_counter() - start:.0f} s),"
                      f" any-deletion rates so far {rates}", flush=True)
        results[name] = runs
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "calibration.json"), "w") as fh:
        json.dump(results, fh)
    print("wrote", os.path.join(RESULTS, "calibration.json"))


if __name__ == "__main__":
    main()
