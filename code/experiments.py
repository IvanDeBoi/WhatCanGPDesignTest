"""Experiments S1, S2 and S3: detection and forgetting (appendix of the paper).

All experiments use n = 40 training inputs, true noise variance 0.0025,
target region A = [0.3, 0.9], 20 seeds and 200 held-out test inputs in A per
seed. The pipeline is identical for every method:

1. draw the data;
2. fit the RBF lengthscale, output scale and noise by MAP on all 40 points;
3. apply the forgetting method, which returns the points to keep;
4. refit only the noise (MAP) on the kept points;
5. predict with the same kernel.

Run:  python code/experiments.py            (all experiments)
      python code/experiments.py S1 S3      (a subset)
Results are written to code/results/<experiment>.json.
"""

import json
import os
import sys
import time

import numpy as np

import methods
from gpcore import GP, fit_hyperparameters, fit_noise

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

N = 40
NOISE_VAR = 0.0025
TARGET = (0.3, 0.9)
SEEDS = 20
N_TEST = 200
N_TARGET_GRID = 30          # target set used inside the F2I objective

METHODS = ["full", "oracle", "f2i_style", "single", "gale", "group"]


def cubic(x):
    return 0.5 + 0.8 * x - 1.2 * x ** 2 + 0.9 * x ** 3


def rough(x):
    return np.sin(3 * x) + 1.2 * np.sin(35 * x) * (x < -0.3)


def make_data(name, rng, k=4, batch_centre=0.2):
    """Return inputs, noisy responses, latent function and planted-misfit mask."""
    if name == "S1":
        x = rng.uniform(-1, 1, N)
        f = cubic
        y = f(x) + rng.normal(0, np.sqrt(NOISE_VAR), N)
        misfit = np.zeros(N, bool)
        misfit[np.argsort(np.abs(x + 0.1))[:3]] = True
        y[misfit] += 0.4
    elif name == "S2":
        x = rng.uniform(-1, 1, N)
        f = rough
        y = f(x) + rng.normal(0, np.sqrt(NOISE_VAR), N)
        misfit = x < -0.3
    elif name == "S3":
        x = np.concatenate([rng.uniform(-1, 1, N - k),
                            batch_centre + rng.uniform(-0.01, 0.01, k)])
        f = cubic
        y = f(x) + rng.normal(0, np.sqrt(NOISE_VAR), N)
        misfit = np.zeros(N, bool)
        misfit[N - k:] = True
        y[misfit] += 0.3
    else:
        raise ValueError(name)
    return x, y, f, misfit


def run_one(name, seed, **data_kwargs):
    rng = np.random.default_rng(seed)
    x, y, f, misfit = make_data(name, rng, **data_kwargs)
    in_target = (x >= TARGET[0]) & (x <= TARGET[1])
    x_test = rng.uniform(*TARGET, N_TEST)
    y_test = f(x_test) + rng.normal(0, np.sqrt(NOISE_VAR), N_TEST)
    x_target = np.linspace(*TARGET, N_TARGET_GRID)

    lengthscale, scale, _ = fit_hyperparameters(x, y, rng)
    gp = GP(lengthscale, scale)
    ref_noise = methods.reference_noise(gp, x, y, in_target)
    max_group = 5 if name == "S3" else 3
    groups = methods.candidate_groups(x, max_group)

    out = {"seed": seed, "lengthscale": lengthscale, "scale": scale, "ref_noise": ref_noise,
           "x": x.tolist(), "y": y.tolist(), "misfit": misfit.tolist(), "methods": {}}
    for m in METHODS:
        start = time.perf_counter()
        if m == "full":
            keep = list(range(N))
        elif m == "oracle":
            keep = list(np.flatnonzero(~misfit))
        else:
            keep = getattr(methods, m)(gp=gp, x=x, y=y, x_target=x_target,
                                       ref_noise=ref_noise, groups=groups)
        elapsed = time.perf_counter() - start
        keep = np.array(sorted(keep), int)
        noise = fit_noise(gp.k(x[keep], x[keep]), y[keep])
        mean, var = gp.predict(x[keep], y[keep], noise, x_test)
        pred_var = var + noise
        deleted = np.setdiff1d(np.arange(N), keep)
        out["methods"][m] = {
            "keep": keep.tolist(),
            "noise": noise,
            "mse": float(np.mean((mean - f(x_test)) ** 2)),
            "coverage": float(np.mean(np.abs(y_test - mean) <= 1.96 * np.sqrt(pred_var))),
            "nlpd": float(np.mean(0.5 * np.log(2 * np.pi * pred_var)
                                  + (y_test - mean) ** 2 / (2 * pred_var))),
            "sharpness": float(np.mean(pred_var)),
            "misfit_deleted": float(misfit[deleted].sum() / max(misfit.sum(), 1)),
            "honest_deleted": float((~misfit)[deleted].sum() / (~misfit).sum()),
            "time": elapsed,
        }
    return out


def run(name, seeds, tag=None, **data_kwargs):
    tag = tag or name
    runs = []
    for s in seeds:
        runs.append(run_one(name, s, **data_kwargs))
        print(f"{tag} seed {s} done", flush=True)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, tag + ".json"), "w") as fh:
        json.dump({"experiment": name, "tag": tag, "settings": data_kwargs, "runs": runs}, fh)


def main(which):
    if "S1" in which:
        run("S1", range(SEEDS))
    if "S2" in which:
        run("S2", range(SEEDS))
    if "S3" in which:
        s3_seeds = range(1000, 1000 + SEEDS)
        for k in (2, 3, 4, 5):
            run("S3", s3_seeds, tag=f"S3_k{k}", k=k)
        run("S3", s3_seeds, tag="S3_k4_inside", k=4, batch_centre=0.5)


if __name__ == "__main__":
    main(sys.argv[1:] or ["S1", "S2", "S3"])
