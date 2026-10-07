"""Deletion trajectories: what happens after each single deletion.

For F2I-style deletion and the classical group test, records the order in which
observations are deleted (the group test deletes whole groups, recorded one
observation at a time in index order), and after every deletion refits the
noise and evaluates the predictions in the target region.

Uses the same data, seeds and fitted kernels as experiments.py.
Run:  python code/trajectories.py
Writes code/results/trajectories.json and output/figures/fig_trajectories.pdf.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import methods
from experiments import N, NOISE_VAR, TARGET, N_TEST, N_TARGET_GRID, SEEDS, make_data
from gpcore import GP, fit_hyperparameters, fit_noise

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
FIGURES = os.path.join(HERE, "..", "output", "figures")

CASES = [("S1", "S1", {}, range(SEEDS)), ("S2", "S2", {}, range(SEEDS)),
         ("S3 ($k=4$)", "S3", {"k": 4}, range(1000, 1000 + SEEDS))]


def f2i_order(gp, x, y, x_target):
    keep = list(range(len(x)))
    current = methods.f2i_objective(gp, x, y, keep, x_target)
    order = []
    while len(keep) > 3:
        trials = [(methods.f2i_objective(gp, x, y, [j for j in keep if j != i], x_target), i)
                  for i in keep]
        best_value, best_i = min(trials)
        if best_value >= current:
            break
        keep.remove(best_i)
        order.append(best_i)
        current = best_value
    return order


def evaluate(gp, x, y, f, keep, x_test, y_test):
    keep = np.array(sorted(keep), int)
    noise = fit_noise(gp.k(x[keep], x[keep]), y[keep])
    mean, var = gp.predict(x[keep], y[keep], noise, x_test)
    pred_var = var + noise
    return {"noise": noise,
            "coverage": float(np.mean(np.abs(y_test - mean) <= 1.96 * np.sqrt(pred_var))),
            "mse": float(np.mean((mean - f(x_test)) ** 2)),
            "sharpness": float(np.mean(pred_var))}


def trajectory(gp, x, y, f, misfit, order, x_test, y_test):
    keep = list(range(len(x)))
    steps = [dict(evaluate(gp, x, y, f, keep, x_test, y_test), misfit=None)]
    for i in order:
        keep.remove(i)
        steps.append(dict(evaluate(gp, x, y, f, keep, x_test, y_test), misfit=bool(misfit[i])))
    return steps


def run_case(name, kwargs, seeds):
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        x, y, f, misfit = make_data(name, rng, **kwargs)
        in_target = (x >= TARGET[0]) & (x <= TARGET[1])
        x_test = rng.uniform(*TARGET, N_TEST)
        y_test = f(x_test) + rng.normal(0, np.sqrt(NOISE_VAR), N_TEST)
        x_target = np.linspace(*TARGET, N_TARGET_GRID)
        lengthscale, scale, _ = fit_hyperparameters(x, y, rng)
        gp = GP(lengthscale, scale)
        ref_noise = methods.reference_noise(gp, x, y, in_target)
        groups = methods.candidate_groups(x, 5 if name == "S3" else 3)
        group_keep = methods.group(gp=gp, x=x, y=y, ref_noise=ref_noise, groups=groups)
        group_order = [i for i in range(N) if i not in group_keep]
        out.append({
            "seed": seed,
            "n_misfit": int(misfit.sum()),
            "f2i_style": trajectory(gp, x, y, f, misfit, f2i_order(gp, x, y, x_target), x_test, y_test),
            "group": trajectory(gp, x, y, f, misfit, group_order, x_test, y_test),
        })
        print(name, kwargs, seed, "done", flush=True)
    return out


def group_steps(name, kwargs, seeds):
    """Trajectory of the classical group test in its own order of selection.

    Each step deletes one selected group. Uses the same data, seeds and fitted
    kernels as run_case, so the final step equals the stored end point.
    """
    out = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        x, y, f, misfit = make_data(name, rng, **kwargs)
        in_target = (x >= TARGET[0]) & (x <= TARGET[1])
        x_test = rng.uniform(*TARGET, N_TEST)
        y_test = f(x_test) + rng.normal(0, np.sqrt(NOISE_VAR), N_TEST)
        lengthscale, scale, _ = fit_hyperparameters(x, y, rng)
        gp = GP(lengthscale, scale)
        ref_noise = methods.reference_noise(gp, x, y, in_target)
        groups = methods.candidate_groups(x, 5 if name == "S3" else 3)
        record = []
        methods.group(gp=gp, x=x, y=y, ref_noise=ref_noise, groups=groups, record=record)
        keep = list(range(N))
        steps = [dict(evaluate(gp, x, y, f, keep, x_test, y_test), deleted=0)]
        for grp in record:
            keep = [i for i in keep if i not in grp]
            steps.append(dict(evaluate(gp, x, y, f, keep, x_test, y_test), deleted=N - len(keep)))
        out.append(steps)
        print(name, kwargs, seed, "group steps done", flush=True)
    return out


def plot(data):
    fig, axes = plt.subplots(3, len(CASES), figsize=(6.3, 5.6), sharex="col")
    for col, (title, tag) in enumerate([(c[0], c[0]) for c in CASES]):
        runs = data[tag]
        max_len = max(len(r["f2i_style"]) for r in runs)
        steps = np.arange(max_len)
        n_mis = np.mean([r["n_misfit"] for r in runs])

        def collect(key, method):
            arr = np.full((len(runs), max_len), np.nan)
            for s, r in enumerate(runs):
                vals = [st[key] for st in r[method]]
                arr[s, :len(vals)] = vals
            return arr

        mis = np.full((len(runs), max_len), np.nan)
        for s, r in enumerate(runs):
            for t, st in enumerate(r["f2i_style"][1:], start=1):
                mis[s, t] = float(st["misfit"])
        frac_mis = np.nanmean(mis[:, 1:], axis=0)
        active = np.sum(~np.isnan(mis[:, 1:]), axis=0)
        ok = active >= 5
        axes[0, col].bar(steps[1:][ok], frac_mis[ok], color="#c0392b", width=0.8)
        axes[0, col].set_ylim(0, 1.05)
        axes[0, col].set_title(title, fontsize=9)

        noise = collect("noise", "f2i_style") / NOISE_VAR
        cov = collect("coverage", "f2i_style")
        for s in range(len(runs)):
            axes[1, col].plot(steps, noise[s], color="#c0392b", alpha=0.15, lw=0.6)
            axes[2, col].plot(steps, cov[s], color="#c0392b", alpha=0.15, lw=0.6)
        med_ok = np.sum(~np.isnan(noise), axis=0) >= 5
        axes[1, col].plot(steps[med_ok], np.nanmedian(noise, axis=0)[med_ok], color="#c0392b", lw=1.6)
        axes[2, col].plot(steps[med_ok], np.nanmedian(cov, axis=0)[med_ok], color="#c0392b", lw=1.6)

        if "group_steps" in data:
            g_runs = data["group_steps"][tag]
            for row, key in [(1, "noise"), (2, "coverage")]:
                for s, steps in enumerate(g_runs):
                    xs = [st["deleted"] for st in steps]
                    vals = [st[key] / (NOISE_VAR if key == "noise" else 1.0) for st in steps]
                    axes[row, col].plot(xs, vals, color="#17a589", alpha=0.5, lw=0.8)

        g_end = [len(r["group"]) - 1 for r in runs]
        g_noise = [r["group"][-1]["noise"] / NOISE_VAR for r in runs]
        g_cov = [r["group"][-1]["coverage"] for r in runs]
        axes[1, col].scatter(g_end, g_noise, s=6, color="#17a589", zorder=4)
        axes[2, col].scatter(g_end, g_cov, s=6, color="#17a589", zorder=4)
        f_end = [len(r["f2i_style"]) - 1 for r in runs]
        f_noise = [r["f2i_style"][-1]["noise"] / NOISE_VAR for r in runs]
        f_cov = [r["f2i_style"][-1]["coverage"] for r in runs]
        axes[1, col].scatter(f_end, f_noise, s=6, color="#c0392b", zorder=4)
        axes[2, col].scatter(f_end, f_cov, s=6, color="#c0392b", zorder=4)

        for row in range(3):
            axes[row, col].axvline(n_mis, color="0.4", ls="--", lw=0.8)
        axes[1, col].axhline(1.0, color="0.5", ls=":", lw=0.8)
        axes[2, col].axhline(0.95, color="0.5", ls=":", lw=0.8)
        axes[1, col].set_yscale("log")
        axes[2, col].set_ylim(-0.02, 1.02)
        axes[2, col].set_xlabel("observations deleted")
    axes[0, 0].set_ylabel("misfit share\nof deletion")
    axes[1, 0].set_ylabel(r"$\hat\sigma^2/\sigma_0^2$")
    axes[2, 0].set_ylabel(r"coverage in $\mathcal{A}$")
    handles = [plt.Line2D([], [], color="#c0392b", lw=1.2, label="F2I-style"),
               plt.Line2D([], [], color="#17a589", lw=1.2, label="group test"),
               plt.Line2D([], [], color="#c0392b", marker="o", ms=3, ls="", label="F2I-style, stop"),
               plt.Line2D([], [], color="#17a589", marker="o", ms=3, ls="", label="group test, stop")]
    axes[1, 0].legend(handles=handles, fontsize=7, frameon=False, loc="lower left")
    for ax in axes.flat:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_trajectories.pdf"))
    plt.close(fig)


def main():
    path = os.path.join(RESULTS, "trajectories.json")
    if os.path.exists(path):
        with open(path) as fh:
            data = json.load(fh)
    else:
        data = {title: run_case(name, kwargs, seeds) for title, name, kwargs, seeds in CASES}
        with open(path, "w") as fh:
            json.dump(data, fh)
    if "group_steps" not in data:
        data["group_steps"] = {title: group_steps(name, kwargs, seeds) for title, name, kwargs, seeds in CASES}
        with open(path, "w") as fh:
            json.dump(data, fh)
    plt.rcParams.update({"font.size": 9, "font.family": "serif"})
    plot(data)


if __name__ == "__main__":
    main()
