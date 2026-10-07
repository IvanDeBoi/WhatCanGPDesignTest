"""Turn code/results/*.json into the tables and figures used in the paper.

Writes output/tables/*.tex and output/figures/*.pdf.
Run after experiments.py, e1_power.py, e2_design.py and calibration.py:
    python code/make_outputs.py
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments import NOISE_VAR as NOISE_TRUE, TARGET, cubic, rough
from gpcore import GP

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUTPUT = os.path.join(HERE, "..", "output")
TABLES = os.path.join(OUTPUT, "tables")
FIGURES = os.path.join(OUTPUT, "figures")

LABELS = {
    "full": "Full GP",
    "oracle": "Oracle",
    "f2i_style": "F2I-style",
    "single": "Single-observation test",
    "gale": "Group test, Gale coordinates",
    "group": "Group test, classical",
}
ORDER = ["full", "oracle", "f2i_style", "single", "group", "gale"]
COLOURS = {"f2i_style": "#c0392b", "single": "#7f8c8d", "gale": "#2471a3",
           "group": "#17a589", "oracle": "#000000", "full": "#b9770e"}

plt.rcParams.update({"font.size": 9, "font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})


def load(tag):
    with open(os.path.join(RESULTS, tag + ".json")) as fh:
        return json.load(fh)


def values(data, method, key):
    return np.array([r["methods"][method][key] for r in data["runs"]])


UP, DOWN = r"$\uparrow$", r"$\downarrow$"
DELETION_RULES = ["f2i_style", "single", "gale", "group"]


def bold_best(cells, scores, candidates):
    """Bold the cells of the candidates whose score is best (lowest); ties by printed value."""
    best = min(scores[c] for c in candidates)
    winners = {cells[c] for c in candidates if scores[c] == best}
    return {k: (rf"\textbf{{{v}}}" if k in candidates and v in winners else v) for k, v in cells.items()}


def summary_table(tag, caption, label):
    data = load(tag)
    lines = [
        r"\begin{table}[!htbp]", r"\centering", r"\small",
        r"\resizebox{\linewidth}{!}{\begin{tabular}{lrrrrrrr}", r"\toprule",
        r"Method & Kept & $\hat\sigma^2$ ($\times10^{-3}$) & MSE ($\times10^{-4}$) " + DOWN
        + r" & Coverage & NLPD " + DOWN + r" & Misfit del. " + UP + r" & Honest del. " + DOWN + r"\\",
        r"\midrule",
    ]
    cols = {k: {} for k in ["kept", "noise", "mse", "cov", "nlpd", "mis", "hon"]}
    score = {k: {} for k in cols}
    for m in ORDER:
        kept = np.mean([len(r["methods"][m]["keep"]) for r in data["runs"]])
        noise = values(data, m, "noise") * 1e3
        mse = values(data, m, "mse") * 1e4
        cov = values(data, m, "coverage")
        nlpd = values(data, m, "nlpd")
        mis = values(data, m, "misfit_deleted")
        hon = values(data, m, "honest_deleted")
        cols["kept"][m] = f"{kept:.1f}"
        cols["noise"][m] = f"{noise.mean():.2f}"
        cols["mse"][m] = f"{mse.mean():.1f} $\\pm$ {mse.std():.1f}"
        cols["cov"][m] = f"{cov.mean():.2f} $\\pm$ {cov.std():.2f}"
        cols["nlpd"][m] = f"{nlpd.mean():.2f}"
        cols["mis"][m] = f"{mis.mean():.2f}"
        cols["hon"][m] = f"{hon.mean():.3f}"
        # Lower is better for every score; rounded as printed so that printed ties are both bold.
        score["noise"][m] = round(abs(np.log(float(cols["noise"][m]) / (NOISE_TRUE * 1e3))), 6)
        score["mse"][m] = round(mse.mean(), 1)
        score["cov"][m] = round(abs(round(cov.mean(), 2) - 0.95), 6)
        score["nlpd"][m] = round(nlpd.mean(), 2)
        score["mis"][m] = -round(mis.mean(), 2)
        score["hon"][m] = round(hon.mean(), 3)
    for k in ["noise", "mse", "cov", "nlpd", "mis", "hon"]:
        cols[k] = bold_best(cols[k], score[k], DELETION_RULES)
    for m in ORDER:
        lines.append(" & ".join([LABELS[m]] + [cols[k][m] for k in cols]) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}}", rf"\caption{{{caption}}}\label{{{label}}}",
              r"\end{table}"]
    with open(os.path.join(TABLES, tag + ".tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def agreement_table():
    rows = [("S1", "S1"), ("S2", "S2")] + [(f"S3, $k={k}$", f"S3_k{k}") for k in (2, 3, 4, 5)] \
        + [("S3, $k=4$, batch inside $\\mathcal A$", "S3_k4_inside")]
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\small",
             r"\resizebox{\linewidth}{!}{\begin{tabular}{lrrrr}", r"\toprule",
             r"Experiment & Identical decisions & MSE, single ($\times10^{-4}$) " + DOWN
             + r" & MSE, classical ($\times10^{-4}$) " + DOWN + r" & MSE, Gale ($\times10^{-4}$) "
             + DOWN + r"\\", r"\midrule"]
    for name, tag in rows:
        data = load(tag)
        same = sum(r["methods"]["gale"]["keep"] == r["methods"]["group"]["keep"] for r in data["runs"])
        per_seed = {m: values(data, m, "mse") * 1e4 for m in ["single", "group", "gale"]}
        cells = {m: f"{v.mean():.1f}" for m, v in per_seed.items()}

        def better(a, b):
            """a has a smaller MSE than b by more than two paired standard errors."""
            d = per_seed[b] - per_seed[a]
            se = d.std(ddof=1) / np.sqrt(len(d))
            return d.mean() > 2 * se if se > 0 else d.mean() > 0

        for m in per_seed:
            others = [o for o in per_seed if o != m]
            if any(better(m, o) for o in others) and not any(better(o, m) for o in others):
                cells[m] = rf"\textbf{{{cells[m]}}}"
        lines.append(f"{name} & {same}/{len(data['runs'])} & {cells['single']} & {cells['group']}"
                     f" & {cells['gale']}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{Group tests in Gale coordinates against classical group tests with the same"
              r" candidate groups, reference noise and stopping rule. ``Identical decisions'' counts the"
              r" seeds in which both delete exactly the same observations. The single-observation test"
              r" is shown for comparison. Means over 20 seeds. Bold marks a method whose MSE is smaller than"
              r" that of at least one other method by more than two paired standard errors over the seeds,"
              r" and not significantly larger than that of any. Arrows indicate that lower values are"
              r" better.}\label{tab:agreement}",
              r"\end{table}"]
    with open(os.path.join(TABLES, "agreement.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def figure_examples():
    cases = [("S1", "S1", cubic, 0), ("S2", "S2", rough, 0), ("S3 ($k=4$)", "S3_k4", cubic, 0)]
    shown = ["f2i_style", "group"]
    fig, axes = plt.subplots(len(cases), len(shown), figsize=(6.3, 6.6), sharex=True)
    grid = np.linspace(-1, 1, 400)
    for row, (title, tag, f, idx) in enumerate(cases):
        run = load(tag)["runs"][idx]
        x, y = np.array(run["x"]), np.array(run["y"])
        misfit = np.array(run["misfit"])
        gp = GP(run["lengthscale"], run["scale"])
        for col, m in enumerate(shown):
            ax = axes[row, col]
            res = run["methods"][m]
            keep = np.array(res["keep"], int)
            dele = np.setdiff1d(np.arange(len(x)), keep)
            mean, var = gp.predict(x[keep], y[keep], res["noise"], grid)
            sd = np.sqrt(var + res["noise"])
            ax.axvspan(*TARGET, color="0.92", zorder=0)
            ax.fill_between(grid, mean - 1.96 * sd, mean + 1.96 * sd, color=COLOURS[m], alpha=0.2, lw=0)
            ax.plot(grid, f(grid), color="0.3", lw=0.8, ls="--")
            ax.plot(grid, mean, color=COLOURS[m], lw=1.2)
            ax.scatter(x[keep], y[keep], s=9, c=np.where(misfit[keep], "#c0392b", "0.2"), zorder=3)
            ax.scatter(x[dele], y[dele], s=22, marker="x", c=np.where(misfit[dele], "#c0392b", "0.2"),
                       zorder=3, linewidths=0.9)
            ax.set_title(f"{title}: {LABELS[m]}, $\\hat\\sigma^2$ = {res['noise']:.1e}", fontsize=8)
            lo, hi = np.percentile(y, [1, 99])
            pad = 0.4 * (hi - lo)
            ax.set_ylim(lo - pad, hi + pad)
    for ax in axes[-1]:
        ax.set_xlabel("$x$")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_examples.pdf"))
    plt.close(fig)


def figure_masking():
    ks = [2, 3, 4, 5]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9))
    for m in ["oracle", "f2i_style", "single", "group", "gale"]:
        mse = [values(load(f"S3_k{k}"), m, "mse") * 1e4 for k in ks]
        found = [values(load(f"S3_k{k}"), m, "misfit_deleted") for k in ks]
        style = dict(color=COLOURS[m], marker="o", ms=3, lw=1.1, label=LABELS[m])
        if m == "group":
            style.update(ls="--", marker="s", mfc="none")
        axes[0].errorbar(ks, [v.mean() for v in mse], yerr=[v.std() / np.sqrt(len(v)) for v in mse],
                         capsize=2, **style)
        axes[1].plot(ks, [v.mean() for v in found], **style)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("batch size $k$")
    axes[0].set_ylabel(r"MSE in $\mathcal{A}$ ($\times10^{-4}$)")
    axes[1].set_xlabel("batch size $k$")
    axes[1].set_ylabel("fraction of batch deleted")
    axes[1].set_ylim(-0.05, 1.05)
    for ax in axes:
        ax.set_xticks(ks)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(os.path.join(FIGURES, "fig_masking.pdf"))
    plt.close(fig)


def figure_calibration():
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    markers = {"S1": "o", "S2": "s", "S3_k4": "^"}
    for tag, mk in markers.items():
        data = load(tag)
        for m in ORDER:
            ax.scatter(values(data, m, "sharpness").mean(), values(data, m, "coverage").mean(),
                       marker=mk, color=COLOURS[m], s=28, zorder=3)
    ax.axhline(0.95, color="0.5", lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel(r"mean predictive variance in $\mathcal{A}$")
    ax.set_ylabel("coverage of 95% intervals")
    handles = [plt.Line2D([], [], color=COLOURS[m], marker="o", ls="", label=LABELS[m]) for m in ORDER]
    handles += [plt.Line2D([], [], color="0.3", marker=mk, ls="", label=t.replace("_k4", ""))
                for t, mk in markers.items()]
    ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_calibration.pdf"))
    plt.close(fig)


E1_STYLES = {"boundary": ("#c0392b", "o"), "interior": ("#2471a3", "s"),
             "one in cluster": ("#17a589", "^"), "whole cluster": ("#b9770e", "D")}


def figure_power():
    data = load("E1")
    # The candidate-scoring panel is shown once, in the introduction (figure_teaser).
    fig = plt.figure(figsize=(6.3, 2.9))
    grid_spec = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
    ax = fig.add_subplot(grid_spec[0, 0])
    ex = data["examples"]
    x, y = np.array(ex["x"]), np.array(ex["y"])
    grid, f_grid = np.array(ex["grid"]), np.array(ex["f_grid"])
    ax.plot(grid, f_grid, color="0.55", lw=0.9, label="latent function $f$")
    ax.plot(grid, f_grid + np.array(ex["bump_grid"]), color="#b9770e", lw=0.9, ls="--",
            label=rf"$f$ plus bump at $x_0={ex['x0']}$")
    ax.scatter(x, y, s=10, color="0.15", zorder=3, label="observations")
    marks = [("boundary", [ex["boundary"]]), ("interior", [ex["interior"]]),
             ("whole cluster", ex["cluster"])]
    for name, idx in marks:
        colour, marker = E1_STYLES[name]
        ax.scatter(x[idx], y[idx], s=55, facecolors="none", edgecolors=colour, linewidths=1.1, zorder=4,
                   label=name)
    b = ex["boundary"]
    ax.annotate("", xy=(x[b], y[b] + ex["offset"]), xytext=(x[b], y[b]),
                arrowprops=dict(arrowstyle="->", color=E1_STYLES["boundary"][0], lw=1.1))
    ax.text(x[b] - 0.02, y[b] + ex["offset"] / 2, "$+c$", color=E1_STYLES["boundary"][0], fontsize=8,
            ha="right", va="center")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_ylim(min(y.min(), f_grid.min()) - 0.8, max(y.max(), f_grid.max()) + 0.2)
    ax.set_title("(a) one simulated data set on the design", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right", ncol=2)

    ax = fig.add_subplot(grid_spec[0, 1])
    offsets = np.array(data["a"]["offsets"])
    for name, g in data["a"]["groups"].items():
        colour, marker = E1_STYLES[name]
        ax.plot(offsets, g["predicted"], color=colour, lw=1.1, label=f"{name} ($|C|={len(g['index'])}$)")
        ax.plot(offsets, g["empirical"], ls="", marker=marker, ms=3.5, mfc="none", color=colour)
    ax.axhline(data["settings"]["alpha"], color="0.6", lw=0.7, ls=":")
    ax.set_xlabel("offset $c$")
    ax.set_ylabel("power")
    ax.set_title("(b) same offset, different locations", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_power.pdf"))
    plt.close(fig)


def figure_structure():
    data = load("E1")["c"]
    alpha = load("E1")["settings"]["alpha"]
    styles = {"two ends": ("#c0392b", "o"), "two ends + centre": ("#17a589", "^"),
              "uniform": ("#2471a3", "s")}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.8))
    ax = axes[0]
    curv = np.array(data["curvatures"])
    for name, d in data["designs"].items():
        colour, marker = styles[name]
        ax.plot(curv, d["predicted"], color=colour, lw=1.1, label=name)
        ax.plot(curv, d["empirical"], ls="", marker=marker, ms=3.5, mfc="none", color=colour)
        ax.plot(curv, d["res_predicted"], color=colour, lw=0.9, ls="--")
    ax.axhline(alpha, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel("curvature $c$")
    ax.set_ylabel("power")
    ax.set_title("(a) three designs", fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="0.3", lw=1.1, label="GP test"),
                plt.Line2D([], [], color="0.3", lw=0.9, ls="--", label="residual test")]
    ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="upper left")
    ax = axes[1]
    big = np.array(data["large_curvatures"])
    d = data["two_ends_large"]
    ax.plot(big, d["gp_predicted"], color="#c0392b", lw=1.1, label="GP test")
    ax.plot(big, d["gp_empirical"], ls="", marker="o", ms=3, mfc="none", color="#c0392b")
    ax.plot(big, d["res_predicted"], color="#c0392b", lw=0.9, ls="--", label="residual test")
    ax.plot(big, d["res_empirical"], ls="", marker="x", ms=3, color="#c0392b")
    ax.axhline(alpha, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel("curvature $c$")
    ax.set_title("(b) two ends only, large curvature", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_structure.pdf"))
    plt.close(fig)


def figure_teaser():
    """Introduction figure: two results of E1, re-plotted from the cached results."""
    e1 = load("E1")
    data, alpha = e1["c"], e1["settings"]["alpha"]
    styles = {"two ends": ("#c0392b", "o"), "two ends + centre": ("#17a589", "^")}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.6))
    ax = axes[0]
    curv = np.array(data["curvatures"])
    for name, (colour, marker) in styles.items():
        d = data["designs"][name]
        ax.plot(curv, d["predicted"], color=colour, lw=1.1, label=name)
        ax.plot(curv, d["res_predicted"], color=colour, lw=0.9, ls="--")
        ax.plot(curv, d["empirical"], ls="", marker=marker, ms=3.5, mfc="none", color=colour)
    ax.axhline(alpha, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel("curvature $c$")
    ax.set_ylabel("power")
    ax.set_title("(a) can the design test a straight line?", fontsize=9)
    handles, _ = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="0.3", lw=1.1, label="GP test"),
                plt.Line2D([], [], color="0.3", lw=0.9, ls="--", label="residual test")]
    ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="upper left")
    ex = e1["b_example"]
    cand, var = np.array(ex["candidates"]), np.array(ex["variance"])
    ax = axes[1]
    ax.plot(cand, ex["power"], color="#2471a3", lw=1.2, label="power after adding $x$")
    ax.plot(cand, var / var.max(), color="0.45", lw=1.0, ls="--", label="predictive variance (scaled)")
    ax.axhline(ex["choices"]["current"]["predicted"], color="#2471a3", lw=0.7, ls=":")
    ax.axvline(ex["x0"], color="#c0392b", lw=0.7)
    x = np.array(ex["x"])
    ax.plot(x, np.full_like(x, -0.04), "|", color="0.2", ms=5)
    ax.set_ylim(-0.08, 1.05)
    ax.set_xlabel("candidate input $x$")
    ax.set_title("(b) where should the next observation go?", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_teaser.pdf"))
    plt.close(fig)


E2_STYLES = {"average power": ("#2471a3", "o", "-"), "worst-case": ("#17a589", "^", "-"),
             "space-filling": ("#b9770e", "s", "--"), "variance": ("#7f8c8d", "D", "--"),
             "random": ("#000000", "x", ":"), "amplitude": ("#c0392b", "v", ":")}


def figure_unknown():
    data = load("E2")
    budget = data["settings"]["budget"]
    m = np.arange(budget + 1)
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7))
    for rule, (colour, marker, ls) in E2_STYLES.items():
        for ax, key in zip(axes, ["mean", "worst"]):
            vals = np.mean([d[rule][key] for d in data["designs"]], axis=0)
            ax.plot(m, vals, color=colour, marker=marker, ms=3.5, ls=ls, lw=1.1, label=rule)
    axes[0].set_ylabel("prior-weighted mean power")
    axes[1].set_ylabel("worst-case power")
    axes[0].set_title("(a) average over locations", fontsize=9)
    axes[1].set_title("(b) worst location", fontsize=9)
    for ax in axes:
        ax.set_xlabel("observations added")
        ax.set_xticks(m)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=7, frameon=False, loc="lower center", ncol=6)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(FIGURES, "fig_unknown.pdf"))
    plt.close(fig)


E3_STYLES = {"grid": ("#17a589", "s", "grid"), "LHS": ("#b9770e", "o", "Latin hypercube"),
             "random": ("#7f8c8d", "x", "random"), "power": ("#2471a3", "^", "power-optimised")}


def figure_additivity():
    data = load("E3")
    curv = np.array(data["settings"]["curvatures"])
    alpha = data["settings"]["alpha"]
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.9), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    for name in ("grid", "LHS", "power"):
        colour, marker, label = E3_STYLES[name]
        x = np.array(data["designs"][name]["x"])
        ax.scatter(x[:, 0], x[:, 1], s=22, marker=marker, facecolors="none" if marker != "x" else colour,
                   edgecolors=colour if marker != "x" else None, linewidths=1.0, label=label)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.45)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title("(a) designs of 16 inputs", fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc="upper center", ncol=3, handletextpad=0.2, columnspacing=0.8)
    ax = axes[1]
    for name, (colour, marker, label) in E3_STYLES.items():
        d = data["designs"][name]
        r = int(round(np.mean(d["relations"])))
        ax.plot(curv, d["gp_power"], color=colour, lw=1.1, label=f"{label} ({r} relations)")
        if d["res_power"] is not None:
            ax.plot(curv, d["res_power"], color=colour, lw=0.9, ls="--")
        # Simulations exist for single designs only. LHS and random curves are means over draws.
        sims = d.get("simulated", {}) if name in ("grid", "power") else {}
        for c, rej in sims.items():
            ax.plot(float(c), rej["gp"], ls="", marker=marker, ms=3.5, mfc="none", color=colour)
    ax.axhline(alpha, color="0.6", lw=0.7, ls=":")
    ax.set_xlabel("interaction $c$")
    ax.set_ylabel("power")
    ax.set_title("(b) additivity against $c\\,x_1x_2$", fontsize=9)
    handles, _ = ax.get_legend_handles_labels()
    handles += [plt.Line2D([], [], color="0.3", lw=1.1, label="GP test"),
                plt.Line2D([], [], color="0.3", lw=0.9, ls="--", label="residual test")]
    ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig_additivity.pdf"))
    plt.close(fig)


def table_calibration():
    data = load("calibration")
    names = {"single": "Single-observation test", "group": "Group test, classical",
             "gale": "Group test, Gale coordinates"}
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\small", r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Uniform inputs} & \multicolumn{2}{c}{With a cluster}\\",
             r"Method & Any deletion " + DOWN + r" & Deleted " + DOWN + r" & Any deletion " + DOWN
             + r" & Deleted " + DOWN + r"\\", r"\midrule"]
    columns = []
    for cfg in ["null_uniform", "null_cluster"]:
        runs = data[cfg]
        anyd = {k: np.mean([r[k] > 0 for r in runs]) for k in names}
        frac = {k: np.mean([r[k] for r in runs]) / 40 for k in names}
        columns.append(bold_best({k: f"{v:.3f}" for k, v in anyd.items()},
                                 {k: round(v, 3) for k, v in anyd.items()}, list(names)))
        columns.append(bold_best({k: f"{v:.4f}" for k, v in frac.items()},
                                 {k: round(v, 4) for k, v in frac.items()}, list(names)))
    for key, label in names.items():
        lines.append(f"{label} & " + " & ".join(col[key] for col in columns) + r"\\")
    n_runs = len(data["null_uniform"])
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{Calibration of the complete deletion pipeline of S1--S3 when the data contain"
              rf" no misfit observations, over {n_runs} seeds per configuration. ``Any deletion'' is the"
              rf" fraction of seeds in which at least one observation is deleted, which the Bonferroni"
              rf" stopping rule is meant to keep near $0.05$. ``Deleted'' is the mean fraction of the 40"
              rf" observations deleted. Bold marks the best value in each column.}}\label{{tab:calibration}}",
              r"\end{table}"]
    with open(os.path.join(TABLES, "calibration.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def table_e2_sensitivity():
    data = load("E2_sensitivity")
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\small",
             r"\resizebox{\linewidth}{!}{\begin{tabular}{lrrrr}", r"\toprule",
             r"True discrepancy & Average power " + UP + r" & Gain over space-filling " + UP
             + r" & Gain over variance " + UP + r" & Gain over random " + UP + r"\\", r"\midrule"]
    for name, v in data.items():
        cells = [f"{v['average power']['mean']:.2f}"]
        for r in ["space-filling", "variance", "random"]:
            cells.append(f"{v[r]['diff']:.3f} $\\pm$ {v[r]['se']:.3f}")
        lines.append(f"{name} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}}",
              r"\caption{E2, sensitivity to a misspecified alternative. The designs were chosen for bumps of"
              r" height $0.2$ and width $0.04$ with a normal location prior centred at $0.6$, and are evaluated"
              r" here under the true discrepancy in the first column, without re-selection. The second column"
              r" is the prior-weighted mean power of the average-power rule after five additions. The other"
              r" columns are paired differences in mean power between the average-power rule and each"
              r" baseline, with one standard error over the $20$ starting designs.}\label{tab:e2sens}",
              r"\end{table}"]
    with open(os.path.join(TABLES, "E2_sensitivity.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def table_e1_baselines():
    data = load("E1_baselines")
    rows = data["rows"]
    labels = {"true": "True hyperparameters", "lengthscale x0.5": r"Lengthscale $\times0.5$",
              "lengthscale x2": r"Lengthscale $\times2$", "noise sd x0.5": r"Noise s.d.\ $\times0.5$",
              "noise sd x2": r"Noise s.d.\ $\times2$"}
    rules = ["power", "KL", "variance"]
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\small", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Assumed when choosing & Power rule " + UP + r" & KL rule " + UP + r" & Variance rule " + UP
             + r" & Power $-$ variance " + UP + r"\\", r"\midrule"]
    seeds = sorted({r["seed"] for r in rows})
    for key, label in labels.items():
        means = {rule: np.mean([r["settings"][key][rule] for r in rows]) for rule in rules}
        # No bold: the power and KL rules differ by at most 0.01, which is not a meaningful difference.
        cells = {k: f"{v:.2f}" for k, v in means.items()}
        per_design = [np.mean([r["settings"][key]["power"] - r["settings"][key]["variance"]
                               for r in rows if r["seed"] == s]) for s in seeds]
        diff, se = np.mean(per_design), np.std(per_design, ddof=1) / np.sqrt(len(seeds))
        lines.append(f"{label} & " + " & ".join(cells[r] for r in rules) + f" & {diff:.3f} $\\pm$ {se:.3f}\\\\")
    same = 100 * data["summary"]["kl_same_choice"]
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\caption{E1, power of the group test against a bump of specified location after one added"
              r" observation, averaged over $20$ designs and $19$ bump locations. The rules choose under the"
              r" hyperparameters in the first column, and every choice is evaluated under the true model"
              rf" (lengthscale $0.3$, noise variance $0.0025$). The current design has mean power"
              rf" ${data['summary']['current']:.2f}$. The KL rule maximises the Kullback--Leibler divergence"
              rf" between the alternative and the null for all observations, and chooses the same input as the"
              rf" power rule in ${same:.0f}\%$ of the cases with the true hyperparameters. The last column is the"
              r" paired difference between the power and variance rules, averaged per design, with one standard"
              r" error over the $20$ designs.}\label{tab:e1baselines}", r"\end{table}"]
    with open(os.path.join(TABLES, "E1_baselines.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def table_calibration_gp():
    data = load("calibration_gp")
    names = {"single": "Single-observation test", "group": "Group test, classical",
             "gale": "Group test, Gale coordinates"}
    lines = [r"\begin{table}[!htbp]", r"\centering", r"\small", r"\begin{tabular}{lrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Uniform inputs} & \multicolumn{2}{c}{With a cluster}\\",
             r"Method & Fixed & Estimated & Fixed & Estimated\\", r"\midrule"]
    for key, label in names.items():
        cells = []
        for cfg in ["null_uniform", "null_cluster"]:
            for mode in ["fixed", "estimated"]:
                cells.append(f"{np.mean([r[f'{mode}_{key}'] > 0 for r in data[cfg]]):.3f}")
        lines.append(f"{label} & " + " & ".join(cells) + r"\\")
    n_runs = len(data["null_uniform"])
    lines += [r"\bottomrule", r"\end{tabular}",
              rf"\caption{{Calibration under a true GP null: fraction of {n_runs} data sets drawn from the GP"
              rf" in which at least one observation is deleted, against a nominal $0.05$. ``Fixed'' uses the"
              rf" true kernel and noise variance, ``Estimated'' the pipeline of S1--S3 with hyperparameters"
              rf" fitted to the same data.}}\label{{tab:calibrationgp}}", r"\end{table}"]
    with open(os.path.join(TABLES, "calibration_gp.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)
    summary_table("S1", r"S1, a cubic function with a cluster of three shifted observations just outside"
                  r" the target region. Means over 20 seeds, $\pm$ one standard deviation across seeds."
                  r" The true noise variance is $\sigma_0^2=2.5\times10^{-3}$."
                  r" Rows: \emph{Full GP} keeps all observations. \emph{Oracle} deletes exactly the planted"
                  r" misfit observations, which no real method can know, and serves as a best-case reference."
                  r" \emph{F2I-style} deletes greedily to minimise predictive variance in $\mathcal A$."
                  r" The remaining three rows are the single-observation test and the group test in classical"
                  r" and in Gale coordinates. Columns: number of observations"
                  r" kept, refitted noise variance, mean squared error and NLPD in $\mathcal A$, coverage of"
                  r" the $95\%$ predictive intervals, and the fractions of misfit and of honest observations"
                  r" deleted. Arrows indicate whether lower or higher values are better. Bold marks the best deletion"
                  r" rule in each column, where best means closest to the true noise variance and to the"
                  r" nominal coverage $0.95$. The Full GP and Oracle rows are references and are not marked.",
                  "tab:s1")
    summary_table("S2", r"S2, a function that oscillates faster than the kernel can follow for $x<-0.3$."
                  r" Same layout as Table~\ref{tab:s1}.", "tab:s2")
    summary_table("S3_k4", r"S3, a masked batch of $k=4$ shifted observations just outside the target"
                  r" region. Same layout as Table~\ref{tab:s1}.", "tab:s3")
    agreement_table()
    figure_examples()
    figure_masking()
    figure_calibration()
    figure_power()
    figure_structure()
    figure_teaser()
    figure_additivity()
    figure_unknown()
    table_e2_sensitivity()
    table_e1_baselines()
    table_calibration()
    table_calibration_gp()


if __name__ == "__main__":
    main()
