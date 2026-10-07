"""Forgetting methods compared in the experiments.

Every method receives the training data, the fitted GP (kernel fixed), the
indices of the training points inside the target region, and returns the set
of indices to keep. The noise variance is refitted on the kept points
afterwards by the experiment driver.

Methods
-------
full          keep everything
oracle        delete exactly the planted misfit points
f2i_style     greedy deletion on the F2I objective J (total predictive
              variance over the target set), with an exact noise refit for
              every candidate deletion; stops when no deletion lowers J
single        classical single-observation test: repeatedly delete the point
              with the largest standardised leave-one-out residual while it
              exceeds a Bonferroni threshold
gale          group tests in Gale coordinates: hard-threshold Gale basis from
              the eigenvectors of K with eigenvalue <= reference noise, greedy
              group selection by explained test failure, Bonferroni stopping
group         classical group test T_C with the same
              candidate groups, reference noise and stopping rule as gale
"""

import numpy as np
from scipy.stats import chi2, norm

from gpcore import fit_noise

ALPHA = 0.05


def candidate_groups(x, max_size):
    """All single points and runs of 2..max_size neighbouring inputs."""
    order = np.argsort(x)
    groups = []
    for size in range(1, max_size + 1):
        for start in range(len(x) - size + 1):
            groups.append(tuple(sorted(order[start:start + size])))
    return groups


def reference_noise(gp, x, y, in_target):
    """MAP noise of the GP restricted to the training points in the target region."""
    idx = np.flatnonzero(in_target)
    return fit_noise(gp.k(x[idx], x[idx]), y[idx])


def f2i_objective(gp, x, y, keep, x_target):
    xk, yk = x[keep], y[keep]
    noise = fit_noise(gp.k(xk, xk), yk)
    _, var = gp.predict(xk, yk, noise, x_target)
    return np.sum(var + noise)


def f2i_style(gp, x, y, x_target, **_):
    keep = list(range(len(x)))
    current = f2i_objective(gp, x, y, keep, x_target)
    while len(keep) > 3:
        trials = []
        for i in keep:
            trial = [j for j in keep if j != i]
            trials.append((f2i_objective(gp, x, y, trial, x_target), i))
        best_value, best_i = min(trials)
        if best_value >= current:
            break
        keep.remove(best_i)
        current = best_value
    return keep


def single(gp, x, y, ref_noise, **_):
    n = len(x)
    threshold = norm.ppf(1 - ALPHA / (2 * n))
    keep = list(range(n))
    while len(keep) > 3:
        xk, yk = x[keep], y[keep]
        q = np.linalg.inv(gp.k(xk, xk) + ref_noise * np.eye(len(keep)))
        z = (q @ yk) / np.sqrt(np.diag(q))
        worst = int(np.argmax(np.abs(z)))
        if abs(z[worst]) <= threshold:
            break
        keep.pop(worst)
    return keep


def _greedy_groups(score, groups, n, record=None):
    """Greedy group selection with a Bonferroni stopping rule.

    score(deleted, group) returns (statistic, degrees of freedom) for adding
    group to the deleted set. If record is a list, the selected groups are
    appended to it in the order of selection.
    """
    deleted = set()
    while True:
        available = [g for g in groups if not deleted.intersection(g)]
        if not available or n - len(deleted) <= 3:
            break
        best_p, best_g = 1.0, None
        for g in available:
            stat, dof = score(deleted, g)
            if dof <= 0:
                continue
            p = chi2.sf(stat, dof)
            if p < best_p:
                best_p, best_g = p, g
        if best_g is None or best_p > ALPHA / len(available):
            break
        deleted.update(best_g)
        if record is not None:
            record.append(list(best_g))
    return [i for i in range(n) if i not in deleted]


def gale(gp, x, y, ref_noise, groups, **_):
    n = len(x)
    lam, u = np.linalg.eigh(gp.k(x, x))
    g = u[:, lam <= ref_noise]                 # hard-threshold Gale basis, n x (number of tests)
    p = g @ g.T
    e = p @ y

    def ess(idx):
        idx = sorted(idx)
        if not idx:
            return 0.0
        return float(e[idx] @ np.linalg.pinv(p[np.ix_(idx, idx)]) @ e[idx])

    def score(deleted, group):
        base = ess(deleted)
        gain = ess(deleted.union(group)) - base
        dof = np.linalg.matrix_rank(p[np.ix_(sorted(deleted.union(group)), sorted(deleted.union(group)))]) \
            - (np.linalg.matrix_rank(p[np.ix_(sorted(deleted), sorted(deleted))]) if deleted else 0)
        return gain / ref_noise, dof

    return _greedy_groups(score, groups, n)


def group(gp, x, y, ref_noise, groups, record=None, **_):
    n = len(x)

    def score(deleted, grp):
        keep = [i for i in range(n) if i not in deleted]
        pos = {i: k for k, i in enumerate(keep)}
        xk, yk = x[keep], y[keep]
        q = np.linalg.inv(gp.k(xk, xk) + ref_noise * np.eye(len(keep)))
        alpha = q @ yk
        c = [pos[i] for i in grp]
        t_c = alpha[c] @ np.linalg.solve(q[np.ix_(c, c)], alpha[c])
        return float(t_c), len(c)

    return _greedy_groups(score, groups, n, record)
