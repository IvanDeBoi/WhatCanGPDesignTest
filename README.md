# What Can a Gaussian Process Design Test?

Code for the paper *What Can a Gaussian Process Design Test? Structural and Prior-Based Detection from a Gale Perspective* by Ivan De Boi (University of Antwerp) and Marnix Van Soom (Vrije Universiteit Brussel).

Every figure and table in the paper is produced by the scripts in `code/`. All simulation results are stored in `code/results/`, so the figures and tables can be regenerated without repeating the simulations.

## Requirements

Python 3.10 or later with the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

`fig_gale_example.py` also needs `pdflatex` with TikZ on the path. It uses the Python standard library only.

## Regenerating the figures and tables

From the repository root:

```
python code/make_outputs.py
python code/fig_schematic.py
python code/fig_kernel_spectrum.py
python code/fig_gale_example.py
```

These read the stored results and write the figures to `output/figures/` and the LaTeX tables to `output/tables/`.

## Rerunning the experiments

Each script below overwrites its stored results in `code/results/`. The longer runs print their progress.

| Script | Experiment | Results |
|---|---|---|
| `e1_power.py` | E1: power against offsets, choice of the next observation, straight line against curvature | `E1.json` |
| `e1e2_baselines.py` | E1 and E2: Kullback–Leibler baseline and misspecified hyperparameters | `E1_baselines.json`, `E2_kl.json` |
| `e2_design.py` | E2: five added observations against a bump of unknown location | `E2.json` |
| `e2_sensitivity.py` | E2: designs evaluated under a misspecified alternative | `E2_sensitivity.json` |
| `e3_additivity.py` | E3: testing additivity in two dimensions | `E3.json` |
| `experiments.py` | S1–S3: detection and forgetting (appendix) | `S1.json`, `S2.json`, `S3_*.json` |
| `calibration.py` | Calibration of the deletion pipeline on a fixed cubic | `calibration.json` |
| `calibration_gp.py` | Calibration under data drawn from the GP | `calibration_gp.json` |
| `trajectories.py` | Step-by-step deletion trajectories for S1–S3 | `trajectories.json`, `fig_trajectories.pdf` |

Run `e2_design.py` before `e2_sensitivity.py` and `e1e2_baselines.py`, because both reuse the E2 designs.

`code/results_unbounded/` holds S1–S3 run without the lower bound on the RBF lengthscale (`MIN_LENGTHSCALE` in `gpcore.py`). The paper uses these results to justify that bound.

## Citation

If you use this code, please cite the paper:

```bibtex
@unpublished{deboi2026gale,
  author = {De Boi, Ivan and Van Soom, Marnix},
  title  = {What Can a {G}aussian Process Design Test? {S}tructural and Prior-Based Detection from a {G}ale Perspective},
  note   = {Manuscript},
  year   = {2026}
}
```

## Layout

- `gpcore.py`: RBF kernel, MAP fitting of the hyperparameters and noise, GP prediction.
- `methods.py`: the deletion rules compared in S1–S3 (F2I-style deletion, single-observation test, group tests).
- `make_outputs.py`: turns the stored results into the figures and tables of the paper.

## Licence

The code is released under the MIT licence (see `LICENSE`).
