# AGENTS.md

Code and results for the paper *The Robustness of Topological Data Analysis Under Imputation-Induced Perturbations* (TMLR submission). The LaTeX source lives in a separate repository (`~/Documents/papers/robustness_tda`); this repository only produces the numbers, tables and figures that go into it.

**Scope rule:** keep only what the latest paper uses. Do not add exploratory notebooks, extra result files or side experiments to the repository.

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | Runs the experiments of `config/experiments.yaml`; optional OpenML IDs restrict it to those datasets, `-e NAME …` to those experiments |
| `src/` | Pipeline: data loading (`data.py`), masking (`missingness.py`), imputers (`imputation.py`), filtrations and vectorisations (`tda.py`), distances (`metrics.py`), baselines (`robustness.py`), experiment drivers (`pipeline.py`, `comparison_metrics.py`, `robustness_pipeline.py`), disk cache (`cache.py`) |
| `config/experiments.yaml` | Single source of truth for the experimental grid, seeds, worker count and every hyperparameter (read by `src/config.py`) |
| `src/constants.py` | Dataset IDs, method keys and labels |
| `results/*_results.csv` | Raw experiment output, one file per experiment (see below) |
| `notebooks/tmlr_figures.ipynb` | Reads `results/`, writes every results table and figure of the paper to `figures/tmlr/`, and prints the statistics quoted in the text |
| `figures/tmlr/` | Generated output; copy into the paper's `tables/` and `figures/` |
| `sort_results.py` | Sorts a results CSV into canonical order |
| `scripts/select_datasets.py` | Reproduces the OpenML dataset selection (filters and manual ID lists) |
| `scripts/export_archive.py` | Exports all diagrams (HDF5), results and config for the Zenodo archive, after the runs |
| `tests/` | Unit tests on toy clouds (`uv run pytest`, a few seconds) |

## Experiments

| Result file | Imputers | Mechanisms | Missing rates | Filtrations |
|-------------|----------|------------|---------------|-------------|
| `comparison_metrics` (RMSE, MAE) | kNN | MCAR, MAR, MNAR | 13 rates, 1–40% | – |
| `topological_metrics` | all 9 | MAR | 5, 10, 25% | VR |
| `topological_metrics_mr` | kNN | MCAR, MAR, MNAR | 13 rates, 1–40% | VR |
| `topological_metrics_filtration` | kNN | MAR | 5, 10, 25% | VR, DTM, KD |
| `baseline` (SGT, RS, CCA) | – | MCAR, MAR, MNAR | 13 rates, 1–40% | VR |

Distances: Wasserstein, bottleneck, L² on persistence landscapes, L² on persistence images, each in H0–H2. Datasets: 3 synthetic manifolds and 15 OpenML datasets (18 in total).

## Running experiments

**Agents change code; the user runs experiments.** `main.py` and the experiment drivers take hours to days. Do not start them. Add or adjust the calls, check them with a minimal smoke test if needed (one small dataset, one rate, abort once it runs), and report which runs the user has to start and which result files they will write. Running the notebook on existing results (about 2 minutes) and quick checks on single point clouds are fine.

## Commands

```bash
uv sync                               # environment (.venv, Python >= 3.11)
uv run pytest                         # unit tests (filtrations, cutoffs, shared vectorisation grids)
python main.py                        # all experiments, all datasets (takes long)
python main.py 223 666                # only these OpenML IDs (-1/-2/-3 = torus/swiss roll/sphere)
python main.py -e baseline            # only these experiments (names from config/experiments.yaml)
```

Set `runtime.workers` in `config/experiments.yaml` before running; `workers > 1` requires `n_jobs: 1`. Intermediate results are cached on disk in `.cache/` (keyed by the function arguments, not by the config), so re-runs only compute what is new. After changing a hyperparameter in the config, delete the affected cache entries. Progress logging (`src/logger.py::Progress`): every stage logs its task count, a progress line every `runtime.progress_interval` seconds, and each task slower than `runtime.slow_task` seconds with its label; these lines are the basis for runtime estimates.

Regenerate the paper outputs (takes about 2 minutes; nbconvert is not a project dependency):

```bash
cd notebooks
uvx --from nbconvert jupyter-nbconvert --to notebook --execute --inplace tmlr_figures.ipynb
```

The kernel must be the project's `.venv` interpreter. If uvx's own Python is picked up instead, point `JUPYTER_PATH` at a kernel spec whose `argv` uses the absolute path `.venv/bin/python`.

## Analysis conventions (notebook)

- The **dataset is the statistical unit**. Compute per-dataset log-ratios, then report medians over datasets. Use Wilcoxon signed-rank tests (Holm-corrected within a table or comparison family), Friedman mean ranks, and Mann–Whitney for synthetic vs. real-world data.
- Exact zeros are kept finite by adding ε = 1% of the median positive value of each (dataset, filtration, distance, dimension) group. Never drop zeros instead, because that changes the ratios.
- **Growth** is measured by the elasticity β, the log–log slope of error against missing rate. Compare it with RMSE **within each dataset** (Δβ = β_topo − β_RMSE). RMSE/MAE are only a reference and are never used to rank imputers.
- **Retention** is ρ = d_topo / d_SGT.
- Tables are written as LaTeX using the paper's macros (`\WS`, `\BN`, `\PL`, `\PI`, `\Hd{}`, `\SGT`, `\RS`, `\CCA`, `\TOPO`, `\IMP`, `\sig{}`, `\MR`). Five of the six generated tables must stay byte-identical to the paper's copies, so port wording changes made in the paper back into the notebook.
- Figures are drawn at print size in Latin Modern, with no titles (captions live in LaTeX). Widths: `growth_by_family` at `\linewidth`, `redundancy_retention` at `0.62\linewidth`, `pairwise_imputers` at `0.6\linewidth`. The pairwise figure is drawn on a 0.75\linewidth canvas, and its tight bounding box makes it about that size.

## Known issues

- The results in `results/` were produced while the manifold rotation (`tadasets.embed`) was unseeded, so their synthetic data cannot be reproduced exactly. `src/data.py::embed` now draws the rotation from a seeded generator.
- Until the re-runs of PLAN 0.3 are in, the results CSVs contain one seed per imputer (only the RS/SGT baselines under MCAR have 9). Cause, fixed in the code: extra seeds ran only when both the mechanism and the method were stochastic. `SEEDS` now holds 10 seeds (`range(1, 11)`).
- Cache keys before 2026-11-02 hashed the pickled arguments, so a DataFrame loaded from the cache (e.g. a mask) hashed differently from the freshly computed one and every downstream entry (imputations) was recomputed once on the second run. `src/cache.py::_canonical` now hashes content only; `tests/test_cache.py` checks that a second run (serial, then parallel) adds no cache entry. Caches written before this change are not reused.
- OpenML ID 43000 (`cnn_stock_pred_dji`) returned the same matrix as `hungarian_chickenpox`, so it is excluded from the code and the results.

## Git

Commit or push only when asked. Do not commit `.cache/`, `.venv/` or notebook checkpoints.
