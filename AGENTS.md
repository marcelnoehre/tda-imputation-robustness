# AGENTS.md

Code and results for the paper *The Robustness of Topological Data Analysis Under Imputation-Induced Perturbations* (TMLR submission). The LaTeX source lives in a separate repository (`~/Documents/papers/robustness_tda`); this repository only produces the numbers, tables and figures that go into it.

**Scope rule:** keep only what the latest paper uses. Do not add exploratory notebooks, extra result files or side experiments to the repository.

## Layout

| Path | Purpose |
|------|---------|
| `main.py` | Runs all five experiments of the paper; optional OpenML IDs restrict it to those datasets |
| `src/` | Pipeline: data loading (`data.py`), masking (`missingness.py`), imputers (`imputation.py`), filtrations and vectorisations (`tda.py`), distances (`metrics.py`), baselines (`robustness.py`), experiment drivers (`pipeline.py`, `comparison_metrics.py`, `robustness_pipeline.py`), disk cache (`cache.py`) |
| `src/constants.py` | Single source of truth for dataset IDs, method keys, seeds, worker count and labels |
| `results/*_results.csv` | Raw experiment output, one file per experiment (see below) |
| `notebooks/tmlr_figures.ipynb` | Reads `results/`, writes every results table and figure of the paper to `figures/tmlr/`, and prints the statistics quoted in the text |
| `figures/tmlr/` | Generated output; copy into the paper's `tables/` and `figures/` |
| `sort_results.py` | Sorts a results CSV into canonical order |

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
python main.py                        # all experiments, all datasets (takes long)
python main.py 223 666                # only these OpenML IDs (-1/-2/-3 = torus/swiss roll/sphere)
```

Set `WORKERS` in `src/constants.py` before running; `WORKERS > 1` requires `N_JOBS == 1`. Intermediate results are cached on disk in `.cache/` (keyed by the function arguments), so re-runs only compute what is new.

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

- Manifold embedding in `tadasets` uses an unseeded global random rotation, and data are standardised after it. The synthetic data therefore differ between runs, so the Swiss Roll and Torus rows of `kd.tex` drift slightly.
- `SEEDS` holds 9 seeds, while the paper text says 10. The results CSVs contain one seed per imputer; only the RS/SGT baselines have 9.
- OpenML ID 43000 (`cnn_stock_pred_dji`) returned the same matrix as `hungarian_chickenpox`, so it is excluded from the code and the results.

## Git

Commit or push only when asked. Do not commit `.cache/`, `.venv/` or notebook checkpoints.
