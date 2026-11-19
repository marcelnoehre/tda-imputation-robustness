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
| `results/*_results.csv` | Experiment output aggregated over seeds, one file per experiment (see below) |
| `results/*_distances.csv` | Per-seed values behind each `*_results.csv` (archived on Zenodo; the notebook reads only `*_results.csv`) |
| `notebooks/tmlr_figures.ipynb` | Reads `results/`, writes every results table and figure of the paper to `figures/tmlr/`, and writes every number quoted in the text to `figures/tmlr/values.tex` |
| `figures/tmlr/` | Generated output; copy `*.tex` into the paper's `tables/`, `*.pdf` into `figures/`, and `values.tex` into the paper's root |
| `sort_results.py` | Sorts a results CSV into canonical order |
| `scripts/select_datasets.py` | Reproduces the OpenML dataset selection (filters and manual ID lists) |
| `scripts/export_archive.py` | Exports all diagrams (HDF5), results and config for the Zenodo archive, after the runs |
| `tests/` | Unit tests on toy clouds (`uv run pytest`, a few seconds) |

## Experiments

| Result file | Imputers | Mechanisms | Missing rates | Filtrations |
|-------------|----------|------------|---------------|-------------|
| `comparison_metrics` (RMSE, MAE) | kNN | MCAR, MAR, MNAR | MAR: 13 rates, 1–40%; MCAR, MNAR: 5, 10, 25% | – |
| `topological_metrics` | all 9 | MAR | 5, 10, 25% | VR |
| `topological_metrics_mr` | kNN | MCAR, MAR, MNAR | MAR: 13 rates, 1–40%; MCAR, MNAR: 5, 10, 25% | VR |
| `topological_metrics_filtration` | kNN | MAR | 5, 10, 25% | VR, DTM, KD |
| `baseline` (SGT, RS, CCA) | – | MCAR, MAR, MNAR | MAR: 13 rates, 1–40%; MCAR, MNAR: 5, 10, 25% | VR |

Per-mechanism rates are set in the config (`missing_rates: {MCAR: main, MAR: sweep, MNAR: main}`); the notebook compares the mechanisms on the rates they share and computes break-even rates only for mechanisms with the full sweep.

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
python main.py --workers 30 -e baseline       # pool size of this run instead of runtime.workers
python main.py --seeds 1 4 5 -e baseline     # these seeds instead of the config list; writes seeds1-4-5_<name> files
python main.py --seeds 1 4 5 --new-seeds-only -e baseline   # stochastic settings only for seeds 4 and 5 (seed 1 left to another run); writes seeds4-5_<name> files
python main.py --cached-only -e topological_metrics_no_imputation   # skip diagrams not yet in the cache (logged as SKIPPED); re-run without it once they are cached
```

Set `runtime.workers` in `config/experiments.yaml` before running, or pass `--workers` for one run (the config is shared by every machine on the NFS mount); `workers > 1` requires `n_jobs: 1`. Intermediate results are cached on disk in `.cache/` (keyed by the function arguments, not by the config), so re-runs only compute what is new. Every task reads the cache when it starts, so machines can split the extra seeds (e.g. `--seeds 1 4` on one, `--seeds 1 5` on another) without repeating work. `--seeds` must start with the first config seed (deterministic settings run under it only) and leaves experiments with their own `seeds` unchanged; its result files carry a `seeds<list>_` prefix, so the main CSVs come from a final run with the full seed list in the config (served from the cache). After changing a hyperparameter in the config, delete the affected cache entries. Progress logging (`src/logger.py::Progress`): every stage logs its task count, a progress line every `runtime.progress_interval` seconds, and each task slower than `runtime.slow_task` seconds with its label; these lines are the basis for runtime estimates.

Regenerate the paper outputs (takes about 2 minutes; nbconvert is not a project dependency):

```bash
cd notebooks
uvx --from nbconvert jupyter-nbconvert --to notebook --execute --inplace tmlr_figures.ipynb
```

The kernel must be the project's environment. If uvx's own Python is picked up instead (`No module named 'pandas'`), point `JUPYTER_PATH` at a directory with `kernels/python3/kernel.json` whose `argv` is `["uv", "run", "--project", "<repo>", "--with", "ipykernel", "python", "-m", "ipykernel_launcher", "-f", "{connection_file}"]` (the `.venv` itself has no ipykernel).

## Analysis conventions (notebook)

- The **dataset is the statistical unit**. Compute per-dataset log-ratios, then report medians over datasets. Use Wilcoxon signed-rank tests (Holm-corrected within a table or comparison family), Friedman mean ranks, and Mann–Whitney for synthetic vs. real-world data.
- Exact zeros are kept finite by adding ε = 1% of the median positive value of each (dataset, filtration, distance, dimension) group. Never drop zeros instead, because that changes the ratios.
- **Growth** is measured by the elasticity β, the log–log slope of error against missing rate. Compare it with RMSE **within each dataset** (Δβ = β_topo − β_RMSE). RMSE/MAE are only a reference and are never used to rank imputers.
- **Relative topological error** is ρ = d_topo / d_SGT (called retention in earlier versions).
- A non-significant test is never read as "the same": claims that two quantities agree (e.g. a growth rate close to $e_\infty$, a filtration as sensitive as VR) are backed by a distribution-free 95% CI of the median over datasets (`median_ci`, order statistics). Growth comparisons are checked with the fit-free two-point elasticity log(d(40%)/d(5%))/log 8, because many BN/PL curves saturate (R² < 0.8).
- In the filtration comparison (RQ6), (dataset, measure, dimension) cells in which a filtration has no features (kNN distance 0 at every rate; mostly KD in H2) are left out for all three filtrations: they give β = 0 by construction and a ρ set by ε (user decision 2026-11-19). Partial zeros keep the ε rule above.
- Tables are written as LaTeX using the paper's macros (`\WS`, `\BN`, `\PL`, `\PI`, `\Hd{}`, `\SGT`, `\RS`, `\CCA`, `\TOPO`, `\IMP`, `\sig{}`, `\MR`). All generated tables must stay byte-identical to the paper's copies, so port wording changes made in the paper back into the notebook.
- **Numbers in the text** are never typed by hand: each section stores them with `val(name, text)` (helpers `tnum`, `tp`), and the last cell writes them as `\newcommand{\v<Name>}{…}` to `values.tex`, which the paper inputs from `macros.tex`. To update the paper after new runs: run the notebook, copy `figures/tmlr/*` into the paper, then check the `% CHECK` comments in `results.tex`/`appendix.tex` (qualitative claims that depend on the values).
- The main text shows results as **figures**; the result tables are in the appendix. The main content (through the conclusion) stays within **12 pages** (TMLR); new material goes to the appendix.
- **Revision marks** in the paper (`macros.tex`): `\added{}` (blue, new), `\changed{}` (red, changed), unmarked = as submitted (the October 2026 version, `main.pdf` of 2026-10-12). `\revisionmarksfalse` prints the clean version.
- Figures are drawn at print size in Latin Modern, with no titles (captions live in LaTeX). Widths are listed in the notebook's first cell (`\linewidth` unless stated; `filtration`, `filtration_sweep` at `0.75\linewidth`, `tightness` at `0.62\linewidth`, `pairwise_imputers` at `0.6\linewidth`). The pairwise figure is drawn on a 0.75\linewidth canvas, and its tight bounding box makes it about that size.

## Known issues

- The results in `results/` were produced while the manifold rotation (`tadasets.embed`) was unseeded, so their synthetic data cannot be reproduced exactly. `src/data.py::embed` now draws the rotation from a seeded generator.
- The results CSVs written from 2026-11-16 on carry 5 seeds for every stochastic setting (`seeds: [1, 2, 3, 4, 5]`; earlier files had one seed per imputer because of a loop bug, fixed in `missingness.repeated`). An experiment can set its own `seeds` (`mice_spread` uses 10 draws).
- Cache keys before 2026-11-02 hashed the pickled arguments, so a DataFrame loaded from the cache (e.g. a mask) hashed differently from the freshly computed one and every downstream entry (imputations) was recomputed once on the second run. `src/cache.py::_canonical` now hashes content only; `tests/test_cache.py` checks that a second run (serial, then parallel) adds no cache entry. Caches written before this change are not reused.
- Worker processes are started with **spawn** (`src/utils.py::process_pool`). With fork, a worker inherits the OpenMP thread pools of the main process (scikit-learn, PyTorch) and deadlocks at its first OpenMP call (observed: all workers idle inside scikit-learn's `radius_neighbors`, called by giotto's Vietoris–Rips). Each worker is limited to one thread (`threadpoolctl`), otherwise scikit-learn's OpenMP code uses every core in every worker; the main process keeps all threads for the sequential deep imputers. Spawned workers do not inherit module-level changes; the cache location is passed through the environment variable `TDA_CACHE_DIR`.
- Cache keys hash the data bitwise, and the generated data (manifolds and `synthetic_*`, through `src/data.py::embed`, which uses `np.linalg.qr`) differ in the last bits between the Xeon Phi machines (lancelot, parceval) and the other machines (zugspitze, pumbaa, watzmann). Their manifold and synthetic cache entries therefore only hit on the CPU type that wrote them (observed 2026-11-16: the final run on zugspitze recomputed the manifold imputations of seeds 4–5 computed on parceval); real datasets are unaffected. Run experiments on generated data on one machine type, and the final run on the type that computed them. Do not change `embed` to fix this; it would invalidate the manifold caches.
- OpenML ID 43000 (`cnn_stock_pred_dji`) returned the same matrix as `hungarian_chickenpox`, so it is excluded from the code and the results.

## Git

Commit or push only when asked. Do not commit `.cache/`, `.venv/` or notebook checkpoints.
