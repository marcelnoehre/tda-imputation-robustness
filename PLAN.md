# PLAN.md — Revision plan

Revision of the code, experiments and paper (`~/Documents/papers/robustness_tda`) along six review points. Each item lists what the code does now (verified on 2026-10-30 against `ce151d0`), the work to do, and how we know it is done. Per `AGENTS.md`, any new result must end up in the paper; drop analyses that don't make it in.

Status legend: `[ ]` open · `[~]` in progress · `[x]` done · **[user]** = run by the user

**Division of work: agents only change code, they do not run experiments.** Running `main.py` (or any experiment driver) takes hours to days and is done by the user. For every task marked **[user]**, an agent adds the call to `main.py` (or the config) and checks that it works, e.g. with a tiny smoke test on one small dataset and one rate, aborted once it runs. It then stops and reports which runs are needed and which result files they will write. Unit tests, quick checks on single clouds, and running the notebook on existing results are fine.

---

## 0. Prerequisites found while preparing this plan

These affect items 3, 5 and 6 and the existing growth results, so resolve them first.

### 0.1 RMSE/MAE are computed over the full matrix, not the masked entries
- **Now:** `src/comparison_metrics.py::compute_distances` passes the full original and imputed arrays to `compute_rmse`/`compute_mae` (`src/metrics.py`). Observed cells contribute zero error. The paper (`setup.tex`) says "computed … on the masked entries".
- **Consequence:** with masked fraction f ≈ MR, RMSE_full = √f · RMSE_masked and MAE_full = f · MAE_masked. This adds ≈ +0.5 to β_RMSE and ≈ +1 to β_MAE. It likely explains why "MAE grows faster than RMSE (1.18 vs 0.65)" in RQ3, and it shifts every Δβ in Table `growth` and Figure `growth_by_family`.
- [x] Compute RMSE/MAE on masked entries only (pass the mask from `introduce_missingness`). Keep the full-matrix variant under a distinct key only if the paper reports it.
  - Resolved (user decision 2026-10-30): **keep the full-matrix RMSE/MAE**; fix the paper wording instead. Reason: the topological distances compare whole diagrams, so the matching imputation error is the total perturbation ‖X−X̂‖_F/√(nd), not a per-cell average. Verified: mdatagen masks exactly MR·n·d cells (torus, all 3 mechanisms, 1/10/40%), so RMSE_full = √MR·RMSE_masked and MAE_full = MR·MAE_masked. With masked errors, β_RMSE (real-world) would drop 0.65→0.15 and β_MAE 1.18→0.18, reversing Findings 3/5 (simulation by rescaling the existing CSV). Code unchanged apart from a comment in `metrics.py`; no result files stale.
- [x] ~~**[user]** Re-run `comparison_metrics`~~ not needed for 0.1 (the re-run for 0.3 seeds and 6 `MAXDISP` covers it).
- **Done when:** the paper describes RMSE/MAE as computed over all entries (see "Paper changes").

### 0.2 Landscape and image grids are fitted per diagram
- **Now:** `src/tda.py::persistence_landscape` / `persistence_image` call giotto-tda `fit_transform` on a single diagram. The sampling range (and for PI, the image extent) is therefore fitted separately for the original and for each imputed diagram. `landscape_l2_distance` then subtracts vectors that may lie on different grids, using Δt from the original only.
- [x] Verify with one dataset: compare the fitted `samplings_` of the original and an imputed diagram.
  - Torus, MAR 25%, kNN, VR: grids differ (PL H1 [0.025, 0.248] vs [0.024, 0.224]; PL H2 [0.085, 0.248] vs [0.068, 0.233]; PI ranges likewise). Two further bugs: `landscape_l2_distance` took Δt from the **H0** range for all dimensions; and giotto PI divides each image by its pixel area, so per-diagram grids rescaled the PI values themselves. giotto clips PI points outside the grid onto its border (no mass lost); PL values outside the original's range are cut off.
- [x] If they differ, fit the vectoriser once per (dataset, filtration, dimension) on the original diagram, and transform both diagrams with that fitted object.
  - `tda.persistence_landscape/persistence_image(pd_fit, pd)` fit once on the original diagram and return both vectors plus the grid; `metrics.py` checks that both vectors match the grid and takes Δt from that dimension's grid. The original is passed as the raw normalised diagram (new key `DIAGRAM`), so existing cached diagrams stay valid; the old cached original PL/PI vectors are simply no longer used. Verified: the original's landscape is unchanged (allclose to the old fit_transform); smoke runs of `experiment` and `experiment_baselines` on the torus (MAR 10%). Torus PI H0 kNN dropped from 3.4e5 to 8.7e2, so **PL/PI columns of all four topological result files are stale**, and RQ1/RQ2 PI statements may change.
- [ ] **[user]** Re-run every experiment so the PL/PI columns are recomputed; diagrams are cached, so only vectorisation is recomputed. (Merged into the consolidated runs under "Waiting for user".)
- **Done when:** both vectors share one grid by construction (assert in `metrics.py`).

### 0.3 Seeds
- **Now:** `SEEDS = range(1, 10)` gives 9 seeds; the paper says 10. `topological_metrics_results.csv` has `n_seeds = 1` for every imputer, including RF/MICE/GAIN/TabCSDI/OT-Impute. Only RS/SGT baselines have 9.
- [x] Find out why the stochastic imputers ran with one seed (was `SEEDS` reduced for the run?).
  - Not a reduced `SEEDS`, but a loop bug: every iterator ran extra seeds only if **both** the mechanism and the method were stochastic (`not det_mt or seed == S0` and `not det_method or seed == S0`). So all MAR/MNAR × imputer runs, kNN × MCAR, CCA × MCAR and RS/SGT × MAR/MNAR had one seed. Fixed with `missingness.repeated(seed, mt, method_deterministic)`: a pair repeats over all seeds as soon as either part is stochastic; deterministic MAR/MNAR masks are computed once and shared by all seeds. Verified by smoke runs (torus, MCAR/MAR 10%, 2 seeds, WORKERS 1 and 4) of all three drivers: n_seeds = 1 only for deterministic × deterministic pairs. Stale: n_seeds and all means in every result file.
- [x] Decide (user): re-run with 10 seeds. User chose `SEEDS = range(1, 11)` (keeps the cache of seeds 1–9; `range(10)` would recompute everything). Set in `constants.py`. The re-runs are listed under "Waiting for user → Runs needed".

---

## 1. Kernel-distance filtration

- **Now (confirmed):** `src/tda.py::kernel_distance` builds **VR on the RKHS metric** D_K(x,y) = √(2 − 2k(x,y)), with a Gaussian kernel and Silverman bandwidth h = (n(d+2)/4)^(−1/(d+4)). This is a change of metric, not the kernel distance-to-measure of Phillips et al. / Chazal et al. `max_edge_length_for` uses the same metric. The appendix table `kd` (85–99.7% of pairs at the √2 ceiling) documents the resulting degeneracy.
- **Target construction:** a weighted Rips filtration with Euclidean distances and weights f(x_i) = √(κ(P,P) + k(x_i,x_i) − 2κ(P,x_i)), where κ(P,x) = (1/n) Σ_j k(x_j,x). This is the Anai et al. construction used for DTM, with the DTM replaced by the kernel distance, so DTM and KD differ only in the density estimator.

Tasks:
- [x] Add `kernel_distance_to_measure(data, h)` to `src/tda.py`, returning the weights f(x_i). Note k(x,x) = 1 for the Gaussian kernel.
  - User decision: the vertex function is **h·d_K(P,x)** (distance-like, 1-Lipschitz in the Euclidean distance like the DTM; h→∞ gives the distance to the centroid, h→0 a constant, i.e. shifted VR). Unscaled d_K ∈ [0, √2] would be small against the edge lengths (torus, h = median: 2·d_K ≤ 1.6 vs edges up to ~3). Implemented as `kernel_distance_to_measure(distances, h)`; unit tests check both limits and the 1-Lipschitz property.
- [x] Build the filtration. Either:
  - use `gudhi.weighted_rips_complex.WeightedRipsComplex(distance_matrix, weights)` and compute persistence with GUDHI, converting to the giotto diagram format used by `transform_pd`; or
  - use giotto `WeightedRipsPersistence` with precomputed weights, if it accepts them.

  Check that the weight convention (radius vs. diameter, p = 2) matches `distance_to_a_measure` exactly.
  - One builder `tda.weighted_rips_complex(data, tda, cutoff, bandwidth)` (giotto `WeightedRipsPersistence`, precomputed Euclidean distances, callable vertex weights 2·f, p = 2) for DTM and every KD variant; `vertex_weights` is the only difference. giotto-ph doubles Anai et al.'s values (diameter convention), and its DTM is 2·√(Σ_kNN d² / (k+1)); our `distance_to_measure` reproduces it, so DTM through the shared builder equals giotto's own `weights='DTM'` (unit test `test_dtm_matches_giotto`).
- [x] Bandwidth: median heuristic h = median pairwise Euclidean distance (outcome-independent) as the main setting. Add a sweep h ∈ {0.25, 0.5, 1, 2} × median, plus Silverman as a documented column.
  - In `config/experiments.yaml` (`filtration.kernel_distance.bandwidths`). User decision: h is computed **once from the original cloud** and shared by all its perturbed versions (`tda.filtration_params`).
- [x] `max_edge_length_for(KD)`: use Euclidean distances like DTM (the weights shift filtration values; reuse the DTM handling of `effective_mel`).
  - Found: the old DTM rule max(VR cutoff, max k-NN distance) lies below many DTM vertex values, so **42% (torus) / 46% (sphere) of the vertices never entered** and torus/sphere had no H2 bars (real data: 0–1%). A weighted-MST rule (2.5·τ₀ of the weighted complex) was too expensive (torus DTM 34 s instead of 0.2 s; "W + VR cutoff" took 707 s for one 1000-point sphere KD diagram; a 5000-point DTM diagram ran > 40 min). User decision: cutoff = **max(VR cutoff, largest doubled vertex value)** for DTM and KD, so every vertex enters; H0/H1 bars equal those of the large cutoffs on torus/sphere, only late H2 bars are clipped, ≤ 1.5 s per 1000-point diagram. The cached cutoff function was renamed (`get_filtration_cutoff`) so that stale DTM cutoffs are not reused. **DTM results are stale** (`topological_metrics_filtration`).
  - User decision (also affects VR): **every diagram of a dataset uses the original cloud's cutoff** (before, each imputed/baseline cloud used its own MST cutoff). Per-cloud cutoffs clip long bars at different values, so manifold BN H2 mostly measured the cutoff difference (sphere MAR 5%: 0.0173 own vs 0.0017 shared; torus 0.0072 vs 0.0029); H1/WS and real data barely changed. Also needed for d_B ≤ 2e_∞ (item 6). `compute_filtration_params` computes (cutoff, params) once per (dataset, filtration); originals keep their cache keys, **every imputed/baseline diagram is recomputed**.
- [x] New method keys in `constants.py` (e.g. `KDM` plus a bandwidth-variant suffix); keep the old `KD` under a renamed key (`KD_RKHS`) only if the paper keeps it as a reference column.
  - `KDM` (h = median), `KDM_H025`, `KDM_H05`, `KDM_H2`, `KDM_SILVERMAN`. User decision: the old RKHS-metric `KD` is **dropped** (code and paper). Verified: 16 unit tests (`uv run pytest`); `main.py -1 -e topological_metrics_filtration` runs all 7 filtrations on the torus in 27 s.
- [ ] **[user]** Run `topological_metrics_filtration` (kNN, MAR, 5/10/25%) for the new KD at all bandwidths (the agent adds the call to `main.py`). (In config; see "Runs needed".)
- [ ] Notebook (runs): Table `filtration` with the new KD; the appendix bandwidth sweep replaces or extends `kd.tex`.
- [ ] Paper: rewrite the KD paragraph in `setup.tex` (construction plus bandwidth) and RQ6 in `results.tex`; update the `limitations.tex` sentence on bandwidth. (setup text done; RQ6/limitations need the runs, see "Paper changes")
- **Done when:** DTM and KD share one code path apart from the weight function (unit test: on a toy cloud, both run through the same filtration builder), and Table `filtration` reports median-heuristic KD with the sweep in the appendix.

---

## 2. Reproducibility specification and code release

### 2.1 Pinning and configuration
- **Now:** `uv.lock` pins everything, but `requirements.txt` is unpinned. Installed versions: gudhi 3.13.0, giotto-tda 0.6.2, scikit-learn 1.3.2, mdatagen 0.2.0, torch 2.13.0, persim 0.3.8, tadasets 0.2.2, numpy 1.26.4, scipy 1.17.1, pandas 3.0.5, statsmodels 0.14.6. Experimental choices are split across `src/constants.py`, `main.py` and hard-coded defaults (e.g. DTM `m = 0.05` inside `tda.py`, GAIN/TabCSDI/OT arguments in `imputation.py`).
- [x] Generate a pinned `requirements.txt` from the lock (`uv export --format requirements-txt --no-hashes`).
  - Generated with `--no-dev`; `pyyaml` added to `pyproject.toml`/`uv.lock` for the config.
- [x] Add `config/experiments.yaml` holding the full grid and every hyperparameter. `main.py` reads it; `constants.py` keeps only keys and labels. Remove hard-coded hyperparameters from `src/`.
  - `src/config.py` loads it (`SEEDS`, `WORKERS`, `N_JOBS`, per-section parameter dicts, `experiments()`); `main.py` runs the config's experiments in order and takes `-e NAME …` to run a subset. Moved out of `src/`: RF/MICE/GAIN/TabCSDI/OT hyperparameters (incl. GAIN Adam lr = 1e-3, TabCSDI architecture, OT ε scale, init noise), DTM m/p, MST cutoff factor/subsample/seed, PL/PI parameters, missing-feature fraction, RS fraction, min. baseline points. Removed the dead `TABCSDI_EPOCHS` default and the unused non-Chazal DTM branch. Verified identical behaviour against `HEAD` on a 300×8 cloud: MAR mask and kNN/RF/MICE/GAIN/TabCSDI/OT imputations bit-identical; VR/DTM/KD diagrams and cutoffs bit-identical; `experiments()` reproduces the old `main.py` grid exactly; `main.py -1 -e topological_metrics_filtration` runs. Cache keys hash function arguments, not the config, so existing cache entries stay valid (documented: delete affected entries after changing a hyperparameter).
- [x] Seed everything from the config. This includes manifold sampling and the currently **unseeded rotation** in `tadasets.embed` (generate the rotation with `np.random.default_rng(seed)` in `src/data.py`, which also fixes the `kd.tex` drift).
  - `data.embed(data, ambient, seed)` reproduces `tadasets.embed` with the rotation drawn from `default_rng([seed, 1])` (own stream, independent of the sample). Manifolds are now identical across calls (rank 3 kept). Notebook re-run: all printed statistics identical; only the manifold rows of `kd.tex` change (new fixed rotation). Stale: manifold rows of every result file (their rotation is not reproducible); the manifold cache misses once.
- [ ] Zenodo (deferred to step 7, once the grid is final): archive original and imputed diagrams (from `.cache/`, exported to a documented HDF5/NPZ layout), all `results/*.csv`, and the config. Add the DOI to `README.md` and the paper.
  - Per-seed distances: every experiment now also writes `results/<experiment>_distances.csv` (one row per dataset, mechanism, rate, method, filtration, dimension, seed; `store_distances` in `pipeline.py`/`comparison_metrics.py`), copied into the archive with the other CSVs. Test: per-seed counts equal `n_seeds` and their mean equals the aggregated mean. `sort_results.py` keeps the seed column.
  - Script done: `scripts/export_archive.py OUT [-e NAME …]` walks every topological/baseline experiment of the config through the cache and writes `OUT/diagrams.h5` (`/<dataset>/<filtration>/original` and `/<dataset>/<filtration>/<mechanism>/<rate>/<method>/seed_<seed>`, padding removed, normalised; group attributes cutoff, diameter, bandwidth), plus `results/*.csv` and the config. Verified on the torus (filtration experiments: 148 diagrams, all cache hits, 24 s). **[user]** after all runs: `python scripts/export_archive.py archive/`, upload to Zenodo, then the DOI goes into `README.md` and the paper (`% TODO(PLAN 2.1)` in `setup.tex`).

### 2.2 Table "Experimental grid" (paper, appendix or setup)
- [x] Rows: datasets, mechanisms, rates, imputers, filtrations, measures, dimensions, references, seeds. Columns: levels, and which RQ/table uses them. Generate it from `config/experiments.yaml` in the notebook, so it can't drift from the code.
  - New notebook section "experimental grid" writes `figures/tmlr/grid.tex` from `src/config.py` (levels grouped by the RQs of the experiments using them; experiment → RQ map `USED_IN` in the notebook). Copied to the paper (`tables/grid.tex`, new appendix subsection, referenced from `setup.tex`); paper compiles. Full notebook re-run: all other tables unchanged.

### 2.3 State explicitly in the paper (current values from the code; verify the marked ones)

| Item | Current implementation | Action |
|------|------------------------|--------|
| PH on 3.5k–5k points | No point subsampling. VR via giotto/ripser with `collapse_edges=True` and `max_edge_length` = 2.5 × the longest MST edge of a 500-point subsample (seed 42) of the original (`_max_edge_length`) | State it. Check how giotto treats essential bars (expected: deaths set to `max_edge_length`); state whether they are clipped before WS/BN. **Checked:** giotto (`infinity_values=None`) sets the death of every feature alive at the cutoff to the cutoff, and `reduced_homology=True` drops the one infinite H0 bar; WS/BN see the clipped bars. **Fixed (item 1 decision):** imputed/baseline clouds used their *own* cutoff; now all use the original's. DTM/KD cutoff = max(VR cutoff, largest vertex value) |
| Diagram scaling | Births/deaths divided by the diameter of the original cloud (`normalize.py`) | State it (it matters for item 6) |
| Wasserstein | `persim.wasserstein`: optimal matching, sum of L∞ costs (p = 1, q = ∞) | Verify, then state p and the ground metric. **Verified: p = 1 with the Euclidean ground metric (q = 2)**, diagonal cost (d − b)/√2, non-finite deaths ignored. BN (GUDHI) uses q = ∞. `ph.tex` updated (item 6) |
| PI | giotto `PersistenceImage` defaults (`sigma=0.1`, `n_bins=100`, `weight_function=None`), fitted per diagram (see 0.2) | Fix 0.2, then state σ, resolution and weight. Grid now fitted on the original; points outside it are clipped onto its border; images are divided by the pixel area. **Fixed (user decision):** σ = 2 pixels of the persistence axis of each original diagram's grid, per dimension (`image.sigma_pixels`), so the image scales with the diagram (unit test: scaling a diagram by c rescales its image by 1/c² only, up to pixel assignment). Torus original vs kNN-imputed image correlation now H0 0.995 / H1 0.95 / H2 0.04 (before 0.92–1.00 blobs) |
| PL | giotto `PersistenceLandscape` defaults (`n_layers=1`, `n_bins=100`), fitted per diagram (see 0.2) | Fix 0.2, then state layers and resolution |
| DTM | k = ⌈0.05 n⌉ neighbours, p = 2, giotto `WeightedRipsPersistence(weights='DTM')` | State it. Now our own weights through the shared builder (identical to giotto's 'DTM'): f = √(Σ_kNN d² / (k+1)) (r = 2, the point itself counted in the mass), vertex values doubled (diameter convention), p = 2 |
| MCAR | Random half of the columns (`random.seed(seed)`); mdatagen `mMCAR` with the rate scaled by d/(d/2), so the overall rate equals MR | State it |
| MAR | mdatagen `mMAR(n_xmiss=d//2).correlated(MR)`, no seed (deterministic) | Verify in mdatagen 0.2.0 which columns are masked, which drive the mechanism, and whether MR is overall or per column. **Verified:** features are paired by correlation; in each pair one feature is masked in the rows where its partner has its lowest 2·MR values (1.5·MR in a triple), so the overall rate is exactly MR and the masked column rate 2·MR. Deterministic (reproducible, independent of global RNG state) |
| MNAR | mdatagen `mMNAR(n_xmiss=d//2).random(MR, deterministic=True)` | Same verification. **Verified:** d/2 features are drawn with `np.random.choice` from the **unseeded global RNG**; each is masked at its own lowest n·d·MR/(d/2) values (self-masking), overall rate exactly MR. `deterministic=True` refers to self-masking, not to the RNG. **Fixed (user decision):** the column choice is seeded with the run seed (global state restored afterwards), so MNAR is stochastic like MCAR (10 masks). The cached mask function was renamed (`apply_mask`) so the unseeded masks are not reused; MCAR/MAR masks recompute identically. **All MNAR results are stale** |
| Paired design | Masks are computed once per (seed, dataset, mechanism, rate) and shared by all imputers and baselines (`introduce_missingness`) | State it. Since 0.3, the deterministic MAR mask is shared by all seeds |
| Standardisation | `StandardScaler` on the complete data **before** masking (`data.py::preprocess`): a leak | Preferred: standardise on observed entries after masking and evaluate in the original scale. Otherwise state the leak and argue it is negligible (z-scoring is a per-column affine map). **Fixed (user decision):** `imputation.impute` z-scores each masked dataset with its observed-entry means/SDs, imputes and maps back; errors stay in the complete-data z-score scale (the evaluation geometry). Constant keeps filling the complete-data mean (0) and is labelled an **oracle reference** (user decision). Mean/median/RF/MICE/GAIN/TabCSDI/OT are (nearly) invariant to this per-column affine change; kNN changes, mostly under MAR/MNAR. Cached function renamed (`apply_imputation_observed_scale`): **all imputations are recomputed**. Unit tests in `tests/test_imputation.py` |
| kNN | `KNNImputer(n_neighbors=min(⌊√n⌋, n/2))` | State it |
| RF | `IterativeImputer(estimator=RandomForestRegressor(n_estimators=30, max_depth=15))`, default `max_iter=10`; not the missForest port | State it; adjust "missForest" in `imputers.tex` |
| MICE | statsmodels `MICEData`, 5 iterations, a **single draw** (`mice_imp.data`) | State "single imputation from the chained equations" |
| GAIN | batch 128, hint 0.9, α = 100, iterations clip(300·⌈n/128⌉, 2000, 10000) | State it, plus the learning rate (read from the code) |
| TabCSDI | 20 diffusion steps, Adam lr 1e-3, batch 128, epochs max(20, ⌈2000/steps per epoch⌉) (the `TABCSDI_EPOCHS` default is overridden) | State it; clean up the dead default |
| OT-Impute | 500 iterations, batch 128, RMSprop lr 1e-2, 20 Sinkhorn iterations | State it, plus ε |
| Seeds | See 0.3 | — |
| OpenML filter | Active, no missing values, no symbolic features, 8–50 numeric features, 500–5000 instances, newest version per name, then manual removal of non-numeric/categorical IDs (lists in `git show aa0575b:notebooks/find-datasets.ipynb`), then dropping labels/date columns (`data.py::preprocess`) | Restore the filter as a script (`scripts/select_datasets.py`) with the ID lists; state it |
| Air Quality, 5000 rows | OpenML 46762 has exactly 5000 × 9 natively; it sits at the upper filter bound | State it |

- [x] State the table items in the paper.
  - New appendix subsection "Implementation Details" (datasets/filter, masking, PH incl. cutoff and clipped essential bars, distances, PL/PI grids, imputer hyperparameters, all read from the code/config); `imputers.tex`: RF = iterative per-feature forest regression (not the missForest port), MICE = single draw; `tables/datasets.tex`: feature counts corrected to the analysed data (the table listed raw OpenML columns incl. target/label/date, e.g. Boston Weather 24 → 20; all but Concrete were off by 1–4) with a caption note. `scripts/select_datasets.py` restores the OpenML filter with the ID lists (and `MAX_DID = 47038`, since newer uploads now pass the automatic filter); verified that it reproduces exactly the 15 datasets of `src/constants.py` (counts 177 → 152 → 148 → 111 → 63 → 50 → 15, as in the original notebook). Open: the PI σ and standardisation items (`% TODO` in the appendix; see Questions).
- [ ] Paper: a reproducibility paragraph with the code link and Zenodo DOI, and a compute statement (hardware, core-hours). Collect the core-hours from the run logs, or estimate them from `src/logger.py` timings.
  - Paragraph added to `setup.tex` with the GitHub link; `% TODO(PLAN 2.1)` for the DOI and `% TODO(PLAN 2.3)` for the compute statement. **[user]**: core-hours/hardware (no run logs on this machine).

---

## 3. Noise-floor confound

- **Now:** the noise floor is an 80% random subsample (`robustness.py::random_subsample`, `SUBSAMPLE_FRACTION = 0.8`). It changes the cardinality, which mainly moves H0 and the PI distances (see RQ1).
- [x] **RMSE-matched jitter** (`robustness.py::matched_jitter`): perturb only the masked entries with N(0, σ²), where σ is the masked-entry kNN RMSE at the same (dataset, mechanism, rate, seed). Requires 0.1. The ratio d_topo(imputer)/d_topo(jitter) shows whether imputation error is topologically more or less harmful than unstructured error of the same size.
  - Scope: kNN, VR, MAR, 13 rates, no new imputations (reuse the cached kNN output for σ).
  - Add it as a baseline method in `robustness_pipeline.py` (needs access to the kNN imputation of the same mask).
  - `MATCHED_JITTER`: σ = masked-entry RMSE of the kNN imputation of the same mask, read through the cached `apply_imputation` (no new imputations once `topological_metrics_mr` has run), noise seeded per run seed.
- [x] **Fresh sample** (manifolds only): resample n points with a different seed (same rotation) as the true sampling-noise floor.
  - `data.fresh_sample(key, seed)`: sample seed [42, seed], same rotation, scaled with the original's `StandardScaler`; skipped for real datasets.
- [x] **Subsample at 70/80/90%:** parametrise `random_subsample` by fraction. Report MR* and the ordering in Table `baseline` for each fraction.
  - `RANDOM_SUBSAMPLE` stays the 80% key; `RANDOM_SUBSAMPLE_70/90` added (`baselines.subsample_fractions` in the config). New experiment `baseline_reference` (MAR, 13 rates, VR) → `results/baseline_reference_results.csv`; `experiment_baselines` now takes `baseline_methods`. Verified: smoke runs (torus + concrete, MCAR/MAR 10%, 2 seeds, WORKERS 1 and 4): identical values in serial and parallel mode, fresh sample only on manifolds.
- [ ] **[user]** Run the extended `experiment_baselines` (jitter, fresh sample, 70/80/90% subsample) once the agent has added the calls to `main.py`. (See "Runs needed".)
- [ ] Notebook (runs): add jitter and fresh-sample columns to Table `baseline`, or a new table. Check that MR* and the Table `baseline` ordering hold; if not, rewrite the RQ1 claims against the matched reference.
- **Done when:** the RQ1 statements are tied to a same-cardinality reference and the subsample sensitivity is reported.

---

## 4. Sensitivity for DTM and KD (RQ6)

- **Now:** baselines (SGT, RS, CCA) exist only for VR, so Table `filtration` reports only β (stability), with no sensitivity.
- [ ] **[user]** Run `experiment_baselines('baseline_filtration', [MAR], [5, 10, 25], [DTMC, KD_new], …)`. The function already takes `tda_methods`; the same masks give a paired design with Table `filtration`.
  - Added to the config as `baseline_filtration` (DTM, KDM; RS and SGT; 10 seeds) → `results/baseline_filtration_results.csv`; smoke-tested on the torus (18 s). VR's SGT/RS at 5/10/25% come from `baseline`.
- [ ] Notebook (runs): add ρ = d_topo/d_SGT and d_topo/d_RS next to β in Table `filtration` for each filtration.
- [ ] Paper: present Finding 8 as a stability–sensitivity pair. If DTM's ρ is close to 1, say that its lower β partly reflects smoothing.
- Depends on item 1 (KD construction).

---

## 5. Non-degenerate synthetic condition

- **Now (verified):** `tadasets.{torus,sphere,swiss_roll}(n=1000, ambient=20, seed=42)` pads the 3-d sample to 20 columns and multiplies by Q from the QR decomposition of an **unseeded** uniform 20×20 matrix. This is equivalent to X = S Qᵀ with Q ∈ ℝ^{20×3} orthonormal (the first three rows of the rotation). `preprocess` then z-scores each column. Rank 3 means a leave-one-out R² of 1.
- [x] Document this embedding in `setup.tex` (including the z-scoring after rotation).
  - Done (X = S Qᵀ, Q ∈ ℝ^{20×3} orthonormal, seeded rotation, rank 3, z-scoring after the rotation).
- [x] Add a generator `src/data.py::synthetic(manifold, m_noise, sigma, embedding, seed)`, seeded throughout:
  - **Redundancy sweep:** 3 intrinsic coordinates plus linear mixtures in 20 − m columns, and m ∈ {0, 4, 8, 12, 16} columns of i.i.d. noise. Report the realised mean leave-one-out R² for each setting.
  - **Ambient-noise sweep:** X + σZ with σ ∈ {0, 0.05, 0.1, 0.2} × the manifold diameter.
  - Optional: a nonlinear embedding (a random 2-layer tanh network from ℝ³ to ℝ²⁰, seeded).
  - Implemented with separate seeded streams for sample, rotation, noise columns, ambient noise and the tanh weights; `synthetic(m)` with defaults is bit-identical to the study manifold (shares its cache). Sweep in the config (`data.synthetic_sweep`): m ∈ {0, 4, 8, 12, 16}, σ ∈ {0.05, 0.1, 0.2} × diameter of the 3-d sample, tanh embedding; 27 variants (keys like `torus_noise8`, `sphere_sigma0.1`, `swiss_roll_tanh`). Realised mean LOO R²: noise columns 1.00 / 0.80 / 0.60 / 0.41 / 0.22; σ = 0.05/0.1/0.2: 0.60–0.70 / 0.30–0.42 / 0.09–0.15; tanh 1.00 (still linearly predictable, so it tests nonlinearity, not redundancy). Experiments `synthetic_topological` (kNN, VR, MAR, 13 rates) and `synthetic_baseline` (RS, SGT; 10 seeds); `main.py` runs them with their generated datasets (skipped when dataset IDs are given unless named with `-e`). Smoke-tested on two variants at 10%: runs; the 40 RS/SGT diagrams on full-rank noisy clouds took ~12 min under CPU contention, so expect this run to take hours.
- [ ] **[user]** Run kNN/VR/MAR at 13 rates on every variant, plus the SGT/RS baselines (ρ needs SGT). (See "Runs needed".)
- [ ] Notebook (runs): figure of ρ against R² across the sweep, overlaid on the 18 study datasets. Fit within the sweep (it is the controlled test).
- [ ] Paper: Finding 6 gets the controlled test; the 15-point correlation becomes supporting evidence.
- Cost: 3 manifolds × (5 + 4 + 1) variants × 3 mechanisms or MAR only × 13 rates, kNN only, n = 1000, so a small run compared with the main grid.

---

## 6. Theory and the max-displacement error

- **Now:** `ph.tex` states function-based stability (d_W ≤ C_p ‖f − g‖_∞). Finding 3 explains BN/PL vs WS/PI only by max-type vs sum-type aggregation.
- [x] `ph.tex`: replace this with Rips stability for point clouds (Chazal, de Silva & Oudot 2014). With the identity correspondence between X and X̂, |d(x_i, x_j) − d(x̂_i, x̂_j)| ≤ 2e_∞, so d_B(D(X), D(X̂)) ≤ 2 d_GH(X, X̂) ≤ 2e_∞ under the diameter (edge-length) convention used by ripser/giotto. State the convention. Our diagrams are divided by diam(X) (`normalize.py`), so the bound in our units is 2e_∞/diam(X).
  - Done in the paper, with `Chazal2014` added to `main.bib`; compiles. Also states the equivalent argument (both filtrations live on the same full simplex, values differ by ≤ 2e_∞). The bound needs a common cutoff, which is now the case (item 1 decision).
- [x] Fix the Wasserstein statement: the stability constants depend on the space (number of points, dimension), not only on p. Diagonal matching costs (d − b)/2 under the L∞ ground metric.
  - Verified the code: `persim.wasserstein` is **W₁ with the Euclidean ground metric** (q = 2, diagonal cost (d − b)/√2), whereas `ph.tex` defined W_p with ‖·‖_∞. `ph.tex` now defines W_p with a ground metric q, states both diagonal costs, and says \WS = W₁ with q = 2. Stability paragraph cites Skraba & Turner and Cohen-Steiner et al. 2010 for the space-dependent constants; the landscape sentence now uses ‖λ − λ'‖_∞ ≤ d_B (Bubenik).
- [x] Compute e_∞ = max_i ‖x_i − x̂_i‖₂ for every imputed cloud. Add it as a comparison metric (`MAXDISP` in `metrics.py` and `comparison_metrics`) on kNN, 3 mechanisms and 13 rates.
  - `metrics.compute_max_displacement`, key `MAXDISP`, in the `comparison_metrics` experiment of the config. Verified on a hand example (5.0) and a smoke run (torus kNN 10%: MCAR 0.53, MAR 2.47). First look: 2e_∞/diam is far from tight (0.07–1.0 vs BN ≤ 0.045 on torus/sphere/concrete/treasury).
- [ ] **[user]** Run `comparison_metrics` with `MAXDISP`; it is cheap, since the imputations are cached. Combine this with the 0.1 re-run.
- [ ] Notebook (runs): tightness τ = d_B / (2e_∞ / diam X), shown as a distribution over datasets and rates. Account for `max_edge_length` truncation and clipped essential bars, which can break the bound in edge cases, and report any violations.
- [ ] (runs) Add e_∞ (and β_e∞) to Table `growth` and Figure `growth_by_family` as the theory-relevant reference next to RMSE (over all entries, see 0.1).
- [ ] (runs) Rewrite Finding 3 as a claim in the form "BN/PL track e_∞ with elasticity …, whereas WS/PI track RMSE", using within-dataset Δβ against both references.

---

## Review round 2 (items 7–18)

Second list of review points (2026-11-02). Only 9 and 10 change what the runs compute; the rest is notebook (after the runs) or paper-only work.

- [x] **9 Imputation-free baseline.** `NO_IMPUTATION` keeps the masked matrix; `tda.vietoris_rips_complex` then runs VR on `nan_euclidean_distances` (not a metric, which VR tolerates; pairs without shared coordinates → ∞). New experiment `topological_metrics_no_imputation` (MAR, 13 rates, VR) → `results/topological_metrics_no_imputation_{results,distances}.csv`; also part of `topological_metrics_manifolds`. Unit test: VR on a masked matrix equals VR on its nan-Euclidean distance matrix. Smoke test (torus, MAR): runs; it is closer to the original than kNN at 25% (WS H1 0.56 vs 1.08).
- [x] **10 Betti recovery.** Metric `BETTI` (plus `BETTI_CLEAN` for the clean diagram) counts, per dimension, features alive at the cutoff or with persistence > 2c, c = 95% quantile of a bottleneck bootstrap on the clean cloud (30 half-size subsamples, seed 42; `tda.betti_thresholds`, cached; config `betti`). User decision (2026-11-02): this rule, reported against both the ground truth and the clean diagram. Reason: the reviewer's bootstrap threshold alone misses the torus and sphere H2 voids (their bars are clipped at the cutoff); with "alive at the cutoff" the clean torus (1,2,1) and sphere (1,0,1) are exact, the clean Swiss roll gives (1,3,0) instead of (1,0,0) because VR bridges the gaps between the layers at the cutoff scale. New experiment `topological_metrics_manifolds` (3 manifolds, all 9 imputers + no imputation, MCAR/MAR/MNAR, 5/10/25% (agent default, user did not specify), 10 seeds) → `results/topological_metrics_manifolds_{results,distances}.csv`; the torus figure (clean, MNAR 25% kNN, MNAR 25% mean) can be drawn in the notebook from these settings. Unit test: a clean circle gives (1,1,0). Smoke test (torus, MCAR/MNAR 25%, 2 seeds): clean (1,2,1); MNAR kNN mostly (1,2,1), mean loses the loops.
  - Found while testing: **fork deadlock**. With the bootstrap in the main process, every forked worker hung at 0% CPU inside scikit-learn's OpenMP code (`radius_neighbors`, called by giotto's VR). Fixed: all pools use the spawn start method (`utils.process_pool`); the cache directory can be set with `TDA_CACHE_DIR` (spawned workers do not inherit module patches). The old code had the same hazard after the PyTorch imputers ran in the main process; it may explain why the last run "did not feel right". Smoke test now 1:52 min; cache tests pass with spawned workers.
- [~] **7 Statistics hygiene.** Done on the current results (the notebook regenerates everything after the runs): Holm family and size plus one star legend in every caption (baseline: 12 per column; imputation: tests per block, computed; growth: 12 per Δβ column; filtration: 24; mechanism: uncorrected Friedman per row); Figure 3 normalised by the median of the three lowest rates; appendix tables `growth_fit` (R² per fit, R² < 0.8 count, d(40%)/d(5%); currently 127 of 252 fits have R² < 0.8, mostly BN/PL, i.e. saturating curves), `mrstar` (MR* per dataset × mechanism; RS 70/90%, jitter, fresh columns appear with `baseline_reference`), `baseline_iqr`, `imputation_iqr`, `growth_iqr`; consensus-rank definition and Holm families in `setup.tex`; Mann–Whitney rephrased; "4.3% vs 12.9%" replaced by the MR* table. Open (runs): PL-H0 jump check, IQR across seeds (per-seed files exist from now on), discussion of the flagged fits. Original note: Holm families and test counts in captions, R² per β fit (flag < 0.8) and d(40%)/d(5%), Figure 3 normalised by the median of the three lowest rates, IQR tables, consensus-rank definition, Mann–Whitney rephrasing, per-dataset MR* table, PL-H0 jump (expected to vanish with the 0.2 grid fix; giotto removes the infinite H0 bar, so it is not essential-bar clipping).
- [~] **8 RQ4.** New notebook section and appendix table `rq4` on the current results: OLS log ρ ~ R² + log RMSE (only R² significant), partial Spearman ρ–R² | RMSE −0.65 (p = .012), kNN-regression R² (r_s 0.19, n.s.), TwoNN intrinsic dimension (r_s 0.59, p = .021), participation ratio, type-stratified correlations (−0.50 / −0.57, n.s. with n = 7/8). Added to RQ4 with a TODO to recheck after the runs; the controlled test (item 5) needs the synthetic sweep. Original note: ρ ~ R² + log RMSE, partial Spearman, kNN-regression R², intrinsic dimension (TwoNN / participation ratio), type-stratified correlation.
- [~] **11 Figures.** Figure 3 moved into RQ3; Table 3 replaced in the text by `baseline_lines.pdf` (table in the appendix). Open (runs): line plot for Table 7 (needs the new KD keys and the DTM/KD references). Original note: line plots instead of Tables 3 and 7 (tables to the appendix); move Figure 3 to RQ3.
- [x] **12–18** paper-only items done (see "Paper changes"); open: compute statement (user), result-dependent rewrites of Findings 2/8 and the contributions' headline findings (TODOs).

## Paper changes

Running list of every change the paper (`paper/robustness_tda`) needs, with the PLAN item that causes it. `[x]` = edited in the paper (possibly as a `% TODO(PLAN x.y)` placeholder), `[ ]` = still to do, `(runs)` = needs the user's runs first.

- [x] **0.1** `setup.tex`: "computed between $X$ and $\widehat{X}$ on the masked entries" is wrong. RMSE/MAE are computed over all $n \times d$ entries (observed ones contribute zero), i.e. $\lVert X - \widehat{X} \rVert_F / \sqrt{nd}$, the total perturbation, matching the whole-diagram topological distances. Mention RMSE$_\text{all}$ = $\sqrt{\MR}\,$RMSE$_\text{masked}$.
- [ ] **0.2** (runs) all PL/PI numbers in RQ1–RQ6, Tables `baseline`, `imputation`, `growth`, `mechanism`, `filtration` and Figures `growth_by_family`, `pairwise_imputers`, `redundancy_retention` change once PL/PI are recomputed on a shared grid. Re-check every PI claim (e.g. RQ1 "PI ratios 0.23/0.36", "PI exceeds the noise floor only at 10–30%").
- [ ] **2.1** `tables/kd.tex`: manifold rows change with the seeded rotation (superseded by the item 1 rewrite of `kd.tex`; copy from `figures/tmlr/` once item 1 is done).
- [x] **5** `setup.tex`: manifold embedding documented (pad + seeded rotation, X = S Qᵀ, rank 3, z-scoring after rotation).
- [ ] **5** (runs) RQ4 / Finding 6: controlled test (ρ vs R² across the synthetic sweep, fitted within the sweep), the 15-dataset correlation as supporting evidence; new figure; `limitations.tex` ("This relationship is correlational …").
- [x] **2.2** appendix: new subsection "Experimental Grid" with `tables/grid.tex` (copy of `figures/tmlr/grid.tex`), referenced from `setup.tex`. Re-copy whenever the config changes.
- [x] **2.3** observed-entry standardisation and Constant as oracle: `setup.tex` (scaling sentence, Constant sentence), `tables/imputers.tex` (Constant row), appendix (datasets paragraph); PI width (two pixels of the persistence axis) in the appendix.
- [ ] **2.3** (runs) RQ2 text and Table `imputation`/pairwise figure: Constant is now an oracle reference (not a "simple imputer" without information); kNN changes slightly (observed-entry scaling).
- [x] **2.3** appendix "Implementation Details" (new), `tables/imputers.tex` (RF, MICE), `tables/datasets.tex` (analysed feature counts), reproducibility paragraph in `setup.tex` (code link; TODOs for DOI and compute statement). Paper compiles.
- [x] **2** review gaps closed (2026-11-02): appendix lists library versions (Python 3.12, GUDHI 3.13.0, giotto-tda 0.6.2 / giotto-ph 0.2.4, persim 0.3.8, scikit-learn 1.3.2, statsmodels 0.14.6, mdatagen 0.2.0, PyTorch 2.13.0, NumPy 1.26.4, SciPy 1.17.1, pandas 3.0.5, tadasets 0.2.2), the exact mdatagen calls, and a "Seeds" paragraph (seeds 1–10 and what each controls; manifold and cutoff seed 42; per-seed distances archived). Paper compiles.
- [ ] **2.1/2.3** (user) Zenodo DOI and compute statement (hardware, core-hours) into the reproducibility paragraph.
- [x] **2.1/2.3** reproducibility paragraph: mention `config/experiments.yaml` (full grid + hyperparameters), the pinned environment (`uv.lock`, `requirements.txt`), and that every random component, incl. the manifold rotation, is seeded.
- [x] **1** `setup.tex`: DTM/KD paragraph rewritten (weighted Rips of Anai et al., f = DTM or h·d_K, Gaussian kernel, median-heuristic bandwidth with the sweep in the appendix, cutoff rule, cutoff and bandwidth shared from the original). Leaves `% TODO(PLAN 1)` for the appendix label.
- [ ] **1** (runs) `results.tex` RQ6 (Finding 8 and the KD paragraph: the Silverman/√2 degeneracy argument no longer applies), `limitations.tex` (bandwidth sentence, "between Vietoris–Rips and the kernel distance"), abstract ("DTM significantly outperforms VR"), appendix `kd.tex` → bandwidth-sweep table. DTM numbers change too (all vertices now enter; shared cutoff).
- [ ] **1/4** (runs) Finding 8 as a stability–sensitivity pair (ρ next to β in Table `filtration`).
- [ ] **2.3** (runs) everything that uses MNAR: Table `mechanism`, RQ5 text ("MNAR/MCAR ratio 0.78–1.26", "SGT lower under MNAR for 14–15 of 15"), RQ1 mechanism sentence, Table `baseline` if it uses MNAR. MNAR is now averaged over 10 masks.
- [ ] **shared cutoff** (runs) all imputed/baseline diagrams change; manifold BN H2 values drop most (they mostly measured the cutoff difference). Re-check every BN H2 statement, e.g. RQ3 "BN H2 Δβ" and Table `growth`.
- [ ] **3** (runs) RQ1 / Finding 1 against the same-cardinality references (jitter, fresh sample) and the 70/90% subsample sensitivity; `limitations.tex` "noise floor estimated from an 80% subsample".
- [x] **6** `ph.tex`: Rips stability d_B ≤ 2e_∞ (diameter convention; 2e_∞/diam X in our units), space-dependent Wasserstein constants, W_p defined with a ground metric q, diagonal costs (d − b)/2 (q = ∞) and (d − b)/√2 (q = 2), \WS = W₁ with q = 2, landscape bound ‖λ − λ'‖_∞ ≤ d_B. `Chazal2014` added to `main.bib`. Paper compiles.
- [ ] **6** (runs) Finding 3 in terms of e_∞ vs RMSE; e_∞ in Table `growth` and Figure `growth_by_family`; tightness τ = d_B/(2e_∞/diam X) (appendix or text).
- [x] **0.3** `setup.tex`: the seeds sentence now says which settings are repeated over 10 seeds and that the MAR mask is shared (MNAR is stochastic since the 2.3 fix) (with a `% TODO(PLAN 0.3)` until the re-runs are in). Check `limitations.tex` ("10 seeds for each non-deterministic method") and the compute statement once the runs are done.
- [x] **0.1** `results.tex` RQ3: "MAE naturally has a steeper growth rate than RMSE ($\beta = 1.18$ vs.\ 0.65)": explain it, since over all entries MAE scales with MR and RMSE with $\sqrt{\MR}$ for a fixed per-cell error.

### Paper changes from review round 2 (2026-11-02)
- [x] **12** `related.tex`: gap paragraph citing Jäger et al. 2021, HyperImpute (Jarrett et al. 2022), Turkeš, Montúfar & Otter 2022 (new bib entries; Turkeš DOI checked via Crossref).
- [x] **13** "retention ratio" → "relative topological error" everywhere (text, captions, figure label); SGT described as random hot-deck imputation from the hidden values of the same feature.
- [x] **14** `sections/impact.tex`: Broader Impact Statement (human-related datasets, risk of false structure, compute; TODO to align with the compute statement).
- [x] **15** Section 4 (Methodology) merged into Setup as "Problem formulation" (`methodology.tex` removed); contributions updated (references, theory, Betti recovery; headline findings as TODO); SGT-by-mechanism sentence moved from RQ1 to RQ5; Findings 2 and 8 marked with TODOs (Finding 2 in the reviewer's form is not supported by the current results for MICE).
- [x] **16** Title → "The Robustness of Persistent Homology Under Imputation-Induced Perturbations".
- [x] **17** Abbreviation macros now upright uppercase (`\textnormal{WS}` etc., matching the figure labels); every caption has the same star legend and its Holm family.
- [x] **18** Bibliography: Chazal et al. 2018 authors, Rubin capitalisation, Edelsbrunner & Harer (AMS 2010, DOI 10.1090/mbk/069 instead of a wrong Springer chapter), `@aricle` typo that dropped Santos 2019 from the bibliography, arXiv IDs printed for all 9 arXiv entries (tmlr.bst ignores `eprint`), Zheng & Charoenphakdee → 2022, Damrich et al. as NeurIPS 2024 proceedings, two DOIs stored as URLs. All 45 DOIs resolve (doi.org).
- [x] **9/10** `setup.tex`: imputation-free reference and Betti-recovery method (incl. the Swiss-roll caveat).
- [x] Bug fixed: an earlier `% TODO(PLAN 0.3)` comment had commented out the whole reference paragraph of `setup.tex`; it is restored and extended.
- [ ] (runs) RQ1 with the new references, Table 4 no-imputation row (code ready), Figure 3 line for no imputation (open design question: too many lines in one panel), Betti results and the torus figure, Findings 2/3/8, abstract and conclusion.

## Waiting for user

Items here are blocked on the user. Agents add to this section (see `CLAUDE.md`); the user clears entries.

### Questions
- (none open; PI σ and standardisation were decided on 2026-11-01, see 2.3)
- ~~**(0.2 / 2.3) PI bandwidth.** giotto's `sigma=0.1` is in filtration units, but the normalised diagrams span ~0.003 (H0), ~0.025 (H1) and ~0.08 (H2). σ is therefore 4–400× the grid extent, and the image is one broad blob (torus: min/max pixel 0.14–0.77; correlation of original and imputed image 0.92–1.00). PI then mostly measures the difference in feature counts, which explains its cardinality sensitivity in RQ1. Options: keep σ = 0.1 and state it; or set σ relative to the grid (e.g. σ = 0.01 → H1 correlation 0.99, H2 0.55). Decide before the PL/PI re-run.~~ Decided: σ relative to the grid.

### Runs needed
Exact `main.py` calls in the order they should run, with the result files they write. Set `runtime.workers` in `config/experiments.yaml` first. Because of 0.2 (shared PL/PI grid), 0.3 (10 seeds), the shared original cutoff (item 1) and the seeded MNAR (2.3), **every result file must be regenerated**; the original diagrams and the MCAR/MAR imputations stay cached. Since the observed-entry standardisation (2.3), all imputations are recomputed too.
1. `python main.py -e comparison_metrics` → `results/comparison_metrics_results.csv` (RMSE, MAE, MAXDISP; kNN; 3 mechanisms × 13 rates; 10 seeds for MCAR/MNAR). Cheap; also caches the kNN imputations that `baseline_reference` needs.
2. `python main.py -e topological_metrics_mr` → `results/topological_metrics_mr_results.csv`.
3. `python main.py -e baseline` → `results/baseline_results.csv`.
4. `python main.py -e baseline_reference` → `results/baseline_reference_results.csv` (RS 70/90%, matched jitter, fresh sample; MAR, 13 rates). Needs step 1.
5. `python main.py -e topological_metrics_filtration` → `results/topological_metrics_filtration_results.csv` (VR, DTM, KDM + 4 bandwidth variants).
6. `python main.py -e baseline_filtration` → `results/baseline_filtration_results.csv` (RS/SGT for DTM and KDM; MAR 5/10/25%).
7. `python main.py -e topological_metrics` → `results/topological_metrics_results.csv` (all 9 imputers; RF/MICE/GAIN/TabCSDI/OT now 10 seeds; the most expensive run).
8. `python main.py -e topological_metrics_no_imputation` → `results/topological_metrics_no_imputation_*.csv` (imputation-free baseline, MAR, 13 rates; cheap).
9. `python main.py -e topological_metrics_manifolds` → `results/topological_metrics_manifolds_*.csv` (Betti recovery; 3 manifolds × 10 methods × 3 mechanisms × 5/10/25%; the deep imputers take a few hours).
10. `python main.py -e synthetic_topological synthetic_baseline` → `results/synthetic_topological_results.csv`, `results/synthetic_baseline_results.csv` (27 synthetic variants, MAR, 13 rates; kNN and RS/SGT). Several hours.
(`python main.py` without `-e` runs all of them in config order.)

## Order and dependencies

1. **0.1, 0.2, 0.3**: they change the existing numbers, so do them before anything new is compared against them.
2. **2.1** (config, pinning, seeding including the rotation): new runs should already use it.
3. **1**, then **4** (KD construction, then the DTM/KD baselines).
4. **6** (e_∞ is cheap; it reuses 0.1's mask plumbing).
5. **3** (needs masked RMSE from 0.1).
6. **5** (needs the seeded generator from 2.1).
7. **2.2, 2.3**, the Zenodo archive and the paper paragraphs last, once the grid is final.

After each step's **[user]** runs have finished: re-run `notebooks/tmlr_figures.ipynb`, copy `figures/tmlr/` into the paper, and update the text numbers it prints.
