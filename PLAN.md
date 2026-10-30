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
- [ ] Find out why the stochastic imputers ran with one seed (was `SEEDS` reduced for the run?).
- [ ] Decide (user): **[user]** re-run with 10 seeds (`range(10)`), or state the actual number in the paper. Item 2 depends on this.

---

## 1. Kernel-distance filtration

- **Now (confirmed):** `src/tda.py::kernel_distance` builds **VR on the RKHS metric** D_K(x,y) = √(2 − 2k(x,y)), with a Gaussian kernel and Silverman bandwidth h = (n(d+2)/4)^(−1/(d+4)). This is a change of metric, not the kernel distance-to-measure of Phillips et al. / Chazal et al. `max_edge_length_for` uses the same metric. The appendix table `kd` (85–99.7% of pairs at the √2 ceiling) documents the resulting degeneracy.
- **Target construction:** a weighted Rips filtration with Euclidean distances and weights f(x_i) = √(κ(P,P) + k(x_i,x_i) − 2κ(P,x_i)), where κ(P,x) = (1/n) Σ_j k(x_j,x). This is the Anai et al. construction used for DTM, with the DTM replaced by the kernel distance, so DTM and KD differ only in the density estimator.

Tasks:
- [ ] Add `kernel_distance_to_measure(data, h)` to `src/tda.py`, returning the weights f(x_i). Note k(x,x) = 1 for the Gaussian kernel.
- [ ] Build the filtration. Either:
  - use `gudhi.weighted_rips_complex.WeightedRipsComplex(distance_matrix, weights)` and compute persistence with GUDHI, converting to the giotto diagram format used by `transform_pd`; or
  - use giotto `WeightedRipsPersistence` with precomputed weights, if it accepts them.

  Check that the weight convention (radius vs. diameter, p = 2) matches `distance_to_a_measure` exactly.
- [ ] Bandwidth: median heuristic h = median pairwise Euclidean distance (outcome-independent) as the main setting. Add a sweep h ∈ {0.25, 0.5, 1, 2} × median, plus Silverman as a documented column.
- [ ] `max_edge_length_for(KD)`: use Euclidean distances like DTM (the weights shift filtration values; reuse the DTM handling of `effective_mel`).
- [ ] New method keys in `constants.py` (e.g. `KDM` plus a bandwidth-variant suffix); keep the old `KD` under a renamed key (`KD_RKHS`) only if the paper keeps it as a reference column.
- [ ] **[user]** Run `topological_metrics_filtration` (kNN, MAR, 5/10/25%) for the new KD at all bandwidths (the agent adds the call to `main.py`).
- [ ] Notebook: Table `filtration` with the new KD; the appendix bandwidth sweep replaces or extends `kd.tex`.
- [ ] Paper: rewrite the KD paragraph in `setup.tex` (construction plus bandwidth) and RQ6 in `results.tex`; update the `limitations.tex` sentence on bandwidth.
- **Done when:** DTM and KD share one code path apart from the weight function (unit test: on a toy cloud, both run through the same filtration builder), and Table `filtration` reports median-heuristic KD with the sweep in the appendix.

---

## 2. Reproducibility specification and code release

### 2.1 Pinning and configuration
- **Now:** `uv.lock` pins everything, but `requirements.txt` is unpinned. Installed versions: gudhi 3.13.0, giotto-tda 0.6.2, scikit-learn 1.3.2, mdatagen 0.2.0, torch 2.13.0, persim 0.3.8, tadasets 0.2.2, numpy 1.26.4, scipy 1.17.1, pandas 3.0.5, statsmodels 0.14.6. Experimental choices are split across `src/constants.py`, `main.py` and hard-coded defaults (e.g. DTM `m = 0.05` inside `tda.py`, GAIN/TabCSDI/OT arguments in `imputation.py`).
- [ ] Generate a pinned `requirements.txt` from the lock (`uv export --format requirements-txt --no-hashes`).
- [ ] Add `config/experiments.yaml` holding the full grid and every hyperparameter. `main.py` reads it; `constants.py` keeps only keys and labels. Remove hard-coded hyperparameters from `src/`.
- [ ] Seed everything from the config. This includes manifold sampling and the currently **unseeded rotation** in `tadasets.embed` (generate the rotation with `np.random.default_rng(seed)` in `src/data.py`, which also fixes the `kd.tex` drift).
- [ ] Zenodo: archive original and imputed diagrams (from `.cache/`, exported to a documented HDF5/NPZ layout), all `results/*.csv`, and the config. Add the DOI to `README.md` and the paper.

### 2.2 Table "Experimental grid" (paper, appendix or setup)
- [ ] Rows: datasets, mechanisms, rates, imputers, filtrations, measures, dimensions, references, seeds. Columns: levels, and which RQ/table uses them. Generate it from `config/experiments.yaml` in the notebook, so it can't drift from the code.

### 2.3 State explicitly in the paper (current values from the code; verify the marked ones)

| Item | Current implementation | Action |
|------|------------------------|--------|
| PH on 3.5k–5k points | No point subsampling. VR via giotto/ripser with `collapse_edges=True` and `max_edge_length` = 2.5 × the longest MST edge of a 500-point subsample (seed 42) of the original (`_max_edge_length`) | State it. Check how giotto treats essential bars (expected: deaths set to `max_edge_length`); state whether they are clipped before WS/BN |
| Diagram scaling | Births/deaths divided by the diameter of the original cloud (`normalize.py`) | State it (it matters for item 6) |
| Wasserstein | `persim.wasserstein`: optimal matching, sum of L∞ costs (p = 1, q = ∞) | Verify, then state p and the ground metric |
| PI | giotto `PersistenceImage` defaults (`sigma=0.1`, `n_bins=100`, `weight_function=None`), fitted per diagram (see 0.2) | Fix 0.2, then state σ, resolution and weight |
| PL | giotto `PersistenceLandscape` defaults (`n_layers=1`, `n_bins=100`), fitted per diagram (see 0.2) | Fix 0.2, then state layers and resolution |
| DTM | k = ⌈0.05 n⌉ neighbours, p = 2, giotto `WeightedRipsPersistence(weights='DTM')` | State it |
| MCAR | Random half of the columns (`random.seed(seed)`); mdatagen `mMCAR` with the rate scaled by d/(d/2), so the overall rate equals MR | State it |
| MAR | mdatagen `mMAR(n_xmiss=d//2).correlated(MR)`, no seed (deterministic) | Verify in mdatagen 0.2.0 which columns are masked, which drive the mechanism, and whether MR is overall or per column |
| MNAR | mdatagen `mMNAR(n_xmiss=d//2).random(MR, deterministic=True)` | Same verification |
| Paired design | Masks are computed once per (seed, dataset, mechanism, rate) and shared by all imputers and baselines (`introduce_missingness`) | State it |
| Standardisation | `StandardScaler` on the complete data **before** masking (`data.py::preprocess`): a leak | Preferred: standardise on observed entries after masking and evaluate in the original scale. Otherwise state the leak and argue it is negligible (z-scoring is a per-column affine map) |
| kNN | `KNNImputer(n_neighbors=min(⌊√n⌋, n/2))` | State it |
| RF | `IterativeImputer(estimator=RandomForestRegressor(n_estimators=30, max_depth=15))`, default `max_iter=10`; not the missForest port | State it; adjust "missForest" in `imputers.tex` |
| MICE | statsmodels `MICEData`, 5 iterations, a **single draw** (`mice_imp.data`) | State "single imputation from the chained equations" |
| GAIN | batch 128, hint 0.9, α = 100, iterations clip(300·⌈n/128⌉, 2000, 10000) | State it, plus the learning rate (read from the code) |
| TabCSDI | 20 diffusion steps, Adam lr 1e-3, batch 128, epochs max(20, ⌈2000/steps per epoch⌉) (the `TABCSDI_EPOCHS` default is overridden) | State it; clean up the dead default |
| OT-Impute | 500 iterations, batch 128, RMSprop lr 1e-2, 20 Sinkhorn iterations | State it, plus ε |
| Seeds | See 0.3 | — |
| OpenML filter | Active, no missing values, no symbolic features, 8–50 numeric features, 500–5000 instances, newest version per name, then manual removal of non-numeric/categorical IDs (lists in `git show aa0575b:notebooks/find-datasets.ipynb`), then dropping labels/date columns (`data.py::preprocess`) | Restore the filter as a script (`scripts/select_datasets.py`) with the ID lists; state it |
| Air Quality, 5000 rows | OpenML 46762 has exactly 5000 × 9 natively; it sits at the upper filter bound | State it |

- [ ] Paper: a reproducibility paragraph with the code link and Zenodo DOI, and a compute statement (hardware, core-hours). Collect the core-hours from the run logs, or estimate them from `src/logger.py` timings.

---

## 3. Noise-floor confound

- **Now:** the noise floor is an 80% random subsample (`robustness.py::random_subsample`, `SUBSAMPLE_FRACTION = 0.8`). It changes the cardinality, which mainly moves H0 and the PI distances (see RQ1).
- [ ] **RMSE-matched jitter** (`robustness.py::matched_jitter`): perturb only the masked entries with N(0, σ²), where σ is the masked-entry kNN RMSE at the same (dataset, mechanism, rate, seed). Requires 0.1. The ratio d_topo(imputer)/d_topo(jitter) shows whether imputation error is topologically more or less harmful than unstructured error of the same size.
  - Scope: kNN, VR, MAR, 13 rates, no new imputations (reuse the cached kNN output for σ).
  - Add it as a baseline method in `robustness_pipeline.py` (needs access to the kNN imputation of the same mask).
- [ ] **Fresh sample** (manifolds only): resample n points with a different seed (same rotation) as the true sampling-noise floor.
- [ ] **Subsample at 70/80/90%:** parametrise `random_subsample` by fraction. Report MR* and the ordering in Table `baseline` for each fraction.
- [ ] **[user]** Run the extended `experiment_baselines` (jitter, fresh sample, 70/80/90% subsample) once the agent has added the calls to `main.py`.
- [ ] Notebook: add jitter and fresh-sample columns to Table `baseline`, or a new table. Check that MR* and the Table `baseline` ordering hold; if not, rewrite the RQ1 claims against the matched reference.
- **Done when:** the RQ1 statements are tied to a same-cardinality reference and the subsample sensitivity is reported.

---

## 4. Sensitivity for DTM and KD (RQ6)

- **Now:** baselines (SGT, RS, CCA) exist only for VR, so Table `filtration` reports only β (stability), with no sensitivity.
- [ ] **[user]** Run `experiment_baselines('baseline_filtration', [MAR], [5, 10, 25], [DTMC, KD_new], …)`. The function already takes `tda_methods`; the same masks give a paired design with Table `filtration`.
- [ ] Notebook: add ρ = d_topo/d_SGT and d_topo/d_RS next to β in Table `filtration` for each filtration.
- [ ] Paper: present Finding 8 as a stability–sensitivity pair. If DTM's ρ is close to 1, say that its lower β partly reflects smoothing.
- Depends on item 1 (KD construction).

---

## 5. Non-degenerate synthetic condition

- **Now (verified):** `tadasets.{torus,sphere,swiss_roll}(n=1000, ambient=20, seed=42)` pads the 3-d sample to 20 columns and multiplies by Q from the QR decomposition of an **unseeded** uniform 20×20 matrix. This is equivalent to X = S Qᵀ with Q ∈ ℝ^{20×3} orthonormal (the first three rows of the rotation). `preprocess` then z-scores each column. Rank 3 means a leave-one-out R² of 1.
- [ ] Document this embedding in `setup.tex` (including the z-scoring after rotation).
- [ ] Add a generator `src/data.py::synthetic(manifold, m_noise, sigma, embedding, seed)`, seeded throughout:
  - **Redundancy sweep:** 3 intrinsic coordinates plus linear mixtures in 20 − m columns, and m ∈ {0, 4, 8, 12, 16} columns of i.i.d. noise. Report the realised mean leave-one-out R² for each setting.
  - **Ambient-noise sweep:** X + σZ with σ ∈ {0, 0.05, 0.1, 0.2} × the manifold diameter.
  - Optional: a nonlinear embedding (a random 2-layer tanh network from ℝ³ to ℝ²⁰, seeded).
- [ ] **[user]** Run kNN/VR/MAR at 13 rates on every variant, plus the SGT/RS baselines (ρ needs SGT).
- [ ] Notebook: figure of ρ against R² across the sweep, overlaid on the 18 study datasets. Fit within the sweep (it is the controlled test).
- [ ] Paper: Finding 6 gets the controlled test; the 15-point correlation becomes supporting evidence.
- Cost: 3 manifolds × (5 + 4 + 1) variants × 3 mechanisms or MAR only × 13 rates, kNN only, n = 1000, so a small run compared with the main grid.

---

## 6. Theory and the max-displacement error

- **Now:** `ph.tex` states function-based stability (d_W ≤ C_p ‖f − g‖_∞). Finding 3 explains BN/PL vs WS/PI only by max-type vs sum-type aggregation.
- [ ] `ph.tex`: replace this with Rips stability for point clouds (Chazal, de Silva & Oudot 2014). With the identity correspondence between X and X̂, |d(x_i, x_j) − d(x̂_i, x̂_j)| ≤ 2e_∞, so d_B(D(X), D(X̂)) ≤ 2 d_GH(X, X̂) ≤ 2e_∞ under the diameter (edge-length) convention used by ripser/giotto. State the convention. Our diagrams are divided by diam(X) (`normalize.py`), so the bound in our units is 2e_∞/diam(X).
- [ ] Fix the Wasserstein statement: the stability constants depend on the space (number of points, dimension), not only on p. Diagonal matching costs (d − b)/2 under the L∞ ground metric.
- [ ] Compute e_∞ = max_i ‖x_i − x̂_i‖₂ for every imputed cloud. Add it as a comparison metric (`MAXDISP` in `metrics.py` and `comparison_metrics`) on kNN, 3 mechanisms and 13 rates.
- [ ] **[user]** Run `comparison_metrics` with `MAXDISP`; it is cheap, since the imputations are cached. Combine this with the 0.1 re-run.
- [ ] Notebook: tightness τ = d_B / (2e_∞ / diam X), shown as a distribution over datasets and rates. Account for `max_edge_length` truncation and clipped essential bars, which can break the bound in edge cases, and report any violations.
- [ ] Add e_∞ (and β_e∞) to Table `growth` and Figure `growth_by_family` as the theory-relevant reference next to RMSE (over all entries, see 0.1).
- [ ] Rewrite Finding 3 as a claim in the form "BN/PL track e_∞ with elasticity …, whereas WS/PI track RMSE", using within-dataset Δβ against both references.

---

## Paper changes

Running list of every change the paper (`paper/robustness_tda`) needs, with the PLAN item that causes it. `[x]` = edited in the paper (possibly as a `% TODO(PLAN x.y)` placeholder), `[ ]` = still to do, `(runs)` = needs the user's runs first.

- [x] **0.1** `setup.tex`: "computed between $X$ and $\widehat{X}$ on the masked entries" is wrong. RMSE/MAE are computed over all $n \times d$ entries (observed ones contribute zero), i.e. $\lVert X - \widehat{X} \rVert_F / \sqrt{nd}$, the total perturbation, matching the whole-diagram topological distances. Mention RMSE$_\text{all}$ = $\sqrt{\MR}\,$RMSE$_\text{masked}$.
- [ ] **0.2** (runs) all PL/PI numbers in RQ1–RQ6, Tables `baseline`, `imputation`, `growth`, `mechanism`, `filtration` and Figures `growth_by_family`, `pairwise_imputers`, `redundancy_retention` change once PL/PI are recomputed on a shared grid. Re-check every PI claim (e.g. RQ1 "PI ratios 0.23/0.36", "PI exceeds the noise floor only at 10–30%").
- [x] **0.1** `results.tex` RQ3: "MAE naturally has a steeper growth rate than RMSE ($\beta = 1.18$ vs.\ 0.65)": explain it, since over all entries MAE scales with MR and RMSE with $\sqrt{\MR}$ for a fixed per-cell error.

## Waiting for user

Items here are blocked on the user. Agents add to this section (see `CLAUDE.md`); the user clears entries.

### Questions
- **(0.2 / 2.3) PI bandwidth.** giotto's `sigma=0.1` is in filtration units, but the normalised diagrams span ~0.003 (H0), ~0.025 (H1) and ~0.08 (H2). σ is therefore 4–400× the grid extent, and the image is one broad blob (torus: min/max pixel 0.14–0.77; correlation of original and imputed image 0.92–1.00). PI then mostly measures the difference in feature counts, which explains its cardinality sensitivity in RQ1. Options: keep σ = 0.1 and state it; or set σ relative to the grid (e.g. σ = 0.01 → H1 correlation 0.99, H2 0.55). Decide before the PL/PI re-run.

### Runs needed
Exact `main.py` calls in the order they should run, with the result files they write.
- (none yet)

## Order and dependencies

1. **0.1, 0.2, 0.3**: they change the existing numbers, so do them before anything new is compared against them.
2. **2.1** (config, pinning, seeding including the rotation): new runs should already use it.
3. **1**, then **4** (KD construction, then the DTM/KD baselines).
4. **6** (e_∞ is cheap; it reuses 0.1's mask plumbing).
5. **3** (needs masked RMSE from 0.1).
6. **5** (needs the seeded generator from 2.1).
7. **2.2, 2.3**, the Zenodo archive and the paper paragraphs last, once the grid is final.

After each step's **[user]** runs have finished: re-run `notebooks/tmlr_figures.ipynb`, copy `figures/tmlr/` into the paper, and update the text numbers it prints.
