@AGENTS.md

# Working mode: autonomous revision

The current goal is to complete `PLAN.md`. Work through it autonomously, task after task, until every task is done, blocked on the user, or marked **[user]**. Do not stop after one task to ask whether to continue.

## Loop

1. Read `PLAN.md` and pick the next open `[ ]` task, following "Order and dependencies". Skip tasks whose dependencies are not done, tasks marked **[user]**, and tasks listed under "Waiting for user".
2. Mark it `[~]` and implement it. Keep the existing code style (see `AGENTS.md`) and change only what the task needs.
3. Verify it without running experiments: import checks, unit tests or quick checks on a single small point cloud, and a minimal smoke test of any new experiment call (one small dataset, one rate, stop once it runs). If existing results are affected, run the notebook (about 2 minutes) and check that its output is still consistent.
4. Mark the task `[x]` and add a one-line note under it: what changed, how it was verified, and which result files are now stale.
5. If the task produced experiments to run, add them to "Waiting for user → Runs needed" with the exact `main.py` call and the result files they write.
6. Commit the task on the branch `revision` (create it from `main` if missing) with a message naming the PLAN item, e.g. `PLAN 1: kernel distance-to-measure filtration`. Never push.
7. Continue with the next task.

When nothing is left except **[user]** tasks and items under "Waiting for user", stop. Give a short summary: what was done, which runs to start (in order), and which questions are open.

## When to ask the user (critical issues)

Ask immediately, and do not guess, when:
- a fix would **change the direction of a paper finding** or a claimed significance, or the evidence shows a planned approach is not valid;
- a **design decision** has no clear default in `PLAN.md` (e.g. seed count in 0.3, whether to keep a legacy method as a reference column);
- the code and the paper disagree and it is unclear which one is right;
- a change would **delete or overwrite results** in `results/` or cached data in `.cache/`, or invalidate most of the cache;
- you would have to run an experiment to make progress.

For anything that is not critical, decide yourself, note the decision in the task note, and continue. If one task is blocked, move it to "Waiting for user" with the question and keep working on independent tasks; ask the collected questions together when no unblocked task is left.

## Paper repository

The paper lives at `~/Documents/papers/robustness_tda`. Edit it only where a `PLAN.md` task says so, e.g. setup or theory text, a reproducibility paragraph, or tables and figures copied from `figures/tmlr/`. Leave result statements that depend on new runs as `% TODO(PLAN x.y): …` comments until the user's runs are in. Do not commit in the paper repository.
