# autoresearch — interpretable regressors

This is an experiment to have the LLM autonomously research scikit-learn regressors that score well on two metrics: predictive performance (mean RMSE) and interpretability (fraction of LLM-graded tests passed).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar20`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `readme.md` — repository context.
   - `run_baselines.py` — the fixed baseline evaluation harness. This is NOT modified by the agent and has already been run for you.
   - `interpretable_regressor.py` — the file you modify. Regressor definition and evaluation loop.
   - `results/overall_results.csv` — current scores for all models including baselines.
4. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Run an experiment with: `uv run interpretable_regressor.py`

This trains `InterpretableRegressor`, runs interpretability tests, and updates `results/overall_results.csv`.

**What you CAN do:**
- Modify `interpretable_regressor.py` — this is the only file you edit. Everything is fair game:
  - The `InterpretableRegressor` class definition (algorithm, structure, hyperparameters)
  - Switching to another model type (rule lists, linear models, GAMs, sparse models, etc.)
  - Feature engineering or preprocessing inside the regressor
  - Hyperparameter values
  - The training loop structure

**What you CANNOT do:**
- Modify `run_baselines.py`. It is read-only.
- Modify anything in the `eval/` folder. It contains the ground truth tests.
- Install new packages. You can only use what's already in `pyproject.toml`.

## Goal

Optimize both metrics in `results/overall_results.csv`:

- **`mean_rmse`** — mean RMSE across TabArena regression datasets (lower is better)
- **`frac_interpretability_tests_passed`** — fraction of LLM-graded interpretability tests passed (higher is better)

Both metrics matter. A model that scores well on RMSE but poorly on interpretability tests, or vice versa, is not ideal. Look at the baseline scores in `overall_results.csv` to understand the trade-off space. We want pareto improvements over the baselines.

**The first run**: Your very first run should always be to establish the baseline — run the script as is, record the results.

## Output format

Once the script finishes it prints a summary like this:

```
---
tests_passed:  5/18 (27.78%)  [std 0/8  hard 0/5  insight 0/5]
total_seconds: 0.8s
```

It also updates `results/overall_results.csv` with the row for `InterpretableRegressor`.

## Logging results

When an experiment is done, it should log to the `results/overall_results.csv`

The CSV has a header row and 5 columns:

```
commit	mean_rmse	frac_interpretability_tests_passed	status	description
```

1. git commit hash (short, 7 chars)
2. mean_rmse achieved — use empty for crashes (note: interpretable_regressor.py does not compute mean_rmse; check overall_results.csv for the baseline value to compare against)
3. frac_interpretability_tests_passed — from the script output
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried - make sure to update this clearly

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar20`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Edit `interpretable_regressor.py` with an experimental idea
3. git commit
4. Run the experiment: `uv run interpretable_regressor.py > run.log 2>&1`
5. Read results: `tail -n 5 run.log` and `grep InterpretableRegressor results/overall_results.csv`
6. If the run crashed, check `tail -n 50 run.log` for the stack trace and attempt a fix
7. Record results in `results/overall_results.csv` (do not commit this file)
8. If either metric improved without the other getting significantly worse, keep the commit
9. Otherwise, `git reset --hard` back to the previous commit

**NEVER STOP**: Once the experiment loop has begun, do NOT pause to ask the human if you should continue. Run until manually stopped.

**Ideas to try** (not exhaustive — be creative):
- Try novel splitting criteria
- Try novel ways to induce sparsity or perform elaborate feature selection
- Try new regularization techniques

Do not simply import a known interpretable model and change its hyperparameters — build your own from scratch using basic building blocks or substantially modify an existing one. The goal is to discover new models, not just find the best hyperparameters for known models.