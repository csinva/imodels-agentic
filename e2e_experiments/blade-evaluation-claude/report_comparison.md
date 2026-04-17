# Blade Evaluation — Claude Code Agent

This folder mirrors `blade-evaluation/` but drives the agent with the Claude
Code CLI (`claude -p`, model `sonnet`) instead of OpenAI Codex. The Blade task
prompts, dataset preparation, LLM-as-a-judge scoring (Azure OpenAI), and
aggregation scripts are otherwise unchanged so results are directly comparable.

## Setup

- Agent: `claude -p --model sonnet --permission-mode bypassPermissions`
- Judge: Azure OpenAI `gpt-4o` via keyless Entra ID auth (unchanged from
  original pipeline)
- Datasets: 13 Blade tasks, sourced from
  `../blade-evaluation/outputs_standard_run1/` (info.json + CSV per dataset)
- Modes:
  - `standard` — agent instructed to use scikit-learn / imodels / statsmodels
  - `custom_v2` — `standard` plus `interp_models.py`
    (`SmartAdditiveRegressor`, `HingeEBMRegressor`) included in each run dir
- Runs: 1 agent run per mode × 1 judge run = 13 evaluations per mode

## Results (mean over 13 datasets, 1–10 scale)

| Dimension     | standard | custom_v2 | Δ (custom − std) |
| ------------- | -------- | --------- | ---------------- |
| Correctness   | 6.92     | 8.38      | +1.46            |
| Completeness  | 6.00     | 8.15      | +2.15            |
| Clarity       | 6.62     | 8.62      | +2.00            |
| **Overall**   | **6.51** | **8.38**  | **+1.87**        |

All 13/13 datasets produced a valid `conclusion.txt` in both modes.

## Observations

- Giving Claude the custom interpretability tools (`SmartAdditiveRegressor`,
  `HingeEBMRegressor`) raises the overall judge score from **6.51 → 8.38**,
  with the largest lifts on completeness (+2.15) and clarity (+2.00).
- The `custom_v2` explanations more consistently report feature-importance
  rankings, effect directions/shapes, and robustness across models — the
  dimensions the judge rubric rewards explicitly.
- `standard` struggles most on `boxes` (3/2/3) and `crofoot` (5/3/4), where
  the agent stopped at basic bivariate tests; `custom_v2` keeps both ≥8 on
  every dimension.
- `fish` is the only dataset where `custom_v2` slightly underperforms on
  correctness (4 vs 4), driven by an over-confident "Yes" score that the
  judge flagged as miscalibrated.

## Files

- `run_all.sh` — runs `claude -p` on each dataset's run directory
- `prepare_run.py` — builds per-dataset run dirs (uses sibling repo's data)
- `evaluate.py` — Azure OpenAI LLM-as-a-judge (unchanged from original)
- `aggregate_results.py` — aggregates judge CSVs (unchanged)
- `outputs_standard_run1/`, `outputs_custom_v2_run1/` — per-dataset working
  dirs with `analysis.py`, `conclusion.txt`, and Claude CLI logs
- `judge_results/results_{mode}_run1_judge1.csv` — per-dataset scores
- `judge_results/results_aggregated.csv` — summary across modes
