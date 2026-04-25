# BLADE Evaluation: Standard vs. agentic-imodels SKILL

## Overview

This report compares how well an AI agent (OpenAI Codex with `gpt-5.3-codex`) performs on the 13 BLADE benchmark data-science tasks under two conditions:

1. **Standard tools**: Agent uses scikit-learn, imodels, statsmodels, scipy.
2. **Custom tools (`agentic-imodels` skill)**: Standard tools plus the
   evolved `agentic_imodels` package (10 regressors) and a `SKILL.md` that
   documents the API, model-selection table, and recommended analysis
   workflow.

Each condition was run **3 times** through the Codex agent, and each run was evaluated **3 times** by the LLM judge (gpt-4o), for a total of **9 evaluations per dataset per mode**. Results are reported as **mean ± standard error**.

## Results Summary (1-10 scale, mean ± SE, n=9 per mode)

| Dimension     | Standard    | Custom       | Difference       |
|---------------|-------------|--------------|------------------|
| Correctness   | 8.30 ± 0.08 | **8.94 ± 0.06** | **+0.64** |
| Completeness  | 7.66 ± 0.08 | **8.37 ± 0.06** | **+0.71** |
| Clarity       | 8.32 ± 0.03 | **8.88 ± 0.05** | **+0.56** |
| **Overall**   | **8.09 ± 0.05** | **8.73 ± 0.05** | **+0.64 (+7.9%)** |

**All 13 / 13 datasets improved** with the custom tools. SE bars do not overlap on any dimension.

## Per-Dataset Scores (mean ± SE, n=9)

| Dataset         | Std Corr | Std Comp | Std Clar | Cus Corr | Cus Comp | Cus Clar |
|-----------------|----------|----------|----------|----------|----------|----------|
| affairs         | 7.9±0.6  | 7.4±0.3  | 7.9±0.6  | **9.2±0.1** | **8.9±0.3** | **8.9±0.2** |
| amtl            | 8.9±0.2  | 8.0±0.2  | 8.8±0.1  | 8.8±0.1  | 8.3±0.2  | 8.9±0.1  |
| boxes           | 9.1±0.1  | 8.1±0.1  | 9.0±0.0  | 9.3±0.2  | 8.6±0.2  | 9.0±0.0  |
| caschools       | 7.3±0.4  | 7.4±0.2  | 7.8±0.3  | **9.1±0.1** | **8.4±0.2** | **9.0±0.0** |
| crofoot         | 6.3±0.4  | 5.3±0.5  | 6.1±0.4  | **8.4±0.2** | **8.0±0.2** | **8.6±0.2** |
| fertility       | 8.9±0.1  | 7.7±0.2  | 8.6±0.2  | 9.2±0.1  | 8.3±0.2  | 8.9±0.2  |
| fish            | 8.4±0.2  | 8.2±0.2  | 8.6±0.2  | 8.9±0.1  | 8.1±0.1  | 8.9±0.1  |
| hurricane       | 8.9±0.1  | 8.2±0.2  | 9.0±0.0  | 9.0±0.0  | 8.3±0.2  | 9.0±0.0  |
| mortgage        | 7.4±0.7  | 6.9±0.4  | 7.2±0.5  | **8.0±0.4** | **7.8±0.4** | **8.3±0.4** |
| panda_nuts      | 8.6±0.2  | 7.9±0.2  | 8.8±0.1  | 9.0±0.0  | 8.4±0.2  | 8.9±0.1  |
| reading         | 9.1±0.1  | 8.1±0.1  | 8.9±0.1  | **9.7±0.2** | **9.0±0.0** | 9.2±0.1  |
| soccer          | 8.6±0.2  | 8.1±0.1  | 8.7±0.2  | 8.6±0.2  | 8.2±0.1  | 8.9±0.1  |
| teachingratings | 8.4±0.2  | 8.1±0.2  | 8.9±0.1  | 9.0±0.2  | 8.3±0.2  | 9.0±0.0  |

## Analysis

### Correctness (+0.64)

- **crofoot** (6.3→8.4): Standard runs used inappropriate methods (OLS on a binary outcome). The `SKILL.md` workflow guided the agent toward a controlled logistic regression and cross-model corroboration.
- **caschools** (7.3→9.1): The agent picks a GAM plus a sparse linear model and reports that the student-teacher ratio effect weakens under controls.
- **affairs** (7.9→9.2): Bivariate tests flip sign against the hypothesis, the controlled GLM is null, and two sparse models zero out the IV — the agent now writes a calibrated "strong No" (Likert 5–10) rather than a weakly-justified 30-ish.

### Completeness (+0.71)

- **crofoot** (5.3→8.0): The agent fits `SmartAdditiveRegressor`, `HingeEBMRegressor`, and `WinsorizedSparseOLS` and reports direction/importance/shape for every feature.
- **mortgage** (6.9→7.8): Importance rankings surface gender's small but real effect; the agent quantifies it in the conclusion.
- **affairs** (7.4→8.9): The agent uses counterfactual toggling on the binary IV + feature-importance rankings to show the bivariate association is not robust.

### Clarity (+0.56)

- Explanations increasingly quote the `str(model)` form directly — sparse equations for `HingeEBM`/`HingeGAM`, piecewise tables for `SmartAdditive` — which the judge rewards.

### No regression on soccer

Unlike the previous two-model custom prompt, the SKILL-based setup **does not regress on soccer** (standard 8.4 avg, custom 8.6 avg). All 13 / 13 datasets improve.

## Evaluation Methodology: BLADE Ground-Truth

The evaluation checks each AI agent's analysis against **human expert annotations** from the [BLADE benchmark](https://github.com/behavioral-data/BLADE). The ground-truth annotations (`annotations.csv` per dataset) are summarized and passed to an **LLM-as-a-judge** (GPT-4o) alongside the agent's conclusion. The judge scores three dimensions (1-10 each):

1. **Correctness**: Is the conclusion well-supported and well-calibrated relative to what experts found?
2. **Completeness**: Does the analysis go beyond basic tests to deeply understand the data, as experts did?
3. **Clarity**: Does the explanation convey interpretable insight about feature effects and relationships?

The judge prompt explicitly instructs scoring based on convergent evidence, effect calibration, and depth of understanding — not just whether the agent got the "right answer."

## Experimental Design

- **Agent**: OpenAI Codex CLI (`@openai/codex` v0.118.0) with `gpt-5.3-codex`, reasoning effort `high`
- **Azure deployment**: `dl-openai-3`, keyless Entra ID auth
- **Custom library**: `agentic_imodels` (10 evolved regressors) + `SKILL.md`, copied into each run directory
- **Ground truth**: BLADE human expert annotations
- **Judge**: Azure OpenAI `gpt-4o` with 1-10 rubric
- **Repetitions**: 3 Codex runs × 3 judge evaluations = 9 evaluations per dataset per mode

## How to Reproduce

```bash
# Prepare datasets
for run in 1 2 3; do
    python prepare_run.py --mode standard --output-dir outputs_standard_run${run}
    python prepare_run.py --mode custom_v2 --output-dir outputs_custom_v2_run${run}
done

# Run Codex
for run in 1 2 3; do
    bash run_all.sh --output-dir outputs_standard_run${run}
    bash run_all.sh --mode custom_v2 --output-dir outputs_custom_v2_run${run}
done

# Evaluate (3 judge repeats per run)
for mode in standard custom_v2; do
    for run in 1 2 3; do
        for judge in 1 2 3; do
            python evaluate.py --output-dir outputs_${mode}_run${run} \
                --results-path judge_results/results_${mode}_run${run}_judge${judge}.csv
        done
    done
done

# Aggregate results
python aggregate_results.py
```
