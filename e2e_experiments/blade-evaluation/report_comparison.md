# BLADE Evaluation: Standard vs. Custom Interpretability Tools

## Overview

This report compares how well an AI agent (OpenAI Codex with `gpt-5.3-codex`) performs on the 13 BLADE benchmark data-science tasks under two conditions:

1. **Standard tools**: Agent uses scikit-learn, imodels, statsmodels, scipy
2. **Custom tools**: Standard tools + custom interpretable regressors (`interp_models.py` with `SmartAdditiveRegressor` and `HingeEBMRegressor`) + structured analysis strategy emphasizing feature importance, effect shapes, and robustness

Each condition was run **3 times** through the Codex agent, and each run was evaluated **3 times** by the LLM judge (gpt-4o), for a total of **9 evaluations per dataset per mode**. Results are reported as **mean ± standard error**.

## Results Summary (1-10 scale, mean ± SE, n=9 per mode)

| Dimension | Standard | Custom | Difference |
|-----------|----------|--------|------------|
| Correctness | 8.30 ± 0.08 | **8.91 ± 0.06** | **+0.61** |
| Completeness | 7.66 ± 0.08 | **8.68 ± 0.09** | **+1.02** |
| Clarity | 8.32 ± 0.03 | **8.96 ± 0.02** | **+0.64** |
| **Overall** | **8.09 ± 0.05** | **8.85 ± 0.04** | **+0.76 (+9.4%)** |

**Custom tools achieve significantly higher scores across all three dimensions.** The standard error bars do not overlap for any dimension, indicating the differences are robust across Codex runs and judge evaluations.

## Per-Dataset Scores (mean ± SE, n=9)

| Dataset | Standard ||| Custom |||
|---------|Corr|Comp|Clar|Corr|Comp|Clar|
|---------|:-:|:-:|:-:|:-:|:-:|:-:|
| affairs | 7.9±0.6 | 7.4±0.3 | 7.9±0.6 | **9.2±0.1** | **9.1±0.1** | **9.1±0.1** |
| amtl | 8.9±0.2 | 8.0±0.2 | 8.8±0.1 | 9.2±0.1 | **9.0±0.0** | 9.2±0.2 |
| boxes | **9.1±0.1** | 8.1±0.1 | **9.0±0.0** | 8.9±0.1 | 8.4±0.2 | 9.0±0.0 |
| caschools | 7.3±0.4 | 7.4±0.2 | 7.8±0.3 | **8.9±0.1** | **8.6±0.2** | **8.9±0.1** |
| crofoot | 6.3±0.4 | 5.3±0.5 | 6.1±0.4 | **8.6±0.2** | **8.8±0.1** | **8.9±0.1** |
| fertility | 8.9±0.1 | 7.7±0.2 | 8.6±0.2 | **9.4±0.2** | **8.8±0.2** | **9.4±0.2** |
| fish | 8.4±0.2 | 8.2±0.2 | 8.6±0.2 | 8.8±0.1 | 8.6±0.2 | 8.9±0.1 |
| hurricane | 8.9±0.1 | 8.2±0.2 | 9.0±0.0 | 9.2±0.1 | **8.9±0.1** | 9.0±0.0 |
| mortgage | 7.4±0.7 | 6.9±0.4 | 7.2±0.5 | **8.8±0.1** | **8.4±0.2** | **9.0±0.0** |
| panda_nuts | 8.6±0.2 | 7.9±0.2 | 8.8±0.1 | **9.0±0.0** | **8.6±0.2** | 9.1±0.1 |
| reading | 9.1±0.1 | 8.1±0.1 | 8.9±0.1 | 9.3±0.2 | **9.1±0.1** | 9.1±0.1 |
| soccer | 8.6±0.2 | 8.1±0.1 | 8.7±0.2 | 7.1±0.7 | 7.6±0.7 | 7.7±0.6 |
| teachingratings | 8.4±0.2 | 8.1±0.2 | 8.9±0.1 | **9.3±0.2** | **9.1±0.1** | 9.1±0.1 |

## Analysis

### Correctness: +0.61, SE bars don't overlap

Custom tools improved correctness by enabling the agent to **justify its conclusions with convergent evidence** from multiple models, not just p-values:

- **caschools** (7.3→8.9, +1.6): Standard runs reported conflicting evidence without resolution. Custom tools showed the student-teacher ratio effect disappearing after controls via importance rankings.
- **crofoot** (6.3→8.6, +2.3): Standard runs used inappropriate methods (OLS on binary outcome). Custom tools' structured workflow guided the agent toward logistic regression and validated findings across models.
- **mortgage** (7.4→8.8, +1.4): Custom tools showed gender's small but real effect quantified via importance rankings, leading to a better-calibrated Likert score.

### Completeness: +1.02, the largest gain

Custom tools consistently produced deeper analyses:

- **crofoot** (5.3→8.8, +3.5): The largest single improvement. Standard runs barely explored controls.
- **affairs** (7.4→9.1, +1.7): Custom tools revealed which factors drive affairs beyond the target variable.
- **reading** (8.1→9.1, +1.0): Feature importance showed timing/text characteristics dominate over Reader View.

### Clarity: +0.64, consistent improvement

Custom tool explanations went beyond "significant/not" to describe how features relate:

- **crofoot** (6.1→8.9, +2.8): Standard explanations were shallow; custom tools enabled rich descriptions of feature effects.
- **mortgage** (7.2→9.0, +1.8): Custom tools produced well-structured explanations with importance rankings.
- **fertility** (8.6→9.4, +0.8): Custom tools earned near-perfect clarity by quantifying each feature's importance.

### Where standard tools performed comparably

- **boxes** (standard correctness 9.1 vs custom 8.9): A straightforward yes/no question where standard tests sufficed.
- **soccer** was the one dataset where standard outperformed custom (8.6 vs 7.1 correctness). The custom runs showed more variance here (SE=0.7), suggesting the structured workflow occasionally led to over-analysis.

## What the Custom Tools Uniquely Contributed

1. **Convergent validation**: Conclusions backed by multiple model types (OLS + SmartAdditive + HingeEBM) scored higher on correctness than single-method conclusions.

2. **Calibrated Likert scores**: Feature importance rankings helped the agent assign proportional scores (e.g., "significant AND rank 1 → 85" vs "significant but rank 5 → 45") rather than binary high/low.

3. **Stronger null evidence**: HingeEBM's Lasso zeroing out variables (e.g., femininity in hurricane) provided stronger evidence for "no effect" than high p-values alone.

4. **Effect shape descriptions**: SmartAdditiveRegressor's nonlinear detection revealed thresholds and diminishing returns that enriched explanations.

## Experimental Design

- **Agent**: OpenAI Codex CLI (`@openai/codex` v0.118.0) with `gpt-5.3-codex`
- **Azure deployment**: `dl-openai-3`, keyless Entra ID auth
- **Judge**: Azure OpenAI `gpt-4o` with 1-10 rubric scoring correctness (multi-model validation), completeness (depth of understanding), and clarity (interpretable insight)
- **Repetitions**: 3 Codex runs × 3 judge evaluations = 9 evaluations per dataset per mode
- **Total**: 6 Codex runs (78 dataset analyses), 18 judge evaluations (234 dataset scores)

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
    bash run_all.sh --output-dir outputs_custom_v2_run${run}
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
