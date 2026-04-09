# BLADE Evaluation Report: AI Agent on End-to-End Data Science Tasks

## Overview

This evaluation assesses how well an AI agent system performs on end-to-end data science tasks from the [BLADE benchmark](https://github.com/behavioral-data/blade) (Benchmark for Language-model-based Analysis of Data Experiments). BLADE contains 13 datasets, each paired with a research question and expert-annotated statistical analysis specifications.

## Setup

- **Agent**: OpenAI Codex CLI (`@openai/codex` v0.118.0) via Azure OpenAI with keyless Entra ID authentication
- **Model**: `o4-mini` (Azure deployment: `dl-openai-1`)
- **Configuration**: `model_reasoning_effort="high"`, sandbox mode `danger-full-access`
- **Fallback**: For datasets where Codex did not execute its scripts, a generic statistical analysis pipeline (`run_analysis.py`) was used
- **Evaluation**: LLM-as-a-judge (Azure OpenAI `gpt-4o` via `dl-openai-3`) scoring correctness, completeness, and clarity (1-5 each)
- **Authentication**: Keyless via `ChainedTokenCredential` (AzureCli -> ManagedIdentity)

## Datasets

| # | Dataset | Research Question (abbreviated) |
|---|---------|-------------------------------|
| 1 | affairs | Extramarital affairs and relationship factors |
| 2 | amtl | AMTL analysis |
| 3 | boxes | Boxes experiment |
| 4 | caschools | California school test scores and class size |
| 5 | crofoot | Crofoot behavioral study |
| 6 | fertility | Fertility dataset analysis |
| 7 | fish | Fish dataset analysis |
| 8 | hurricane | Hurricane name femininity and fatalities |
| 9 | mortgage | Mortgage data analysis |
| 10 | panda_nuts | Panda nuts experiment |
| 11 | reading | Reading study |
| 12 | soccer | Skin tone and red cards in soccer |
| 13 | teachingratings | Teaching ratings and instructor characteristics |

## Methodology

For each dataset, the Codex agent:
1. Receives an `AGENTS.md` prompt, `info.json` (metadata + research question), and the dataset CSV
2. Autonomously explores the data, selects statistical methods, and produces an analysis
3. Outputs a `conclusion.txt` with a Likert score (0-100) and written explanation

The LLM-as-a-judge evaluation compares each conclusion against human expert annotations (from `annotations.csv`) on three criteria:
- **Correctness** (1-5): Does the analysis use sound methodology and reach a defensible conclusion?
- **Completeness** (1-5): Does it consider confounders, data issues, and alternative explanations?
- **Clarity** (1-5): Is the explanation well-structured and precise?

## Results

### Summary Scores

| Dataset | Response | Correctness | Completeness | Clarity |
|---------|----------|-------------|--------------|---------|
| affairs | 0 | 3 | 2 | 4 |
| amtl | 100 | 2 | 2 | 3 |
| boxes | 0 | 1 | 1 | 2 |
| caschools | 100 | 2 | 2 | 3 |
| crofoot | 0 | 2 | 1 | 2 |
| fertility | 0 | 3 | 2 | 4 |
| fish | 61 | 2 | 2 | 3 |
| hurricane | 0 | 2 | 1 | 3 |
| mortgage | 0 | 3 | 2 | 4 |
| panda_nuts | 100 | 2 | 2 | 3 |
| reading | 100 | 2 | 2 | 3 |
| soccer | 50 | 2 | 1 | 3 |
| teachingratings | 100 | 2 | 1 | 2 |
| **AVERAGE** | | **2.15** | **1.62** | **3.00** |

**Overall average score: 2.26 / 5.00**

### Key Findings

1. **Correctness (avg 2.15/5)**: The AI agent frequently chose inappropriate statistical tests. Common issues:
   - Used Pearson correlation for count data (hurricane fatalities, fish catches) instead of Poisson/Negative Binomial regression
   - Applied t-tests or ANOVA when regression was needed (caschools, reading)
   - Used correlation for binary/categorical outcomes (boxes, crofoot) instead of logistic regression

2. **Completeness (avg 1.62/5)**: The weakest dimension. The AI consistently failed to:
   - Include control variables identified by human experts
   - Consider confounders and alternative explanations
   - Apply data transformations before modeling
   - Use multiple model specifications as experts did

3. **Clarity (avg 3.00/5)**: The strongest dimension. Explanations were generally structured and readable, though often shallow in statistical justification.

### Observations

- **Codex execution reliability**: Only 3/13 datasets (affairs, fertility, hurricane) had Codex successfully write AND execute analysis scripts. The remaining 10 required fallback to a generic analysis pipeline.
- **Model selection gap**: Human experts used GLM, Poisson, Negative Binomial, logistic regression, and OLS with controls. The AI defaulted to simpler methods (correlation, t-tests).
- **Variable selection gap**: Human annotations identified specific IVs, DVs, and control variables. The AI often tested irrelevant variable pairs.

## How to Run

```bash
cd blade-evaluation

# 1. Setup Azure OpenAI authentication (writes ~/.codex/config.toml)
source setup_azure.sh

# 2. Prepare dataset directories
python prepare_run.py

# 3. Run Codex on all datasets
bash run_all.sh

# 4. Fallback: run generic analysis on datasets missing conclusions
python run_analysis.py --all

# 5. Run LLM-as-a-judge evaluation
python evaluate.py --verbose
```

## File Structure

```
blade-evaluation/
├── setup_azure.sh      # Azure OpenAI + Codex CLI configuration
├── refresh_token.sh    # Entra ID token refresh (legacy, for scripts needing env var)
├── prepare_run.py      # Prepares dataset directories under outputs/
├── run_all.sh          # Runs Codex on all 13 datasets
├── run_analysis.py     # Generic statistical analysis fallback
├── evaluate.py         # LLM-as-a-judge evaluation (keyless Azure auth)
├── REPORT.md           # This report
├── results.csv         # Generated evaluation results
└── outputs/
    ├── affairs/        # Each contains: AGENTS.md, info.json, {name}.csv,
    ├── amtl/           #   packages.txt, and (after run) conclusion.txt
    ├── ...
    └── teachingratings/
```
