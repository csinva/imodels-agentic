"""Prepare run directories for each Blade dataset.

For each dataset, creates a subdirectory under the output directory containing:
  - info.json   (task metadata with research question)
  - {dataset}.csv  (the data)
  - AGENTS.md   (instructions for Codex)
  - interp_models.py (custom interpretability tools, only in --mode custom)

Usage:
    python prepare_run.py                          # prepare all datasets (standard mode)
    python prepare_run.py --mode custom            # prepare with custom interp tools
    python prepare_run.py --dataset soccer         # prepare one dataset
"""

import argparse
import json
import os
import shutil
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Try example-blade-repo first, fall back to existing outputs/ directory
_BLADE_DIR = os.path.join(
    SCRIPT_DIR, "..", "example-blade-repo", "blade_bench", "datasets"
)
DATASETS_DIR = _BLADE_DIR if os.path.isdir(_BLADE_DIR) else os.path.join(SCRIPT_DIR, "outputs")

DATASETS = [
    "affairs",
    "amtl",
    "boxes",
    "caschools",
    "crofoot",
    "fertility",
    "fish",
    "hurricane",
    "mortgage",
    "panda_nuts",
    "reading",
    "soccer",
    "teachingratings",
]

AGENTS_MD_STANDARD = """You are an expert data scientist. You MUST write and execute a Python script to analyze a dataset and answer a research question.

## Instructions

1. Read `info.json` to get the research question and dataset metadata.
2. Load the dataset from `{dataset_name}.csv`.
3. Write a Python script called `analysis.py` that:
   - Loads and explores the data (summary statistics, distributions, correlations)
   - Builds interpretable models using scikit-learn and imodels to understand feature relationships
   - Performs appropriate statistical tests (t-tests, ANOVA, regression, etc.)
   - Interprets the results in context of the research question
4. **Execute the script** by running: `python3 analysis.py`
5. The script MUST write a file called `conclusion.txt` containing ONLY a JSON object:

```json
{{"response": <integer 0-100>, "explanation": "<your reasoning>"}}
```

Where `response` is a Likert scale score: 0 = strong "No", 100 = strong "Yes".

## Interpretability Tools

You should heavily use interpretable models to understand the data. Available tools:

- **scikit-learn**: Use `LinearRegression`, `Ridge`, `Lasso`, `DecisionTreeRegressor`, `DecisionTreeClassifier` for interpretable models. Use `feature_importances_` and `coef_` to understand feature effects.
- **imodels**: Use `from imodels import RuleFitRegressor, FIGSRegressor, HSTreeRegressor` for rule-based and tree-based interpretable models. These provide human-readable rules and feature importance.
- **statsmodels**: Use `statsmodels.api.OLS` for regression with p-values and confidence intervals.
- **scipy.stats**: Use for statistical tests (t-test, chi-square, correlation, ANOVA).

Focus on building interpretable models that help you understand the relationships in the data, not just black-box predictions. Use the model coefficients, rules, and feature importances to inform your conclusion.

## Important

- You MUST actually run the script, not just write it. The `conclusion.txt` file must exist when you are done.
- When asked if a relationship between two variables exists, use statistical significance tests.
- Relationships lacking significance should receive a "No" (low score), significant ones a "Yes" (high score).
- Available packages: numpy, pandas, scipy, statsmodels, sklearn, imodels, matplotlib, seaborn.
"""

AGENTS_MD_CUSTOM_V2 = """You are an expert data scientist. You MUST write and execute a Python script to analyze a dataset and answer a research question.

## Instructions

1. Read `info.json` to get the research question and dataset metadata.
2. Load the dataset from `{dataset_name}.csv`.
3. Write a Python script called `analysis.py` that follows the analysis strategy below.
4. **Execute the script** by running: `python3 analysis.py`
5. The script MUST write a file called `conclusion.txt` containing ONLY a JSON object:

```json
{{"response": <integer 0-100>, "explanation": "<your reasoning>"}}
```

Where `response` is a Likert scale score: 0 = strong "No", 100 = strong "Yes".

## Analysis Strategy

### Step 1: Understand the question and explore
- Read the research question. Identify the dependent variable (DV) and independent variable (IV).
- Print summary statistics, check distributions, compute bivariate correlations.

### Step 2: Statistical tests with controls
Run OLS (or logistic regression for binary DVs) with relevant control variables:

```python
import statsmodels.api as sm
X = df[feature_columns]
X = sm.add_constant(X)
model = sm.OLS(df[dv_column], X).fit()
print(model.summary())
```

### Step 3: Use custom interpretable models to understand HOW features affect the outcome

This is where you go beyond p-values. The file `interp_models.py` provides two
models that reveal the **shape, direction, and relative importance** of each feature.

```python
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

# Include ALL numeric columns — pass the DataFrame directly for column names
X = df[numeric_columns]
y = df[dv_column]

# SmartAdditiveRegressor: reveals nonlinear effects and thresholds
smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X, y)   # Automatically uses column names from DataFrame
print(smart)       # Shows per-feature effects: linear slopes AND nonlinear patterns
                   # e.g., "age: nonlinear effect (importance=32.1%)"
                   #        "age <= 25: -0.42"
                   #        "age > 25: +0.31"  <-- threshold at 25!

effects = smart.feature_effects()
print(effects)
# {{'age': {{'direction': 'nonlinear (increasing trend)', 'importance': 0.321, 'rank': 1}},
#  'income': {{'direction': 'positive', 'importance': 0.198, 'rank': 2}},
#  'gender': {{'direction': 'zero', 'importance': 0.0, 'rank': 0}}}}
# -> age is the most important predictor, income is second, gender doesn't matter

# HingeEBMRegressor: sparse linear model that zeroes out unimportant features
hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X, y)
print(hinge)       # Shows clean equation: y = 0.52*income + -0.13*age + 1.23
print(hinge.feature_effects())
```

### Step 4: Write a rich conclusion

Your explanation should go BEYOND just "significant or not." Include:
- **Direction**: Is the effect positive or negative?
- **Magnitude**: How strong is it relative to other features? (use importance rankings)
- **Shape**: Is it linear, or does it have thresholds/nonlinear patterns?
- **Robustness**: Does the relationship hold across multiple models (OLS, SmartAdditive, HingeEBM)?
- **Confounders**: Which other variables also matter, and do they change the story?

Example good explanation: "Hours fishing has a significant positive effect on fish
caught (OLS coef=0.34, p=0.002). The SmartAdditive model confirms this with hours
ranked 2nd in importance (19.8%%), showing a roughly linear positive effect. The
relationship is robust after controlling for livebait, persons, and camper. Livebait
is actually the strongest predictor (importance=45.2%%), suggesting that bait choice
matters more than time spent."

### Scoring guidelines
- Strong significant effect that persists across models -> 75-100
- Moderate or partially significant effect -> 40-70
- Weak, inconsistent, or marginal effect -> 15-40
- No significant effect in any analysis -> 0-15
- Weigh BOTH bivariate and controlled results. If the effect weakens but doesn't
  vanish with controls, give a moderate score reflecting the partial effect.

## Custom Interpretability Tools Reference

**SmartAdditiveRegressor** — Learns additive per-feature shape functions:
- Accepts DataFrames directly (column names in output automatically)
- `str(model)`: Shows each feature's effect — linear coefficients for linear features,
  piecewise-constant lookup tables for nonlinear features (with thresholds!)
- `model.feature_effects()`: Returns dict with direction, importance (0-1), and rank
- Best for: discovering nonlinear effects, thresholds, feature importance rankings

**HingeEBMRegressor** — Sparse piecewise-linear model:
- Accepts DataFrames directly
- `str(model)`: Shows sparse equation with only important features (Lasso selection)
- `model.feature_effects()`: Returns dict with direction, importance, and rank
- Best for: identifying which features truly matter (others get zeroed out)

## Important

- You MUST actually run the script. The `conclusion.txt` file must exist.
- Use the custom models from `interp_models.py` to understand feature relationships.
- Report feature importance rankings and effect shapes in your explanation.
- Available packages: numpy, pandas, scipy, statsmodels, sklearn, imodels, interpret, matplotlib, seaborn.
"""


def get_installed_packages():
    """Get list of installed Python packages for reference."""
    try:
        result = subprocess.run(
            ["pip", "list", "--format=columns"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout
    except Exception:
        return "numpy\npandas\nscipy\nstatsmodels\nsklearn\nmatplotlib\nseaborn\nimodels\ninterpret\n"


def prepare_dataset(dataset_name: str, mode: str, output_dir: str):
    """Create a run directory for a single dataset."""
    src_dir = os.path.join(DATASETS_DIR, dataset_name)
    dst_dir = os.path.join(output_dir, dataset_name)

    if not os.path.isdir(src_dir):
        print(f"  SKIP: {dataset_name} (source not found at {src_dir})")
        return False

    os.makedirs(dst_dir, exist_ok=True)

    # Copy info.json
    shutil.copy2(os.path.join(src_dir, "info.json"), os.path.join(dst_dir, "info.json"))

    # Copy data CSV (may be data.csv or {dataset_name}.csv)
    src_csv = os.path.join(src_dir, "data.csv")
    if not os.path.exists(src_csv):
        src_csv = os.path.join(src_dir, f"{dataset_name}.csv")
    if os.path.exists(src_csv):
        shutil.copy2(src_csv, os.path.join(dst_dir, f"{dataset_name}.csv"))
    else:
        print(f"  WARN: {dataset_name} - no CSV found")
        return False

    # Write AGENTS.md based on mode
    templates = {
        "standard": AGENTS_MD_STANDARD,
        "custom_v2": AGENTS_MD_CUSTOM_V2,
    }
    template = templates[mode]
    with open(os.path.join(dst_dir, "AGENTS.md"), "w") as f:
        f.write(template.format(dataset_name=dataset_name))

    # Copy interp_models.py for custom modes
    if mode == "custom_v2":
        shutil.copy2(
            os.path.join(SCRIPT_DIR, "interp_models.py"),
            os.path.join(dst_dir, "interp_models.py"),
        )

    # Write packages.txt
    with open(os.path.join(dst_dir, "packages.txt"), "w") as f:
        f.write(get_installed_packages())

    print(f"  OK: {dataset_name} -> {dst_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Prepare Blade dataset run directories")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single dataset to prepare (default: all 13)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["standard", "custom_v2"],
        default="standard",
        help="Tool mode: 'standard' (sklearn/imodels) or 'custom_v2' (+ interp_models.py)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Explicit output directory name (overrides default outputs_{mode})",
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = os.path.join(SCRIPT_DIR, args.output_dir)
    else:
        output_dir = os.path.join(SCRIPT_DIR, f"outputs_{args.mode}")
    datasets = [args.dataset] if args.dataset else DATASETS
    print(f"Preparing {len(datasets)} dataset(s) in {args.mode} mode...")

    success = 0
    for ds in datasets:
        if prepare_dataset(ds, args.mode, output_dir):
            success += 1

    print(f"\nDone: {success}/{len(datasets)} datasets prepared in {output_dir}/")


if __name__ == "__main__":
    main()
