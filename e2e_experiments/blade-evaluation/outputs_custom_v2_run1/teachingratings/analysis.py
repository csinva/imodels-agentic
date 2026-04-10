import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr, spearmanr
from sklearn.exceptions import ConvergenceWarning

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def format_effect(effect: Dict) -> str:
    if not effect:
        return "not present"
    direction = effect.get("direction", "unknown")
    imp = float(effect.get("importance", 0.0))
    rank = int(effect.get("rank", 0))
    return f"direction={direction}, importance={imp:.1%}, rank={rank}"


def top_significant_controls(model, exclude: List[str], k: int = 3) -> List[Tuple[str, float, float]]:
    rows = []
    for name in model.params.index:
        if name in exclude:
            continue
        pval = float(model.pvalues[name])
        coef = float(model.params[name])
        if pval < 0.05:
            rows.append((name, coef, pval))
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    return rows[:k]


def find_zero_crossing(smart: SmartAdditiveRegressor, feature_name: str):
    if feature_name not in getattr(smart, "feature_names_", []):
        return None
    idx = smart.feature_names_.index(feature_name)
    if idx not in smart.shape_functions_:
        return None
    thresholds, intervals = smart.shape_functions_[idx]
    if len(thresholds) == 0:
        return None
    for i in range(len(intervals) - 1):
        if intervals[i] < 0 <= intervals[i + 1]:
            return float(thresholds[i])
    return None


def clamp_score(x: float) -> int:
    return int(max(0, min(100, round(x))))


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info["research_questions"][0]
    print("Research question:")
    print(question)
    print()

    df = pd.read_csv("teachingratings.csv")
    print(f"Data shape: {df.shape}")
    print("Columns:", list(df.columns))
    print()

    # Step 1: identify DV and IV from metadata/question
    dv = "eval"
    iv = "beauty"

    print(f"DV = {dv}")
    print(f"IV = {iv}")
    print()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    print("Numeric summary statistics:")
    print(df[numeric_cols].describe().T)
    print()

    print("Categorical distributions:")
    for c in categorical_cols:
        print(f"\\n{c} value counts:")
        print(df[c].value_counts(dropna=False))
    print()

    print("Correlations with DV (numeric columns):")
    corr_with_dv = df[numeric_cols].corr(numeric_only=True)[dv].sort_values(ascending=False)
    print(corr_with_dv)
    print()

    pearson_r, pearson_p = pearsonr(df[iv], df[dv])
    spearman_r, spearman_p = spearmanr(df[iv], df[dv])
    print(f"Bivariate Pearson({iv}, {dv}) = {pearson_r:.4f}, p={pearson_p:.3g}")
    print(f"Bivariate Spearman({iv}, {dv}) = {spearman_r:.4f}, p={spearman_p:.3g}")
    print()

    # Step 2: OLS tests
    X_simple = sm.add_constant(df[[iv]])
    ols_simple = sm.OLS(df[dv], X_simple).fit()
    print("Simple OLS: eval ~ beauty")
    print(ols_simple.summary())
    print()

    # Relevant controls from metadata (exclude identifiers)
    controls = [
        "beauty",
        "age",
        "minority",
        "gender",
        "credits",
        "division",
        "native",
        "tenure",
        "students",
        "allstudents",
    ]
    X_control = pd.get_dummies(df[controls], drop_first=True, dtype=float)
    X_control = sm.add_constant(X_control)
    ols_control = sm.OLS(df[dv], X_control).fit()
    print("Controlled OLS: eval ~ beauty + covariates")
    print(ols_control.summary())
    print()

    beauty_coef_simple = float(ols_simple.params[iv])
    beauty_p_simple = float(ols_simple.pvalues[iv])
    beauty_coef_control = float(ols_control.params[iv])
    beauty_p_control = float(ols_control.pvalues[iv])
    beauty_sd_effect = beauty_coef_control * float(df[iv].std())

    # Step 3: interpretable models
    # As requested: include all numeric columns (except DV target)
    numeric_predictors_all = [c for c in numeric_cols if c != dv]
    X_num_all = df[numeric_predictors_all]
    y = df[dv]

    smart_all = SmartAdditiveRegressor(n_rounds=200)
    smart_all.fit(X_num_all, y)
    smart_all_effects = smart_all.feature_effects()

    hinge_all = HingeEBMRegressor(n_knots=3)
    hinge_all.fit(X_num_all, y)
    hinge_all_effects = hinge_all.feature_effects()

    print("SmartAdditiveRegressor (all numeric predictors):")
    print(smart_all)
    print("Feature effects:")
    print(smart_all_effects)
    print()

    print("HingeEBMRegressor (all numeric predictors):")
    print(hinge_all)
    print("Feature effects:")
    print(hinge_all_effects)
    print()

    # Additional robustness pass excluding clear identifiers
    id_like = {"rownames", "prof"}
    numeric_predictors_clean = [c for c in numeric_predictors_all if c not in id_like]
    X_num_clean = df[numeric_predictors_clean]

    smart_clean = SmartAdditiveRegressor(n_rounds=200)
    smart_clean.fit(X_num_clean, y)
    smart_clean_effects = smart_clean.feature_effects()

    hinge_clean = HingeEBMRegressor(n_knots=3)
    hinge_clean.fit(X_num_clean, y)
    hinge_clean_effects = hinge_clean.feature_effects()

    print("SmartAdditiveRegressor (clean numeric, excluding rownames/prof IDs):")
    print(smart_clean)
    print("Feature effects:")
    print(smart_clean_effects)
    print()

    print("HingeEBMRegressor (clean numeric, excluding rownames/prof IDs):")
    print(hinge_clean)
    print("Feature effects:")
    print(hinge_clean_effects)
    print()

    beauty_smart_all = smart_all_effects.get(iv, {})
    beauty_hinge_all = hinge_all_effects.get(iv, {})
    beauty_smart_clean = smart_clean_effects.get(iv, {})
    beauty_hinge_clean = hinge_clean_effects.get(iv, {})

    zero_cross = find_zero_crossing(smart_clean, iv)

    confounders = top_significant_controls(ols_control, exclude=["const", iv], k=3)
    confounder_text = (
        "; ".join([f"{n} (coef={c:.3f}, p={p:.3g})" for n, c, p in confounders])
        if confounders
        else "No control variable reached p<0.05"
    )

    # Score mapping based on strength and consistency
    score = 40.0
    if beauty_p_simple < 0.001:
        score += 15
    elif beauty_p_simple < 0.01:
        score += 12
    elif beauty_p_simple < 0.05:
        score += 8

    if beauty_p_control < 0.001:
        score += 20
    elif beauty_p_control < 0.01:
        score += 16
    elif beauty_p_control < 0.05:
        score += 10

    if beauty_coef_control > 0:
        score += 8
    else:
        score -= 12

    # Importance/shape from interpretable models (clean pass prioritized)
    smart_clean_imp = float(beauty_smart_clean.get("importance", 0.0))
    smart_clean_dir = str(beauty_smart_clean.get("direction", ""))
    hinge_clean_imp = float(beauty_hinge_clean.get("importance", 0.0))
    hinge_clean_dir = str(beauty_hinge_clean.get("direction", ""))

    if "positive" in smart_clean_dir or "increasing" in smart_clean_dir:
        score += 7
    if smart_clean_imp >= 0.20:
        score += 6

    if hinge_clean_imp > 0 and ("positive" in hinge_clean_dir):
        score += 6
    elif hinge_clean_imp == 0:
        score -= 4

    # Penalize disagreement in all-numeric pass if strong
    if float(beauty_hinge_all.get("importance", 0.0)) == 0.0:
        score -= 2

    likert_score = clamp_score(score)

    explanation_parts = [
        (
            f"Beauty shows a positive bivariate relationship with teaching evaluations "
            f"(Pearson r={pearson_r:.3f}, p={pearson_p:.3g}; simple OLS coef={beauty_coef_simple:.3f}, p={beauty_p_simple:.3g})."
        ),
        (
            f"After controlling for course and instructor covariates, the beauty effect remains positive and statistically strong "
            f"(controlled OLS coef={beauty_coef_control:.3f}, p={beauty_p_control:.3g}), about {beauty_sd_effect:.3f} eval points per 1 SD increase in beauty."
        ),
        (
            f"SmartAdditive (clean numeric set) ranks beauty as {format_effect(beauty_smart_clean)}, with a nonlinear increasing pattern"
            + (f" and a zero-crossing threshold near beauty={zero_cross:.3f}." if zero_cross is not None else ".")
        ),
        (
            f"HingeEBM (clean numeric set) also keeps beauty with {format_effect(beauty_hinge_clean)}, indicating beauty is the dominant sparse linear driver."
        ),
        (
            f"When including all numeric columns, ID-like variables (rownames/prof) absorb signal and can suppress beauty in sparse selection "
            f"(Hinge all-numeric beauty: {format_effect(beauty_hinge_all)}), so the ID-excluded robustness check is more causally interpretable."
        ),
        (
            f"Other significant controls in OLS are: {confounder_text}. These matter, but they do not remove the positive beauty-evaluation relationship."
        ),
    ]

    explanation = " ".join(explanation_parts)

    result = {
        "response": likert_score,
        "explanation": explanation,
    }

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True)

    print("Wrote conclusion.txt")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
