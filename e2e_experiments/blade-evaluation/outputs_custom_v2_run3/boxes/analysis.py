import json
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor


def _to_builtin_effects(effects: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Convert numpy scalar values in effects dict to Python built-ins for safe JSON printing."""
    cleaned: Dict[str, Dict[str, Any]] = {}
    for k, v in effects.items():
        cleaned[k] = {}
        for kk, vv in v.items():
            if isinstance(vv, (np.floating, np.integer)):
                cleaned[k][kk] = vv.item()
            else:
                cleaned[k][kk] = vv
    return cleaned


def _top_effects(effects: Dict[str, Dict[str, Any]], n: int = 5) -> List[Tuple[str, float, str, int]]:
    rows = []
    for name, vals in effects.items():
        imp = float(vals.get("importance", 0.0) or 0.0)
        rank = int(vals.get("rank", 0) or 0)
        direction = str(vals.get("direction", "unknown"))
        if imp > 0:
            rows.append((name, imp, direction, rank))
    rows.sort(key=lambda x: (-x[1], x[0]))
    return rows[:n]


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["Unknown research question"])[0]
    df = pd.read_csv("boxes.csv")

    # Research mapping for this dataset/question
    dv_original = "y"
    iv = "age"
    # Majority reliance: 1 if child selected majority option, else 0
    df["majority_choice"] = (df[dv_original] == 2).astype(int)
    dv = "majority_choice"

    print("=== Step 1: Understand Question and Explore ===")
    print("Research question:", question)
    print(f"Dependent variable (DV): {dv} (derived from y==2)")
    print(f"Independent variable (IV): {iv}")
    print("\nDataset shape:", df.shape)
    print("\nSummary statistics:")
    print(df[["y", "majority_choice", "gender", "age", "majority_first", "culture"]].describe().T)

    print("\nOutcome distributions:")
    print("y value counts:")
    print(df["y"].value_counts().sort_index())
    print("\nmajority_choice value counts:")
    print(df["majority_choice"].value_counts().sort_index())

    numeric_cols = ["y", "majority_choice", "gender", "age", "majority_first", "culture"]
    print("\nCorrelation matrix:")
    print(df[numeric_cols].corr(numeric_only=True).round(4))

    r_age_majority, p_age_majority = stats.pearsonr(df["age"], df[dv])
    rho_age_y, p_age_y = stats.spearmanr(df["age"], df["y"])
    print(
        f"\nBivariate age-majority correlation: r={r_age_majority:.4f}, p={p_age_majority:.4g}"
    )
    print(f"Bivariate age-y Spearman: rho={rho_age_y:.4f}, p={p_age_y:.4g}")

    print("\n=== Step 2: Controlled Statistical Tests ===")
    base_features = ["age", "gender", "majority_first", "culture"]
    X_main = pd.get_dummies(
        df[base_features], columns=["culture"], drop_first=True, dtype=float
    )

    X_main_const = sm.add_constant(X_main)
    logit_main = sm.Logit(df[dv], X_main_const).fit(disp=False)
    print("Controlled logistic regression (DV=majority_choice):")
    print(logit_main.summary())

    age_coef_main = float(logit_main.params.get("age", np.nan))
    age_p_main = float(logit_main.pvalues.get("age", np.nan))

    # Age-by-culture interactions to directly assess "across cultural contexts"
    X_inter = X_main.copy()
    culture_dummy_cols = [c for c in X_inter.columns if c.startswith("culture_")]
    for col in culture_dummy_cols:
        X_inter[f"age_x_{col}"] = X_inter["age"] * X_inter[col]

    X_inter_const = sm.add_constant(X_inter)
    logit_inter = sm.Logit(df[dv], X_inter_const).fit(disp=False, maxiter=200)
    print("\nLogistic regression with age-by-culture interactions:")
    print(logit_inter.summary())

    lr_stat = 2.0 * (logit_inter.llf - logit_main.llf)
    lr_df = int(logit_inter.df_model - logit_main.df_model)
    lr_p = float(stats.chi2.sf(lr_stat, lr_df)) if lr_df > 0 else np.nan
    print(
        f"\nLikelihood-ratio test (interaction model vs main): LR={lr_stat:.4f}, df={lr_df}, p={lr_p:.4g}"
    )

    print("\n=== Step 3: Interpretable Models ===")
    X_interp = X_inter.copy()
    y_interp = df[dv].astype(float)

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y_interp)
    print("SmartAdditiveRegressor:")
    print(smart)
    smart_effects = _to_builtin_effects(smart.feature_effects())
    print("\nSmartAdditive feature_effects:")
    print(json.dumps(smart_effects, indent=2))

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_interp, y_interp)
    print("\nHingeEBMRegressor:")
    print(hinge)
    hinge_effects = _to_builtin_effects(hinge.feature_effects())
    print("\nHingeEBM feature_effects:")
    print(json.dumps(hinge_effects, indent=2))

    # Pull key interpretation values for age and confounders
    smart_age = smart_effects.get("age", {"direction": "zero", "importance": 0.0, "rank": 0})
    hinge_age = hinge_effects.get("age", {"direction": "zero", "importance": 0.0, "rank": 0})

    smart_top = _top_effects(smart_effects, n=3)
    hinge_top = _top_effects(hinge_effects, n=3)

    majority_first_p = float(logit_main.pvalues.get("majority_first", np.nan))
    majority_first_coef = float(logit_main.params.get("majority_first", np.nan))
    gender_p = float(logit_main.pvalues.get("gender", np.nan))
    gender_coef = float(logit_main.params.get("gender", np.nan))

    # Step 4: Likert score synthesis (0-100)
    score = 50

    # Bivariate evidence for age effect on majority choice
    if p_age_majority >= 0.05:
        score -= 10
    else:
        score += 8 if r_age_majority > 0 else -8

    # Controlled age effect
    if age_p_main < 0.05:
        score += 20 if age_coef_main > 0 else -15
    elif age_p_main < 0.10:
        score += 8 if age_coef_main > 0 else -8
    else:
        score -= 15

    # Cross-cultural interaction evidence
    if np.isfinite(lr_p) and lr_p < 0.05:
        score += 8
    else:
        score -= 5

    # Interpretable models: age importance and direction
    smart_age_imp = float(smart_age.get("importance", 0.0) or 0.0)
    smart_age_dir = str(smart_age.get("direction", "zero"))
    if smart_age_imp >= 0.10 and (
        "increasing" in smart_age_dir.lower() or smart_age_dir.lower() == "positive"
    ):
        score += 8
    elif smart_age_imp < 0.01:
        score -= 6

    hinge_age_imp = float(hinge_age.get("importance", 0.0) or 0.0)
    hinge_age_dir = str(hinge_age.get("direction", "zero"))
    if hinge_age_imp >= 0.05:
        if hinge_age_dir.lower() == "positive":
            score += 8
        elif hinge_age_dir.lower() == "negative":
            score -= 8
    else:
        score -= 8

    # Strong confounder dominance lowers certainty about age-driven development
    if majority_first_p < 0.05 and abs(majority_first_coef) > abs(age_coef_main):
        score -= 2

    score = int(max(0, min(100, round(score))))

    smart_top_text = "; ".join(
        [f"{n} (imp={imp:.3f}, dir={d}, rank={r})" for n, imp, d, r in smart_top]
    ) or "none"
    hinge_top_text = "; ".join(
        [f"{n} (imp={imp:.3f}, dir={d}, rank={r})" for n, imp, d, r in hinge_top]
    ) or "none"

    explanation = (
        f"Age shows little robust evidence of increasing majority reliance across cultures. "
        f"Bivariate age-majority association is near zero (r={r_age_majority:.3f}, p={p_age_majority:.3g}). "
        f"In controlled logistic regression, age is not significant (coef={age_coef_main:.3f}, p={age_p_main:.3g}) after controlling for gender, majority_first, and culture. "
        f"Age-by-culture interactions do not significantly improve fit (LR p={lr_p:.3g}), so cross-cultural age slope differences are weak overall. "
        f"SmartAdditive gives age a {smart_age_imp:.1%} importance (rank={smart_age.get('rank', 0)}, direction={smart_age_dir}), suggesting some nonlinear age pattern, "
        f"but HingeEBM assigns age {hinge_age_imp:.1%} importance (rank={hinge_age.get('rank', 0)}, direction={hinge_age_dir}), effectively zeroing it out. "
        f"The strongest and most consistent predictor is majority_first (logit coef={majority_first_coef:.3f}, p={majority_first_p:.3g}); gender also has a smaller positive effect (coef={gender_coef:.3f}, p={gender_p:.3g}). "
        f"Top SmartAdditive effects: {smart_top_text}. Top HingeEBM effects: {hinge_top_text}. "
        f"Overall, evidence for an age-driven developmental increase in majority preference is weak/inconsistent once confounders are included."
    )

    out = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(out, f)

    print("\n=== Step 4: Conclusion ===")
    print(json.dumps(out, indent=2))
    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
