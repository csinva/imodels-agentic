import json
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold, cross_val_score

from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


warnings.filterwarnings("ignore")


def effective_coefs_hinge_ebm(model: HingeEBMRegressor) -> Tuple[Dict[int, float], float]:
    """Replicate the effective linearized coefficients used by HingeEBMRegressor.__str__."""
    n_sel = len(model.selected_)
    coefs = model.lasso_.coef_
    intercept = float(model.lasso_.intercept_)

    effective_coefs: Dict[int, float] = {}
    for i in range(n_sel):
        j_orig = int(model.selected_[i])
        effective_coefs[j_orig] = float(coefs[i])

    for idx, (feat_idx, knot, direction) in enumerate(model.hinge_info_):
        j_orig = int(model.selected_[feat_idx])
        c = float(coefs[n_sel + idx])
        if abs(c) < 1e-6:
            continue
        if direction == "pos":
            effective_coefs[j_orig] = effective_coefs.get(j_orig, 0.0) + c
            intercept -= c * float(knot)
        else:
            effective_coefs[j_orig] = effective_coefs.get(j_orig, 0.0) - c
            intercept += c * float(knot)

    return effective_coefs, intercept


def monotonicity_label(intervals: np.ndarray, tol: float = 1e-6) -> str:
    diffs = np.diff(intervals)
    if np.all(diffs >= -tol):
        return "monotone increasing"
    if np.all(diffs <= tol):
        return "monotone decreasing"
    return "non-monotone"


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_q = info["research_questions"][0]
    print("=== Research Question ===")
    print(research_q)

    df = pd.read_csv("teachingratings.csv")
    print("\n=== Data Overview ===")
    print(f"Rows: {len(df)}, Columns: {df.shape[1]}")
    print("Columns:", list(df.columns))

    print("\n=== Missingness ===")
    print(df.isna().sum())

    numeric_cols_all = df.select_dtypes(include=[np.number]).columns.tolist()
    print("\n=== Numeric Summary Statistics ===")
    print(df[numeric_cols_all].describe().T)

    print("\n=== Outcome Distribution (eval) ===")
    print(df["eval"].describe())
    print("Skew(eval):", float(df["eval"].skew()))
    print("Kurtosis(eval):", float(df["eval"].kurtosis()))

    print("\n=== Correlations with eval (numeric features) ===")
    corr_with_eval = df[numeric_cols_all].corr()["eval"].sort_values(ascending=False)
    print(corr_with_eval)

    cat_cols = ["minority", "gender", "credits", "division", "native", "tenure"]
    print("\n=== Mean eval by categorical controls ===")
    for c in cat_cols:
        print(f"\n{c}:")
        print(df.groupby(c)["eval"].agg(["mean", "count"]).sort_index())

    # Bivariate tests
    r, r_p = stats.pearsonr(df["beauty"], df["eval"])
    print("\n=== Bivariate Test: beauty vs eval ===")
    print(f"Pearson r = {r:.4f}, p = {r_p:.4g}")

    bivar_ols = smf.ols("eval ~ beauty", data=df).fit(cov_type="HC3")
    print("\nBivariate OLS (HC3 robust SE):")
    print(bivar_ols.summary())

    # Controlled OLS with relevant controls
    formula_controls = (
        "eval ~ beauty + age + students + allstudents + "
        "C(minority) + C(gender) + C(credits) + C(division) + C(native) + C(tenure)"
    )
    controlled_ols = smf.ols(formula_controls, data=df).fit(cov_type="HC3")
    print("\n=== Controlled OLS (HC3 robust SE) ===")
    print(controlled_ols.summary())

    beauty_beta = float(controlled_ols.params["beauty"])
    beauty_p = float(controlled_ols.pvalues["beauty"])
    beauty_ci_lo, beauty_ci_hi = controlled_ols.conf_int().loc["beauty"].tolist()

    # Prepare encoded features for agentic_imodels (no categorical handling built-in)
    numeric_model_cols = ["beauty", "age", "students", "allstudents"]
    X_df = pd.get_dummies(df[numeric_model_cols + cat_cols], drop_first=True, dtype=float)
    y = df["eval"].astype(float).to_numpy()
    X = X_df.to_numpy(dtype=float)
    feature_names = list(X_df.columns)
    beauty_idx = feature_names.index("beauty")

    print("\n=== Modeling Matrix for agentic_imodels ===")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print("Feature names:")
    print(feature_names)

    models = {
        "SmartAdditiveRegressor": SmartAdditiveRegressor(),
        "HingeEBMRegressor": HingeEBMRegressor(),
        "WinsorizedSparseOLSRegressor": WinsorizedSparseOLSRegressor(),
    }

    fitted = {}
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    print("\n=== agentic_imodels Fits and Printed Forms ===")
    for name, model in models.items():
        model.fit(X, y)
        fitted[name] = model
        print(f"\n--- {name} ---")
        print(model)

        r2_cv = cross_val_score(model, X, y, cv=cv, scoring="r2")
        mse_cv = -cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
        rmse_cv = np.sqrt(mse_cv)
        print(f"CV R^2 mean={np.mean(r2_cv):.4f}, sd={np.std(r2_cv):.4f}")
        print(f"CV RMSE mean={np.mean(rmse_cv):.4f}, sd={np.std(rmse_cv):.4f}")

    # Extract beauty effect from each interpretable model
    smart = fitted["SmartAdditiveRegressor"]
    smart_total_imp = float(np.sum(smart.feature_importances_))
    smart_beauty_imp = float(smart.feature_importances_[beauty_idx])
    smart_beauty_imp_share = smart_beauty_imp / smart_total_imp if smart_total_imp > 1e-12 else 0.0
    smart_rank = int(np.where(np.argsort(-smart.feature_importances_) == beauty_idx)[0][0] + 1)

    if beauty_idx in smart.linear_approx_:
        s_slope, _, s_r2 = smart.linear_approx_[beauty_idx]
        if s_r2 > 0.70:
            smart_shape = "approximately linear"
            smart_direction = "positive" if s_slope > 0 else "negative" if s_slope < 0 else "zero"
        else:
            thresholds, intervals = smart.shape_functions_[beauty_idx]
            smart_shape = f"piecewise ({monotonicity_label(np.array(intervals))})"
            smart_direction = monotonicity_label(np.array(intervals))
    else:
        s_slope, s_r2 = 0.0, 0.0
        smart_shape = "excluded"
        smart_direction = "zero"

    hinge = fitted["HingeEBMRegressor"]
    hinge_eff, _ = effective_coefs_hinge_ebm(hinge)
    hinge_beauty_coef = float(hinge_eff.get(beauty_idx, 0.0))
    hinge_active = abs(hinge_beauty_coef) > 1e-6
    hinge_active_sorted = sorted(
        [(j, c) for j, c in hinge_eff.items() if abs(c) > 1e-6],
        key=lambda t: abs(t[1]),
        reverse=True,
    )
    hinge_rank = (
        next((i + 1 for i, (j, _) in enumerate(hinge_active_sorted) if j == beauty_idx), None)
        if hinge_active_sorted
        else None
    )

    wins = fitted["WinsorizedSparseOLSRegressor"]
    support = list(wins.support_)
    wins_beauty_included = beauty_idx in support
    wins_coef_map = {int(j): float(c) for j, c in zip(wins.support_, wins.ols_coef_)}
    wins_beauty_coef = float(wins_coef_map.get(beauty_idx, 0.0))
    wins_rank = (
        next(
            (i + 1 for i, (j, _) in enumerate(sorted(wins_coef_map.items(), key=lambda t: abs(t[1]), reverse=True)) if j == beauty_idx),
            None,
        )
        if wins_coef_map
        else None
    )

    print("\n=== Interpretable Model Synthesis for beauty ===")
    print(
        f"SmartAdditive: direction={smart_direction}, shape={smart_shape}, "
        f"slope={s_slope:.4f}, approx_R2={s_r2:.3f}, importance_share={smart_beauty_imp_share:.3f}, rank={smart_rank}/{len(feature_names)}"
    )
    print(
        f"HingeEBM: active={hinge_active}, effective_coef={hinge_beauty_coef:.4f}, "
        f"rank={hinge_rank if hinge_rank is not None else 'excluded'}"
    )
    print(
        f"WinsorizedSparseOLS: included={wins_beauty_included}, coef={wins_beauty_coef:.4f}, "
        f"rank={wins_rank if wins_rank is not None else 'excluded'}"
    )

    # Calibrated Likert score (0-100) using significance + robustness + model ranking/zeroing
    score = 50

    # Controlled OLS significance/magnitude anchor
    if beauty_p < 0.001:
        score += 20
    elif beauty_p < 0.01:
        score += 15
    elif beauty_p < 0.05:
        score += 10
    elif beauty_p < 0.10:
        score += 5
    else:
        score -= 15

    if beauty_beta > 0:
        score += 5
    elif beauty_beta < 0:
        score -= 5

    # Bivariate corroboration
    if r_p < 0.05 and np.sign(r) == np.sign(beauty_beta):
        score += 4

    # SmartAdditive importance + direction
    if smart_beauty_imp_share >= 0.15:
        score += 10
    elif smart_beauty_imp_share >= 0.07:
        score += 6
    elif smart_beauty_imp_share >= 0.03:
        score += 2
    else:
        score -= 5

    if s_slope != 0 and np.sign(s_slope) == np.sign(beauty_beta):
        score += 5
    elif s_slope != 0 and np.sign(s_slope) != np.sign(beauty_beta):
        score -= 5

    # Hinge zeroing / sign robustness
    if hinge_active:
        if np.sign(hinge_beauty_coef) == np.sign(beauty_beta):
            score += 8
        else:
            score -= 8
    else:
        score -= 10

    # Lasso-zeroing evidence from WinsorizedSparseOLS
    if wins_beauty_included:
        if np.sign(wins_beauty_coef) == np.sign(beauty_beta):
            score += 10
        else:
            score -= 10
    else:
        score -= 15

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Research question: {research_q} Controlled OLS with HC3 robust SE gives beauty beta={beauty_beta:.3f} "
        f"(95% CI [{beauty_ci_lo:.3f}, {beauty_ci_hi:.3f}], p={beauty_p:.4g}), while bivariate association is "
        f"r={r:.3f} (p={r_p:.4g}). In interpretable models, SmartAdditive estimates a {smart_direction} beauty effect "
        f"with {smart_shape} form (slope={s_slope:.3f}, importance share={smart_beauty_imp_share:.3f}, rank={smart_rank}), "
        f"HingeEBM {'retains' if hinge_active else 'drops'} beauty (effective coef={hinge_beauty_coef:.3f}), and "
        f"WinsorizedSparseOLS {'retains' if wins_beauty_included else 'drops'} beauty "
        f"(coef={wins_beauty_coef:.3f}). Overall evidence is "
        f"{'strong and robust' if score >= 75 else 'moderate' if score >= 40 else 'weak/inconsistent'}, "
        f"so the calibrated Likert response is {score}."
    )

    out = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(out, f)

    print("\n=== Final Conclusion JSON ===")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
