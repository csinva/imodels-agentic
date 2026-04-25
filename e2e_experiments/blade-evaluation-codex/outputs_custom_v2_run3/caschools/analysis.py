import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from agentic_imodels import (
    HingeEBMRegressor,
    HingeGAMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


def effective_coef_hinge_ebm(model: HingeEBMRegressor, feature_idx: int) -> float:
    """Replicate the model's displayed effective coefficient logic for one feature."""
    coefs = model.lasso_.coef_
    n_sel = len(model.selected_)
    effective: Dict[int, float] = {}

    for i in range(n_sel):
        j_orig = int(model.selected_[i])
        effective[j_orig] = float(coefs[i])

    for idx, (feat_idx, _knot, direction) in enumerate(model.hinge_info_):
        j_orig = int(model.selected_[feat_idx])
        c = float(coefs[n_sel + idx])
        if abs(c) < 1e-6:
            continue
        if direction == "pos":
            effective[j_orig] = effective.get(j_orig, 0.0) + c
        else:
            effective[j_orig] = effective.get(j_orig, 0.0) - c

    return float(effective.get(feature_idx, 0.0))


def bounded_int(x: float, lo: int = 0, hi: int = 100) -> int:
    return int(max(lo, min(hi, round(x))))


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)
    question = info["research_questions"][0]
    print("Research question:", question)

    df = pd.read_csv("caschools.csv")
    df["avg_score"] = (df["read"] + df["math"]) / 2.0
    df["str_ratio"] = df["students"] / df["teachers"]
    df["computer_per_student"] = df["computer"] / df["students"]
    df["is_kk08"] = (df["grades"] == "KK-08").astype(int)

    dv = "avg_score"
    iv = "str_ratio"
    controls: List[str] = [
        "calworks",
        "lunch",
        "income",
        "english",
        "expenditure",
        "computer_per_student",
        "is_kk08",
    ]
    features = [iv] + controls

    analysis_cols = [dv] + features
    analysis_df = df[analysis_cols].copy()

    print("\n=== Data overview ===")
    print("Rows, columns:", analysis_df.shape)
    print("Missing values:\n", analysis_df.isna().sum().to_string())
    print("\nSummary statistics:\n", analysis_df.describe().to_string())
    print("\nSkewness (distribution shape):\n", analysis_df.skew().to_string())

    corr = analysis_df.corr(numeric_only=True)
    print("\nCorrelation matrix:\n", corr.round(3).to_string())
    corr_to_dv = corr[dv].drop(dv).sort_values(key=lambda s: s.abs(), ascending=False)
    print("\nTop absolute correlations with avg_score:\n", corr_to_dv.round(3).to_string())

    pearson_r, pearson_p = stats.pearsonr(analysis_df[iv], analysis_df[dv])
    spearman_rho, spearman_p = stats.spearmanr(analysis_df[iv], analysis_df[dv])
    print(
        "\nBivariate association (student-teacher ratio vs avg score):",
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.3g}),",
        f"Spearman rho={spearman_rho:.3f} (p={spearman_p:.3g})",
    )

    print("\n=== Classical tests (statsmodels OLS) ===")
    X_biv = sm.add_constant(analysis_df[[iv]])
    ols_biv = sm.OLS(analysis_df[dv], X_biv).fit()
    print("\nBivariate OLS summary:")
    print(ols_biv.summary())

    X_ctrl = sm.add_constant(analysis_df[[iv] + controls])
    ols_ctrl = sm.OLS(analysis_df[dv], X_ctrl).fit()
    print("\nControlled OLS summary:")
    print(ols_ctrl.summary())

    X = analysis_df[features]
    y = analysis_df[dv]

    print("\n=== Interpretable models (agentic_imodels) ===")
    smart = SmartAdditiveRegressor()
    smart.fit(X, y)
    print("\n--- SmartAdditiveRegressor ---")
    print(smart)

    hinge_gam = HingeGAMRegressor()
    hinge_gam.fit(X, y)
    print("\n--- HingeGAMRegressor ---")
    print(hinge_gam)

    hinge_ebm = HingeEBMRegressor()
    hinge_ebm.fit(X, y)
    print("\n--- HingeEBMRegressor ---")
    print(hinge_ebm)

    sparse_ols = WinsorizedSparseOLSRegressor(max_features=8)
    sparse_ols.fit(X, y)
    print("\n--- WinsorizedSparseOLSRegressor ---")
    print(sparse_ols)

    smart_iv_importance = float(smart.feature_importances_[0])
    smart_total_importance = float(np.sum(smart.feature_importances_))
    smart_iv_rel_importance = (
        smart_iv_importance / smart_total_importance if smart_total_importance > 0 else 0.0
    )
    smart_iv_slope = float(smart.linear_approx_.get(0, (0.0, 0.0, 0.0))[0])

    hinge_iv_importance = float(hinge_gam.feature_importances_[0])
    hinge_iv_inactive = hinge_iv_importance < 1e-9
    hinge_ebm_iv_coef = effective_coef_hinge_ebm(hinge_ebm, 0)
    sparse_ols_iv_coef = (
        float(sparse_ols.ols_coef_[list(sparse_ols.support_).index(0)])
        if 0 in set(int(i) for i in sparse_ols.support_)
        else 0.0
    )

    biv_beta = float(ols_biv.params[iv])
    biv_p = float(ols_biv.pvalues[iv])
    ctrl_beta = float(ols_ctrl.params[iv])
    ctrl_p = float(ols_ctrl.pvalues[iv])
    ctrl_ci_low, ctrl_ci_high = map(float, ols_ctrl.conf_int().loc[iv].tolist())

    # Calibrated to SKILL.md guidance:
    # strong robust evidence -> high score; weak/inconsistent evidence -> 15-40.
    score = 45.0

    if biv_beta < 0 and biv_p < 0.05:
        score += 15
    elif biv_p >= 0.05:
        score -= 8

    if ctrl_beta < 0 and ctrl_p < 0.05:
        score += 30
    elif ctrl_beta < 0 and ctrl_p >= 0.05:
        score -= 12
    else:
        score -= 15

    if ctrl_p >= 0.10:
        score -= 6
    if abs(ctrl_beta) < 0.25:
        score -= 5

    if smart_iv_rel_importance < 0.10:
        score -= 6
    if smart_iv_slope < 0:
        score += 2

    if hinge_iv_inactive:
        score -= 8
    if abs(hinge_ebm_iv_coef) < 0.10:
        score -= 6
    elif hinge_ebm_iv_coef < -0.10:
        score += 3

    if sparse_ols_iv_coef < -0.15:
        score += 3
    elif abs(sparse_ols_iv_coef) < 0.10:
        score -= 2

    response = bounded_int(score, 0, 100)

    explanation = (
        f"Bivariate evidence suggests a negative association (Pearson r={pearson_r:.3f}, "
        f"bivariate OLS beta={biv_beta:.3f}, p={biv_p:.3g}), but this largely disappears after "
        f"controlling for socioeconomic and resource variables (controlled OLS beta={ctrl_beta:.3f}, "
        f"95% CI [{ctrl_ci_low:.3f}, {ctrl_ci_high:.3f}], p={ctrl_p:.3g}). "
        f"In interpretable models, the student-teacher ratio effect is weak: SmartAdditive assigns "
        f"low relative importance ({smart_iv_rel_importance:.1%}) with only a small slope "
        f"({smart_iv_slope:.3f}), HingeGAM effectively zeros it out (importance={hinge_iv_importance:.4f}), "
        f"and HingeEBM's displayed coefficient is near zero ({hinge_ebm_iv_coef:.3f}). "
        f"WinsorizedSparseOLS keeps a small negative coefficient ({sparse_ols_iv_coef:.3f}), but it is much "
        f"smaller than dominant predictors like lunch/income/english. Overall, evidence for a robust "
        f"independent association is weak-to-moderate, so the answer leans 'No'."
    )

    result = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True)

    print("\n=== Final calibrated conclusion ===")
    print(json.dumps(result, indent=2))
    print("\nWrote conclusion.txt")


if __name__ == "__main__":
    main()
