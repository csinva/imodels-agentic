import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import chi2, chi2_contingency, pointbiserialr

from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def hinge_effective_coefficients(model):
    coefs = np.asarray(model.lasso_.coef_, dtype=float)
    intercept = float(model.lasso_.intercept_)
    n_sel = len(model.selected_)

    eff = {int(model.selected_[i]): float(coefs[i]) for i in range(n_sel)}
    eff_intercept = intercept

    for idx, (feat_idx, knot, direction) in enumerate(model.hinge_info_):
        j_orig = int(model.selected_[feat_idx])
        c = float(coefs[n_sel + idx])
        if abs(c) < 1e-6:
            continue
        if direction == "pos":
            eff[j_orig] = eff.get(j_orig, 0.0) + c
            eff_intercept -= c * float(knot)
        else:
            eff[j_orig] = eff.get(j_orig, 0.0) - c
            eff_intercept += c * float(knot)
    return eff, eff_intercept


def clamp(v, lo=0, hi=100):
    return max(lo, min(hi, int(round(v))))


def main():
    warnings.filterwarnings("ignore")

    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info["research_questions"][0]
    print("Research question:")
    print(question)
    print("\nLoading data from boxes.csv ...")

    df = pd.read_csv("boxes.csv")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print(f"Columns: {list(df.columns)}")

    # Outcome for the research question: majority-choice reliance.
    df["majority_choice"] = (df["y"] == 2).astype(int)

    print("\n=== Basic summaries ===")
    print(df.describe(include="all").to_string())

    print("\nOutcome distribution y (1=unchosen,2=majority,3=minority):")
    print(df["y"].value_counts().sort_index().to_string())

    print("\nMajority-choice rate overall:")
    print(df["majority_choice"].mean())

    print("\nMajority-choice rate by age:")
    print(df.groupby("age")["majority_choice"].mean().to_string())

    print("\nMajority-choice rate by culture:")
    print(df.groupby("culture")["majority_choice"].mean().to_string())

    print("\nCorrelation matrix (numeric columns):")
    corr = df[["majority_choice", "age", "gender", "majority_first", "culture"]].corr(numeric_only=True)
    print(corr.to_string())

    # Bivariate tests
    pb = pointbiserialr(df["majority_choice"], df["age"])
    print("\n=== Bivariate tests ===")
    print(f"Point-biserial correlation (majority_choice vs age): r={pb.statistic:.4f}, p={pb.pvalue:.4g}")

    ct = pd.crosstab(df["majority_choice"], df["culture"])
    chi2_stat, chi2_p, chi2_df, _ = chi2_contingency(ct)
    print(f"Chi-square (majority_choice x culture): chi2={chi2_stat:.3f}, df={chi2_df}, p={chi2_p:.4g}")

    # Classical formal tests: logistic regression (binary majority-choice outcome)
    print("\n=== Logistic models (statsmodels) ===")
    model_biv = smf.logit("majority_choice ~ age", data=df).fit(disp=0)
    model_ctrl = smf.logit(
        "majority_choice ~ age + gender + majority_first + C(culture)", data=df
    ).fit(disp=0)
    model_inter = smf.logit(
        "majority_choice ~ age * C(culture) + gender + majority_first", data=df
    ).fit(disp=0, maxiter=200)

    print("\nBivariate logit: majority_choice ~ age")
    print(model_biv.summary())

    print("\nControlled logit: majority_choice ~ age + gender + majority_first + C(culture)")
    print(model_ctrl.summary())

    print("\nInteraction logit: add age x culture interactions")
    print(model_inter.summary())

    # LR test for age-culture interaction block
    lr_stat = 2.0 * (model_inter.llf - model_ctrl.llf)
    df_diff = int(model_inter.df_model - model_ctrl.df_model)
    lr_p = safe_float(chi2.sf(lr_stat, df_diff)) if df_diff > 0 else np.nan
    print(
        f"\nLikelihood-ratio test for age*culture interaction block: "
        f"LR={lr_stat:.4f}, df={df_diff}, p={lr_p:.4g}"
    )

    # Build feature matrix for agentic_imodels (one-hot culture)
    X = pd.concat(
        [
            df[["age", "gender", "majority_first"]].copy(),
            pd.get_dummies(df["culture"].astype(int), prefix="culture", drop_first=True),
        ],
        axis=1,
    ).astype(float)
    y = df["majority_choice"].astype(float).values
    feature_names = list(X.columns)

    print("\n=== agentic_imodels feature mapping ===")
    for i, name in enumerate(feature_names):
        print(f"x{i} = {name}")

    print("\n=== Interpretable model: SmartAdditiveRegressor (honest) ===")
    smart = SmartAdditiveRegressor().fit(X, y)
    print(smart)

    print("\n=== Interpretable model: WinsorizedSparseOLSRegressor (honest sparse linear, Lasso-based) ===")
    winsor = WinsorizedSparseOLSRegressor().fit(X, y)
    print(winsor)

    print("\n=== Interpretable model: HingeEBMRegressor (high-rank decoupled) ===")
    hinge_ebm = HingeEBMRegressor().fit(X, y)
    print(hinge_ebm)

    # Quantify evidence from interpretable models
    age_idx = feature_names.index("age")

    smart_imps = np.asarray(smart.feature_importances_, dtype=float)
    smart_total_imp = float(np.sum(smart_imps)) if np.isfinite(np.sum(smart_imps)) else 0.0
    smart_age_imp = float(smart_imps[age_idx])
    smart_age_imp_frac = smart_age_imp / smart_total_imp if smart_total_imp > 1e-12 else 0.0
    smart_rank_order = list(np.argsort(-smart_imps))
    smart_age_rank = int(smart_rank_order.index(age_idx) + 1)

    age_shape = smart.shape_functions_.get(age_idx)
    age_shape_range = 0.0
    age_nonmonotone = False
    if age_shape is not None:
        _, age_intervals = age_shape
        if len(age_intervals) > 0:
            age_shape_range = float(np.max(age_intervals) - np.min(age_intervals))
            diffs = np.diff(np.asarray(age_intervals, dtype=float))
            if len(diffs) > 0:
                mono_inc = np.all(diffs >= -1e-8)
                mono_dec = np.all(diffs <= 1e-8)
                age_nonmonotone = not (mono_inc or mono_dec)

    winsor_selected_features = {feature_names[i] for i in winsor.support_}
    age_selected_winsor = "age" in winsor_selected_features

    heff, _ = hinge_effective_coefficients(hinge_ebm)
    hinge_age_coef = float(heff.get(age_idx, 0.0))

    print("\n=== Extracted evidence summary ===")
    age_coef_biv = safe_float(model_biv.params.get("age", np.nan))
    age_p_biv = safe_float(model_biv.pvalues.get("age", np.nan))
    age_coef_ctrl = safe_float(model_ctrl.params.get("age", np.nan))
    age_p_ctrl = safe_float(model_ctrl.pvalues.get("age", np.nan))

    print(f"Bivariate logit age coef={age_coef_biv:.4f}, p={age_p_biv:.4g}")
    print(f"Controlled logit age coef={age_coef_ctrl:.4f}, p={age_p_ctrl:.4g}")
    print(f"Age*culture LR-test p={lr_p:.4g}")
    print(
        f"SmartAdditive: age importance={smart_age_imp:.4f} "
        f"(fraction={smart_age_imp_frac:.3f}, rank={smart_age_rank}/{len(feature_names)}), "
        f"shape_range={age_shape_range:.4f}, nonmonotone={age_nonmonotone}"
    )
    print(f"WinsorizedSparseOLS selected age={age_selected_winsor}")
    print(f"HingeEBM effective age coefficient={hinge_age_coef:.4f}")

    # Likert score calibration based on SKILL guidance
    score = 50.0

    # Classical significance anchors
    if np.isfinite(age_p_ctrl):
        if age_p_ctrl < 0.01:
            score += 20
        elif age_p_ctrl < 0.05:
            score += 15
        elif age_p_ctrl < 0.10:
            score += 8
        else:
            score -= 15

    if np.isfinite(lr_p):
        if lr_p < 0.05:
            score += 12
        else:
            score -= 10

    # Interpretable model corroboration / null evidence
    if age_selected_winsor:
        score += 10
    else:
        score -= 10

    if abs(hinge_age_coef) < 0.02:
        score -= 6
    else:
        score += 4

    if smart_age_rank <= 2 and smart_age_imp_frac >= 0.15:
        score += 8
    elif smart_age_rank >= 6 or smart_age_imp_frac < 0.05:
        score -= 6

    if age_nonmonotone and age_shape_range > 0.08:
        score += 5

    response = clamp(score, 0, 100)

    explanation = (
        "The evidence for an age-driven increase in majority reliance across cultural contexts is weak. "
        f"A bivariate logistic model finds age near zero (coef={age_coef_biv:.3f}, p={age_p_biv:.3g}), and with controls "
        f"(gender, majority_first, culture fixed effects) age remains non-significant (coef={age_coef_ctrl:.3f}, p={age_p_ctrl:.3g}). "
        f"Age-by-culture interactions are not jointly significant (LR p={lr_p:.3g}), so developmental differences by culture are not robustly supported. "
        "Interpretable models are mixed: SmartAdditive ranks age relatively high and shows a non-monotonic shape (higher at youngest/oldest ages, lower mid-range), "
        f"but WinsorizedSparseOLS excludes age entirely (Lasso zeroing null evidence), and HingeEBM gives only a very small effective age slope ({hinge_age_coef:.3f}). "
        "By contrast, majority_first and some culture indicators are consistently stronger predictors. "
        "Overall this supports at most a weak/inconsistent age relationship rather than a robust cross-cultural developmental trend."
    )

    payload = {"response": int(response), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True)

    print("\nWrote conclusion.txt:")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
