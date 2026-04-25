import json
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency

from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)

warnings.filterwarnings("ignore")


def safe_logit(y: pd.Series, X: pd.DataFrame):
    """Fit Logit with basic robustness settings."""
    model = sm.Logit(y, X)
    result = model.fit(disp=0, maxiter=200)
    return result


def hinge_effective_coefs(model: HingeEBMRegressor) -> Dict[int, float]:
    """Reconstruct effective per-feature coefficients used in model.__str__."""
    coefs = model.lasso_.coef_
    n_sel = len(model.selected_)

    effective = {}
    for i in range(n_sel):
        j_orig = int(model.selected_[i])
        effective[j_orig] = float(coefs[i])

    for idx, (feat_idx, knot, direction) in enumerate(model.hinge_info_):
        j_orig = int(model.selected_[feat_idx])
        c = float(coefs[n_sel + idx])
        if abs(c) < 1e-10:
            continue
        if direction == "pos":
            effective[j_orig] = effective.get(j_orig, 0.0) + c
        else:
            effective[j_orig] = effective.get(j_orig, 0.0) - c

    return effective


def summarize_smartadditive(
    model: SmartAdditiveRegressor, feature_names: List[str]
) -> Tuple[pd.DataFrame, Dict[str, dict]]:
    importances = np.asarray(model.feature_importances_, dtype=float)
    total = float(importances.sum())

    rows = []
    shape_notes = {}
    for j, name in enumerate(feature_names):
        imp = float(importances[j])
        share = imp / total if total > 0 else 0.0

        if j in model.shape_functions_:
            thresholds, intervals = model.shape_functions_[j]
            vals = np.asarray(intervals, dtype=float)
            diffs = np.diff(vals)
            if np.all(diffs >= -1e-6):
                shape = "increasing"
            elif np.all(diffs <= 1e-6):
                shape = "decreasing"
            else:
                shape = "non-monotone"
            magnitude = float(vals.max() - vals.min())
        else:
            shape = "zero/flat"
            magnitude = 0.0

        slope = 0.0
        r2 = 0.0
        if j in model.linear_approx_:
            slope = float(model.linear_approx_[j][0])
            r2 = float(model.linear_approx_[j][2])

        rows.append(
            {
                "feature": name,
                "importance": imp,
                "importance_share": share,
                "linear_slope": slope,
                "linear_r2": r2,
                "shape": shape,
                "shape_magnitude": magnitude,
            }
        )

        shape_notes[name] = {
            "shape": shape,
            "shape_magnitude": magnitude,
            "linear_slope": slope,
            "linear_r2": r2,
            "importance": imp,
            "importance_share": share,
        }

    out = pd.DataFrame(rows).sort_values("importance", ascending=False)
    return out, shape_notes


def main():
    # Step 1: read metadata
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    print("=== Research Question ===")
    print(info["research_questions"][0])

    # Step 2: load data
    df = pd.read_csv("mortgage.csv")

    # Target and design
    dv = "accept"
    iv = "female"
    controls = [
        "black",
        "housing_expense_ratio",
        "self_employed",
        "married",
        "mortgage_credit",
        "consumer_credit",
        "bad_history",
        "PI_ratio",
        "loan_to_value",
    ]
    features = [iv] + controls

    use_cols = [dv] + features
    data = df[use_cols].copy().dropna()

    print("\n=== Data Overview ===")
    print(f"Rows (after dropna on analysis cols): {len(data)}")
    print(f"Columns used: {use_cols}")
    print("\nSummary statistics:")
    print(data.describe().T[["mean", "std", "min", "max"]].round(4))

    print("\nDistribution checks:")
    print("accept value counts:")
    print(data[dv].value_counts().sort_index())
    print("female value counts:")
    print(data[iv].value_counts().sort_index())

    corrs = data.corr(numeric_only=True)[dv].sort_values(ascending=False)
    print("\nCorrelations with accept:")
    print(corrs.round(4))

    # Step 3: classical tests
    print("\n=== Classical Statistical Tests ===")

    # Bivariate chi-square and bivariate logit
    ct = pd.crosstab(data[iv], data[dv])
    chi2, chi2_p, _, _ = chi2_contingency(ct)
    print("Bivariate contingency (female x accept):")
    print(ct)
    print(f"Chi-square p-value: {chi2_p:.6g}")

    X_biv = sm.add_constant(data[[iv]], has_constant="add")
    y = data[dv].astype(float)
    logit_biv = safe_logit(y, X_biv)

    biv_beta = float(logit_biv.params[iv])
    biv_p = float(logit_biv.pvalues[iv])
    biv_or = float(np.exp(biv_beta))
    biv_ci_low, biv_ci_high = [float(v) for v in np.exp(logit_biv.conf_int().loc[iv].values)]

    print("\nBivariate Logit (accept ~ female):")
    print(logit_biv.summary2().tables[1])
    print(
        f"female beta={biv_beta:.4f}, p={biv_p:.4g}, OR={biv_or:.4f}, "
        f"95% OR CI=[{biv_ci_low:.4f}, {biv_ci_high:.4f}]"
    )

    # Controlled logistic
    X_ctrl = sm.add_constant(data[features], has_constant="add")
    logit_ctrl = safe_logit(y, X_ctrl)

    ctrl_beta = float(logit_ctrl.params[iv])
    ctrl_p = float(logit_ctrl.pvalues[iv])
    ctrl_or = float(np.exp(ctrl_beta))
    ctrl_ci_low, ctrl_ci_high = [float(v) for v in np.exp(logit_ctrl.conf_int().loc[iv].values)]

    print("\nControlled Logit (accept ~ female + controls):")
    print(logit_ctrl.summary2().tables[1])
    print(
        f"female beta={ctrl_beta:.4f}, p={ctrl_p:.4g}, OR={ctrl_or:.4f}, "
        f"95% OR CI=[{ctrl_ci_low:.4f}, {ctrl_ci_high:.4f}]"
    )

    # Step 4: Interpretable models for shape/direction/magnitude/robustness
    X = data[features]

    print("\n=== agentic_imodels: SmartAdditiveRegressor (honest) ===")
    smart = SmartAdditiveRegressor().fit(X, y)
    print(smart)

    smart_table, smart_notes = summarize_smartadditive(smart, features)
    print("\nSmartAdditive feature summary:")
    print(
        smart_table[
            [
                "feature",
                "importance",
                "importance_share",
                "linear_slope",
                "linear_r2",
                "shape",
                "shape_magnitude",
            ]
        ].round(4)
    )

    print("\n=== agentic_imodels: WinsorizedSparseOLSRegressor (honest sparse linear) ===")
    wins = WinsorizedSparseOLSRegressor().fit(X, y)
    print(wins)

    wins_coefs = {feat: 0.0 for feat in features}
    for idx, coef in zip(wins.support_, wins.ols_coef_):
        wins_coefs[features[int(idx)]] = float(coef)

    wins_df = (
        pd.DataFrame(
            {
                "feature": list(wins_coefs.keys()),
                "coef": list(wins_coefs.values()),
                "abs_coef": np.abs(list(wins_coefs.values())),
                "selected": [feat in [features[i] for i in wins.support_] for feat in wins_coefs.keys()],
            }
        )
        .sort_values("abs_coef", ascending=False)
        .reset_index(drop=True)
    )

    print("\nWinsorizedSparseOLS feature coefficients:")
    print(wins_df.round(4))

    print("\n=== agentic_imodels: HingeEBMRegressor (high-rank, decoupled) ===")
    hinge = HingeEBMRegressor().fit(X, y)
    print(hinge)

    hinge_eff = hinge_effective_coefs(hinge)
    hinge_df = pd.DataFrame(
        {
            "feature": features,
            "effective_coef": [hinge_eff.get(i, 0.0) for i in range(len(features))],
        }
    )
    hinge_df["abs_effective_coef"] = hinge_df["effective_coef"].abs()
    hinge_df = hinge_df.sort_values("abs_effective_coef", ascending=False).reset_index(drop=True)

    print("\nHingeEBM effective coefficient approximation:")
    print(hinge_df.round(4))

    # Step 5: evidence synthesis for IV=female
    smart_female = smart_notes[iv]
    wins_female = wins_coefs[iv]
    hinge_female = float(hinge_eff.get(0, 0.0))  # feature index 0 is female

    # Lightweight calibrated scoring, then clipped to [0, 100]
    score = 50.0
    if ctrl_p < 0.05:
        score += 18 if ctrl_beta > 0 else -18
    elif ctrl_p < 0.10:
        score += 8 if ctrl_beta > 0 else -8

    if biv_p > 0.10:
        score -= 10

    # Interpretable-model direction robustness
    if wins_female > 0:
        score += 7
    elif wins_female < 0:
        score -= 7

    if hinge_female > 0:
        score += 7
    elif hinge_female < 0:
        score -= 7

    # Null evidence from honest GAM zeroing/near-zero
    if smart_female["importance_share"] < 0.01:
        score -= 9

    # Small-magnitude penalty if effects are tiny in sparse models
    if abs(wins_female) < 0.02 and abs(hinge_female) < 0.02:
        score -= 5

    response = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Bivariate evidence shows almost no gender association with approval "
        f"(chi-square p={chi2_p:.3f}; bivariate logit beta={biv_beta:.3f}, p={biv_p:.3f}, OR={biv_or:.2f}). "
        f"After controlling for credit/risk covariates, female has a positive statistically significant coefficient "
        f"(beta={ctrl_beta:.3f}, p={ctrl_p:.3f}, OR={ctrl_or:.2f}, 95% CI {ctrl_ci_low:.2f}-{ctrl_ci_high:.2f}). "
        f"Interpretable models are mixed but directionally positive for female in sparse linear and hinge-EBM views "
        f"(WinsorizedSparseOLS coef={wins_female:.3f}; HingeEBM effective coef~{hinge_female:.3f}), while SmartAdditive "
        f"assigns near-zero importance to female (importance share={smart_female['importance_share']:.3f}), giving null-evidence "
        f"from one honest model. Overall this supports a modest, control-dependent positive effect of female on approval, "
        f"not a large or dominant driver versus debt/credit features."
    )

    result = {"response": response, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f)

    print("\n=== Final Likert Decision ===")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
