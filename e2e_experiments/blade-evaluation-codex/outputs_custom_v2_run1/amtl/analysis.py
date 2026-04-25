import json
import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score

from agentic_imodels import (
    HingeEBMRegressor,
    SmartAdditiveRegressor,
    WinsorizedSparseOLSRegressor,
)


warnings.filterwarnings("ignore")


def hinge_ebm_effective_linear_coefs(model):
    """Reconstruct effective per-feature linear coefficients used in model.__str__."""
    coefs = model.lasso_.coef_
    n_sel = len(model.selected_)
    effective = {}
    intercept = float(model.lasso_.intercept_)

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
            intercept -= c * float(knot)
        else:
            effective[j_orig] = effective.get(j_orig, 0.0) - c
            intercept += c * float(knot)

    return effective, intercept


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    rq = info["research_questions"][0]
    print("Research question:")
    print(rq)

    df = pd.read_csv("amtl.csv")
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]

    print("\n=== Data Overview ===")
    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print("\nMissing values:")
    print(df.isna().sum())

    print("\n=== Numeric Summary ===")
    print(df[["num_amtl", "sockets", "amtl_rate", "age", "stdev_age", "prob_male"]].describe())

    print("\n=== Distributions ===")
    print("Genus counts:")
    print(df["genus"].value_counts())
    print("\nTooth class counts:")
    print(df["tooth_class"].value_counts())

    print("\nAMTL rate by genus:")
    print(df.groupby("genus")["amtl_rate"].agg(["mean", "std", "median", "count"]).sort_values("mean", ascending=False))

    print("\nAMTL rate by human status:")
    print(df.groupby("is_human")["amtl_rate"].agg(["mean", "std", "median", "count"]))

    corr_cols = ["num_amtl", "sockets", "amtl_rate", "age", "stdev_age", "prob_male", "is_human"]
    print("\n=== Correlations (Pearson) ===")
    print(df[corr_cols].corr())

    print("\n=== Bivariate Test (Human vs Non-human) ===")
    human_rates = df.loc[df["is_human"] == 1, "amtl_rate"]
    nonhuman_rates = df.loc[df["is_human"] == 0, "amtl_rate"]
    mw_stat, mw_p = stats.mannwhitneyu(human_rates, nonhuman_rates, alternative="two-sided")
    print(f"Mann-Whitney U statistic={mw_stat:.3f}, p-value={mw_p:.3e}")

    X_biv = sm.add_constant(df[["is_human"]].astype(float))
    glm_biv = sm.GLM(
        df["amtl_rate"],
        X_biv,
        family=sm.families.Binomial(),
        var_weights=df["sockets"],
    ).fit()
    biv_beta = float(glm_biv.params["is_human"])
    biv_p = float(glm_biv.pvalues["is_human"])
    biv_or = float(np.exp(biv_beta))
    print(f"Bivariate binomial GLM: beta_human={biv_beta:.4f}, OR={biv_or:.3f}, p={biv_p:.3e}")

    print("\n=== Controlled Binomial GLM (age, sex, tooth class controls) ===")
    X_ctrl = pd.get_dummies(
        df[["is_human", "age", "prob_male", "tooth_class"]],
        columns=["tooth_class"],
        drop_first=True,
    ).astype(float)
    X_ctrl = sm.add_constant(X_ctrl)

    glm_ctrl = sm.GLM(
        df["amtl_rate"],
        X_ctrl,
        family=sm.families.Binomial(),
        var_weights=df["sockets"],
    ).fit()
    print(glm_ctrl.summary())

    beta_human = float(glm_ctrl.params["is_human"])
    p_human = float(glm_ctrl.pvalues["is_human"])
    ci_low, ci_high = glm_ctrl.conf_int().loc["is_human"].values
    or_human = float(np.exp(beta_human))
    or_ci_low, or_ci_high = float(np.exp(ci_low)), float(np.exp(ci_high))

    print(
        f"Controlled human effect: beta={beta_human:.4f}, OR={or_human:.3f}, "
        f"95% CI OR=[{or_ci_low:.3f}, {or_ci_high:.3f}], p={p_human:.3e}"
    )

    print("\n=== Interpretable Models (agentic_imodels) ===")
    feature_cols = X_ctrl.columns.tolist()
    X_model = X_ctrl.astype(float)
    y_model = df["amtl_rate"].values

    models = [
        SmartAdditiveRegressor(),
        HingeEBMRegressor(),
        WinsorizedSparseOLSRegressor(),
    ]

    human_effect_signals = {}
    model_fit_stats = {}

    for model in models:
        name = model.__class__.__name__
        print(f"\n=== {name} ===")
        model.fit(X_model, y_model)
        preds = model.predict(X_model)
        rmse = float(math.sqrt(mean_squared_error(y_model, preds)))
        r2 = float(r2_score(y_model, preds))
        model_fit_stats[name] = {"rmse": rmse, "r2": r2}
        print(f"Training RMSE={rmse:.4f}, R^2={r2:.4f}")
        print(model)

        if name == "SmartAdditiveRegressor":
            # Feature index for is_human from X_model columns
            idx = feature_cols.index("is_human")
            slope, offset, lin_r2 = model.linear_approx_.get(idx, (0.0, 0.0, 0.0))
            importance = float(model.feature_importances_[idx]) if hasattr(model, "feature_importances_") else 0.0
            human_effect_signals[name] = {
                "type": "slope",
                "value": float(slope),
                "importance": importance,
                "linear_r2": float(lin_r2),
                "included": abs(float(slope)) > 1e-6 or importance > 1e-6,
            }
        elif name == "HingeEBMRegressor":
            idx = feature_cols.index("is_human")
            eff_coefs, _ = hinge_ebm_effective_linear_coefs(model)
            coef = float(eff_coefs.get(idx, 0.0))
            human_effect_signals[name] = {
                "type": "effective_linear_coef",
                "value": coef,
                "included": abs(coef) > 1e-6,
            }
        elif name == "WinsorizedSparseOLSRegressor":
            idx = feature_cols.index("is_human")
            if idx in model.support_:
                pos = model.support_.index(idx)
                coef = float(model.ols_coef_[pos])
                included = True
            else:
                coef = 0.0
                included = False
            human_effect_signals[name] = {
                "type": "sparse_coef",
                "value": coef,
                "included": included,
                "support": [feature_cols[j] for j in model.support_],
            }

    print("\n=== Model-derived Human-effect Signals ===")
    print(json.dumps(human_effect_signals, indent=2))

    positive_support = 0
    zeroed_support = 0
    negative_support = 0

    for v in human_effect_signals.values():
        if not v.get("included", False):
            zeroed_support += 1
            continue
        val = float(v.get("value", 0.0))
        if val > 0:
            positive_support += 1
        elif val < 0:
            negative_support += 1

    score = 30

    # Controlled significance and effect size
    if p_human < 1e-6:
        score += 30
    elif p_human < 1e-3:
        score += 24
    elif p_human < 0.01:
        score += 18
    elif p_human < 0.05:
        score += 12
    else:
        score -= 15

    if or_human >= 4:
        score += 20
    elif or_human >= 2:
        score += 14
    elif or_human > 1:
        score += 8
    else:
        score -= 10

    # Bivariate support
    if biv_p < 1e-6:
        score += 8
    elif biv_p < 0.01:
        score += 5

    # Interpretable-model robustness / null evidence
    if positive_support >= 3:
        score += 20
    elif positive_support == 2:
        score += 12
    elif positive_support == 1:
        score += 5

    score -= 15 * zeroed_support
    score -= 10 * negative_support

    score = int(max(0, min(100, round(score))))

    explanation = (
        f"The human indicator shows a strong positive association with AMTL frequency after controls. "
        f"In the controlled binomial GLM (with age, prob_male, and tooth class), beta_human={beta_human:.3f} "
        f"(OR={or_human:.2f}, 95% CI {or_ci_low:.2f}-{or_ci_high:.2f}, p={p_human:.2e}), so humans have higher "
        f"AMTL odds than non-human genera conditional on covariates. Bivariate evidence is also strong "
        f"(beta={biv_beta:.3f}, OR={biv_or:.2f}, p={biv_p:.2e}; Mann-Whitney p={mw_p:.2e}). "
        f"Interpretable models are mostly consistent: SmartAdditive and HingeEBM assign positive human effects "
        f"({human_effect_signals['SmartAdditiveRegressor']['value']:.4f} and "
        f"{human_effect_signals['HingeEBMRegressor']['value']:.4f}), while WinsorizedSparseOLS zeroes out "
        f"the human term (null sparse evidence). Age is also a strong positive driver and there are tooth-class "
        f"differences, so the human effect is substantial but not the only determinant. Overall evidence supports "
        f"'Yes' with some penalty for one sparse-model zeroing."
    )

    out = {"response": score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(out, f)

    print("\nWrote conclusion.txt:")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
