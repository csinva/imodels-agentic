import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor


warnings.filterwarnings("ignore")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def top_effects(effects, top_n=5):
    items = []
    for feature, vals in effects.items():
        imp = safe_float(vals.get("importance", 0.0))
        rank = vals.get("rank", 0)
        direction = vals.get("direction", "unknown")
        items.append((rank, imp, feature, direction))

    items.sort(key=lambda t: (t[0] == 0, t[0], -t[1]))
    return items[:top_n]


def find_direction_importance(effects, feature_name):
    if feature_name not in effects:
        return ("missing", 0.0, 0)
    vals = effects[feature_name]
    return (
        vals.get("direction", "unknown"),
        safe_float(vals.get("importance", 0.0)),
        int(vals.get("rank", 0) or 0),
    )


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info["research_questions"][0].strip()
    print("Research question:", question)

    df = pd.read_csv("affairs.csv")
    print("\nData shape:", df.shape)
    print("Columns:", list(df.columns))

    dv = "affairs"
    iv = "children"
    print(f"\nIdentified DV: {dv}")
    print(f"Identified IV: {iv}")

    # Encode categoricals for modeling
    df_model = df.copy()
    df_model["children_bin"] = (df_model["children"].astype(str).str.lower() == "yes").astype(int)
    df_model["gender_bin"] = (df_model["gender"].astype(str).str.lower() == "male").astype(int)

    # Keep all meaningful numeric predictors; rownames is an identifier, not a covariate
    feature_cols = [
        "children_bin",
        "gender_bin",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]

    # Step 1: Exploration
    print("\n=== Summary statistics (core variables) ===")
    print(df_model[[dv] + feature_cols].describe().T)

    print("\n=== DV distribution ===")
    print(df_model[dv].value_counts().sort_index())

    print("\n=== IV distribution ===")
    print(df_model[iv].value_counts())

    print("\n=== Bivariate relationship (children vs affairs) ===")
    group_means = df_model.groupby("children")[dv].agg(["mean", "median", "count", "std"])
    print(group_means)

    corr_children_affairs = df_model["children_bin"].corr(df_model[dv])
    print(f"Pearson corr(children_bin, affairs): {corr_children_affairs:.4f}")

    no_group = df_model.loc[df_model["children_bin"] == 0, dv]
    yes_group = df_model.loc[df_model["children_bin"] == 1, dv]
    t_stat, t_p = stats.ttest_ind(yes_group, no_group, equal_var=False)
    print(f"Welch t-test children=yes vs no: t={t_stat:.4f}, p={t_p:.4g}")

    print("\n=== Correlations with DV ===")
    corr_series = df_model[[dv] + feature_cols].corr()[dv].sort_values(ascending=False)
    print(corr_series)

    # Step 2: OLS with controls
    print("\n=== OLS with controls ===")
    X = sm.add_constant(df_model[feature_cols])
    y = df_model[dv]
    ols_model = sm.OLS(y, X).fit()
    print(ols_model.summary())

    # Standardized OLS for relative magnitude comparison
    X_std = df_model[feature_cols].apply(lambda c: (c - c.mean()) / c.std(ddof=0))
    y_std = (y - y.mean()) / y.std(ddof=0)
    X_std = sm.add_constant(X_std)
    ols_std = sm.OLS(y_std, X_std).fit()
    std_coefs = ols_std.params.drop("const")
    std_rank = std_coefs.abs().sort_values(ascending=False)
    print("\nStandardized coefficient magnitudes (abs):")
    print(std_rank)

    # Step 3: Interpretable models
    print("\n=== SmartAdditiveRegressor ===")
    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(df_model[feature_cols], y)
    print(smart)
    smart_effects = smart.feature_effects()
    print("Smart effects:", smart_effects)

    print("\n=== HingeEBMRegressor ===")
    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(df_model[feature_cols], y)
    print(hinge)
    hinge_effects = hinge.feature_effects()
    print("Hinge effects:", hinge_effects)

    # Pull key evidence for IV=children
    ols_coef = safe_float(ols_model.params["children_bin"])
    ols_p = safe_float(ols_model.pvalues["children_bin"])
    ols_ci_low, ols_ci_high = [safe_float(v) for v in ols_model.conf_int().loc["children_bin"].tolist()]

    biv_diff = safe_float(group_means.loc["yes", "mean"] - group_means.loc["no", "mean"])

    smart_dir, smart_imp, smart_rank = find_direction_importance(smart_effects, "children_bin")
    hinge_dir, hinge_imp, hinge_rank = find_direction_importance(hinge_effects, "children_bin")

    # Determine score based on consistency and strength of evidence for "children decreases affairs"
    supports_decrease = 0

    # Controlled OLS evidence (most important)
    if ols_coef < 0 and ols_p < 0.05:
        supports_decrease += 2
    elif ols_coef < 0 and ols_p < 0.20:
        supports_decrease += 1

    # Bivariate evidence
    if biv_diff < 0:
        supports_decrease += 1

    # Interpretable model evidence
    if "negative" in smart_dir and smart_imp >= 0.05:
        supports_decrease += 1
    if "negative" in hinge_dir and hinge_imp >= 0.05:
        supports_decrease += 1

    # Map evidence to Likert score for the specific hypothesis (children DECREASE affairs)
    # Here evidence is weak/inconsistent, so keep score in low range.
    if supports_decrease >= 4:
        response = 85
    elif supports_decrease == 3:
        response = 68
    elif supports_decrease == 2:
        response = 52
    elif supports_decrease == 1:
        response = 30
    else:
        response = 12

    # Build concise but rich explanation
    top_smart = top_effects(smart_effects, top_n=4)
    top_hinge = top_effects(hinge_effects, top_n=4)

    top_smart_str = "; ".join(
        [f"{feat} ({direction}, imp={imp:.1%}, rank={rank})" for rank, imp, feat, direction in top_smart]
    )
    top_hinge_str = "; ".join(
        [f"{feat} ({direction}, imp={imp:.1%}, rank={rank})" for rank, imp, feat, direction in top_hinge]
    )

    explanation = (
        f"Hypothesis tested: having children decreases extramarital affairs. "
        f"Bivariate results go the opposite direction: mean affairs is higher with children "
        f"(yes-no diff={biv_diff:.3f}; corr={corr_children_affairs:.3f}; t-test p={t_p:.3g}). "
        f"After controls (gender, age, years married, religiousness, education, occupation, marriage rating), "
        f"children has a small negative OLS coefficient but is not significant "
        f"(coef={ols_coef:.3f}, p={ols_p:.3f}, 95% CI [{ols_ci_low:.3f}, {ols_ci_high:.3f}]). "
        f"SmartAdditive assigns children zero importance (direction={smart_dir}, imp={smart_imp:.1%}, rank={smart_rank}), "
        f"and HingeEBM also zeroes it out (direction={hinge_dir}, imp={hinge_imp:.1%}, rank={hinge_rank}). "
        f"Main drivers are marriage rating, age/years married, and religiousness; top Smart effects: {top_smart_str}. "
        f"Top Hinge effects: {top_hinge_str}. "
        f"Shape evidence is nonlinear for age/religiousness in SmartAdditive, while children shows no stable nonlinear threshold effect. "
        f"Overall, any negative children effect is weak and not robust across models, so support for the claim is low."
    )

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump({"response": int(response), "explanation": explanation}, f)

    print("\nWrote conclusion.txt")
    print(json.dumps({"response": int(response), "explanation": explanation}, indent=2))


if __name__ == "__main__":
    main()
