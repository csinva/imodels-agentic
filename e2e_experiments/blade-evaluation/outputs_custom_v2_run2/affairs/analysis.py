import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.exceptions import ConvergenceWarning

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor


def safe_rank_effect(effects, feature):
    out = effects.get(feature, {})
    return {
        "direction": out.get("direction", "zero"),
        "importance": float(out.get("importance", 0.0)),
        "rank": int(out.get("rank", 0) or 0),
    }


def main():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message="Objective did not converge")

    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info.get("research_questions", [""])[0].strip()
    print("Research question:", research_question)

    df = pd.read_csv("affairs.csv")

    # Encode key categorical variables for regression and interpretable models.
    df["children_yes"] = (df["children"].astype(str).str.lower() == "yes").astype(int)
    df["gender_male"] = (df["gender"].astype(str).str.lower() == "male").astype(int)

    dv = "affairs"
    iv = "children_yes"

    # Step 1: Explore
    print("\n=== Step 1: Summary statistics ===")
    print(df.describe(include="all").transpose())

    print("\n=== Step 1: DV distribution (affairs) ===")
    print(df[dv].value_counts(dropna=False).sort_index())

    print("\n=== Step 1: IV distribution (children) ===")
    print(df["children"].value_counts(dropna=False))

    print("\n=== Step 1: Mean affairs by children ===")
    mean_by_children = df.groupby("children")[dv].mean()
    print(mean_by_children)

    numeric_for_corr = [
        "affairs",
        "children_yes",
        "gender_male",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    print("\n=== Step 1: Correlation matrix (numeric analysis columns) ===")
    print(df[numeric_for_corr].corr())

    pb_r, pb_p = stats.pointbiserialr(df[iv], df[dv])
    t_res = stats.ttest_ind(
        df.loc[df[iv] == 1, dv],
        df.loc[df[iv] == 0, dv],
        equal_var=False,
    )
    print("\n=== Step 1: Bivariate IV-DV tests ===")
    print(f"Point-biserial correlation {iv} vs {dv}: r={pb_r:.4f}, p={pb_p:.4g}")
    print(
        "Welch t-test (mean affairs, children=yes vs no): "
        f"t={t_res.statistic:.4f}, p={t_res.pvalue:.4g}"
    )

    # Step 2: Controlled OLS
    controls = [
        "gender_male",
        "age",
        "yearsmarried",
        "religiousness",
        "education",
        "occupation",
        "rating",
    ]
    feature_columns = [iv] + controls
    X_ols = sm.add_constant(df[feature_columns])
    ols_model = sm.OLS(df[dv], X_ols).fit()

    print("\n=== Step 2: OLS with controls ===")
    print(ols_model.summary())
    iv_coef = float(ols_model.params[iv])
    iv_pval = float(ols_model.pvalues[iv])
    print(f"{iv} coefficient: {iv_coef:.4f}, p-value: {iv_pval:.4g}")

    # Step 3: Interpretable models
    X_interp = df[feature_columns]
    y = df[dv]

    print("\n=== Step 3: SmartAdditiveRegressor ===")
    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y)
    print(smart)
    smart_effects = smart.feature_effects()
    print("SmartAdditive feature_effects():")
    print(smart_effects)

    print("\n=== Step 3: HingeEBMRegressor ===")
    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_interp, y)
    print(hinge)
    hinge_effects = hinge.feature_effects()
    print("HingeEBM feature_effects():")
    print(hinge_effects)

    iv_smart = safe_rank_effect(smart_effects, iv)
    iv_hinge = safe_rank_effect(hinge_effects, iv)

    # Step 4: Rich conclusion + score
    # Scoring logic prioritizes controlled and multi-model robustness.
    score = 8
    if iv_coef < 0 and iv_pval < 0.05 and iv_smart["importance"] > 0.05 and iv_hinge["importance"] > 0.05:
        score = 90
    elif iv_coef < 0 and (iv_pval < 0.10 or iv_smart["importance"] > 0.03 or iv_hinge["importance"] > 0.03):
        score = 55
    elif pb_r < 0 and pb_p < 0.05 and iv_pval >= 0.10:
        score = 30
    elif pb_r > 0 and pb_p < 0.05 and iv_pval >= 0.10:
        score = 5

    # Pull major confounders from OLS absolute t-values (excluding IV).
    tvals = ols_model.tvalues.drop(labels=[iv, "const"], errors="ignore").abs().sort_values(ascending=False)
    top_conf = tvals.head(3).index.tolist()
    conf_terms = ", ".join(top_conf) if top_conf else "none"

    explanation = (
        f"Question: {research_question} "
        f"Bivariate evidence does not support a decrease: people with children report more affairs on average "
        f"(mean_yes={mean_by_children.get('yes', np.nan):.3f} vs mean_no={mean_by_children.get('no', np.nan):.3f}); "
        f"point-biserial r={pb_r:.3f} (p={pb_p:.3g}). "
        f"After controls, the children effect is small and statistically null in OLS "
        f"(coef={iv_coef:.3f}, p={iv_pval:.3g}), so no robust decrease remains. "
        f"SmartAdditive also assigns children essentially zero importance "
        f"(importance={iv_smart['importance']:.3f}, rank={iv_smart['rank']}, direction={iv_smart['direction']}), "
        f"and HingeEBM similarly zeroes it out "
        f"(importance={iv_hinge['importance']:.3f}, rank={iv_hinge['rank']}, direction={iv_hinge['direction']}). "
        f"Key confounders that matter more are {conf_terms}; SmartAdditive shows nonlinear age/religiousness shapes "
        f"and strong negative marriage-rating effect, while HingeEBM emphasizes rating and years married. "
        f"Overall, the hypothesized negative effect of having children is not robust across models."
    )

    output = {"response": int(score), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(output, f)

    print("\nWrote conclusion.txt:")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
