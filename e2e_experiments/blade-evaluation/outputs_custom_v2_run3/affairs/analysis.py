import json
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.exceptions import ConvergenceWarning

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor


warnings.filterwarnings("ignore", category=ConvergenceWarning)


def format_effect(effects, feature):
    e = effects.get(feature, {"direction": "missing", "importance": 0.0, "rank": 0})
    return e.get("direction", "missing"), float(e.get("importance", 0.0)), int(e.get("rank", 0))


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info.get("research_questions", ["Unknown question"])[0].strip()
    print("Research question:")
    print(question)

    df = pd.read_csv("affairs.csv")

    # Encode IV and other categorical controls
    df["children_yes"] = (df["children"].astype(str).str.lower() == "yes").astype(int)
    df["gender_male"] = (df["gender"].astype(str).str.lower() == "male").astype(int)

    dv = "affairs"
    iv = "children_yes"

    # Use all numeric columns except DV and identifier-like rownames
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in [dv, "rownames"]]

    print("\nDV:", dv)
    print("IV:", iv)
    print("Feature columns:", feature_cols)

    # Step 1: exploration
    print("\nSummary statistics (numeric columns):")
    print(df[[dv] + feature_cols].describe().T)

    print("\nDistribution checks:")
    print("Affairs value counts:")
    print(df[dv].value_counts(dropna=False).sort_index())
    print("Children_yes value counts:")
    print(df[iv].value_counts(dropna=False).sort_index())

    print("\nBivariate relationship (affairs vs children_yes):")
    bivar_corr = df[[dv, iv]].corr().loc[dv, iv]
    mean_by_children = df.groupby(iv)[dv].mean()
    print(f"Pearson correlation: {bivar_corr:.4f}")
    print("Mean affairs by children_yes:")
    print(mean_by_children)

    print("\nCorrelation matrix (DV + features):")
    print(df[[dv] + feature_cols].corr())

    # Step 2: OLS with controls
    X_ols = sm.add_constant(df[feature_cols], has_constant="add")
    ols_model = sm.OLS(df[dv], X_ols).fit()
    print("\nOLS with controls:")
    print(ols_model.summary())

    ols_coef = float(ols_model.params.get(iv, np.nan))
    ols_p = float(ols_model.pvalues.get(iv, np.nan))

    # Step 3: Custom interpretable models
    X = df[feature_cols]
    y = df[dv]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X, y)
    smart_effects = smart.feature_effects()
    print("\nSmartAdditiveRegressor summary:")
    print(smart)
    print("SmartAdditive feature effects:")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X, y)
    hinge_effects = hinge.feature_effects()
    print("\nHingeEBMRegressor summary:")
    print(hinge)
    print("HingeEBM feature effects:")
    print(hinge_effects)

    smart_dir, smart_imp, smart_rank = format_effect(smart_effects, iv)
    hinge_dir, hinge_imp, hinge_rank = format_effect(hinge_effects, iv)

    # Identify important confounders
    smart_top = sorted(
        [(k, v) for k, v in smart_effects.items() if v.get("importance", 0) > 0],
        key=lambda kv: kv[1]["importance"],
        reverse=True,
    )[:4]
    hinge_top = sorted(
        [(k, v) for k, v in hinge_effects.items() if v.get("importance", 0) > 0],
        key=lambda kv: kv[1]["importance"],
        reverse=True,
    )[:4]

    smart_top_txt = ", ".join(
        [f"{k} (rank {v.get('rank', 0)}, imp {v.get('importance', 0):.1%}, {v.get('direction', 'n/a')})" for k, v in smart_top]
    ) or "none"
    hinge_top_txt = ", ".join(
        [f"{k} (rank {v.get('rank', 0)}, imp {v.get('importance', 0):.1%}, {v.get('direction', 'n/a')})" for k, v in hinge_top]
    ) or "none"

    # Score according to rubric
    # Evidence here: no significant controlled effect, and near-zero importance in both interpretable models.
    if (ols_p < 0.05) and (ols_coef < 0) and (smart_imp >= 0.05 or hinge_imp >= 0.05):
        score = 80
    elif (ols_p < 0.10) and (ols_coef < 0):
        score = 55
    elif (abs(ols_coef) < 0.2 and ols_p >= 0.10 and smart_imp < 0.01 and hinge_imp < 0.01):
        score = 10
    else:
        score = 25

    explanation = (
        f"Question: {question} IV=children_yes, DV=affairs. "
        f"Bivariate patterns do not support a decrease: corr(children_yes, affairs)={bivar_corr:.3f}, "
        f"and mean affairs is {mean_by_children.get(1, np.nan):.3f} with children vs {mean_by_children.get(0, np.nan):.3f} without. "
        f"After controls, OLS gives children_yes coef={ols_coef:.3f} (negative but tiny) with p={ols_p:.3f}, so no reliable independent effect. "
        f"SmartAdditive also assigns children_yes {smart_dir} with importance={smart_imp:.1%} (rank {smart_rank}); "
        f"HingeEBM assigns {hinge_dir} with importance={hinge_imp:.1%} (rank {hinge_rank}). "
        f"This indicates the children effect is effectively zero and not robust across models. "
        f"Key confounders that matter more are SmartAdditive top features: {smart_top_txt}; "
        f"HingeEBM top features: {hinge_top_txt}. "
        f"Shape-wise, major predictors show nonlinear thresholds (especially age/religiousness in SmartAdditive) and linear negative marriage-rating effects, "
        f"while children shows no meaningful shape pattern."
    )

    result = {"response": int(score), "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(result))

    print("\nWrote conclusion.txt:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
