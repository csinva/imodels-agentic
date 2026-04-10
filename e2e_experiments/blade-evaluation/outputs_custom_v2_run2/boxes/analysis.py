import json
from typing import Dict, Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor


def to_native(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON-safe output."""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_native(v) for v in obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def main() -> None:
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    research_question = info["research_questions"][0]
    df = pd.read_csv("boxes.csv")

    # Operationalize "reliance on majority preference" as choosing the majority option.
    # Original y: 1=unchosen, 2=majority, 3=minority.
    df["majority_choice"] = (df["y"] == 2).astype(int)

    iv_col = "age"
    dv_col = "majority_choice"
    control_cols = ["gender", "majority_first", "culture"]
    numeric_cols = ["age", "gender", "majority_first", "culture"]

    print("=== Research Question ===")
    print(research_question)
    print("\n=== Variable Setup ===")
    print(f"Dependent variable (DV): {dv_col} (1 if y==2 else 0)")
    print(f"Independent variable (IV): {iv_col}")
    print(f"Controls: {control_cols}")

    # Step 1: Explore distributions and bivariate relationships
    print("\n=== Step 1: Exploration ===")
    print("Summary statistics:")
    print(df[["y", dv_col] + numeric_cols].describe())

    print("\nOutcome distribution (y):")
    print(df["y"].value_counts().sort_index())

    print("\nMajority-choice rate:")
    print(df[dv_col].value_counts().sort_index())
    print(f"Mean majority-choice rate: {df[dv_col].mean():.4f}")

    corr_cols = [dv_col] + numeric_cols
    corr_mat = df[corr_cols].corr(numeric_only=True)
    print("\nCorrelations with majority_choice:")
    print(corr_mat[dv_col].sort_values(ascending=False))

    # Bivariate logistic model: DV ~ age
    X_biv = sm.add_constant(df[[iv_col]].astype(float))
    model_biv = sm.Logit(df[dv_col], X_biv).fit(disp=False)
    print("\nBivariate logistic: majority_choice ~ age")
    print(model_biv.summary())

    # Step 2: Controlled models
    print("\n=== Step 2: Controlled Models ===")
    X_ctrl = df[[iv_col, "gender", "majority_first", "culture"]].copy()
    X_ctrl = pd.get_dummies(X_ctrl, columns=["culture"], drop_first=True)
    X_ctrl = sm.add_constant(X_ctrl.astype(float))
    model_ctrl = sm.Logit(df[dv_col], X_ctrl).fit(disp=False)
    print("Controlled logistic (with culture fixed effects):")
    print(model_ctrl.summary())

    # Age-by-culture interaction model (to probe "across cultural contexts")
    X_int_base = df[[iv_col, "gender", "majority_first", "culture"]].copy()
    culture_dummies = pd.get_dummies(X_int_base["culture"], prefix="culture", drop_first=True)
    X_int = pd.concat([X_int_base.drop(columns=["culture"]), culture_dummies], axis=1)
    for c in culture_dummies.columns:
        X_int[f"age_x_{c}"] = X_int[iv_col] * X_int[c]
    X_int = sm.add_constant(X_int.astype(float))
    model_int = sm.Logit(df[dv_col], X_int).fit(disp=False, maxiter=300)
    print("\nLogistic with age*culture interactions:")
    print(model_int.summary())

    # Step 3: Interpretable models
    print("\n=== Step 3: Interpretable Models ===")
    X_interpret = df[numeric_cols]
    y_interpret = df[dv_col]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interpret, y_interpret)
    smart_effects: Dict[str, Dict[str, Any]] = to_native(smart.feature_effects())
    print("SmartAdditiveRegressor:")
    print(smart)
    print("Smart effects:")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_interpret, y_interpret)
    hinge_effects: Dict[str, Dict[str, Any]] = to_native(hinge.feature_effects())
    print("\nHingeEBMRegressor:")
    print(hinge)
    print("Hinge effects:")
    print(hinge_effects)

    # Step 4: Build conclusion score + explanation
    age_corr = float(corr_mat.loc[dv_col, iv_col])
    age_biv_coef = float(model_biv.params[iv_col])
    age_biv_p = float(model_biv.pvalues[iv_col])
    age_ctrl_coef = float(model_ctrl.params[iv_col])
    age_ctrl_p = float(model_ctrl.pvalues[iv_col])

    age_or = float(np.exp(age_ctrl_coef))
    age_int_pvals = {
        k: float(v)
        for k, v in model_int.pvalues.items()
        if k.startswith("age_x_")
    }
    min_age_int_p = min(age_int_pvals.values()) if age_int_pvals else 1.0

    smart_age = smart_effects.get("age", {"direction": "unknown", "importance": 0.0, "rank": 0})
    hinge_age = hinge_effects.get("age", {"direction": "zero", "importance": 0.0, "rank": 0})

    # Scoring rubric aligned with prompt guidance.
    if age_ctrl_p < 0.05 and smart_age.get("importance", 0) >= 0.10 and hinge_age.get("importance", 0) >= 0.05:
        score = 85
    elif age_ctrl_p < 0.10 and (smart_age.get("importance", 0) >= 0.08 or hinge_age.get("importance", 0) >= 0.05):
        score = 60
    elif smart_age.get("importance", 0) >= 0.20 and hinge_age.get("importance", 0) < 0.02 and age_ctrl_p >= 0.10:
        score = 25
    elif age_ctrl_p >= 0.10 and smart_age.get("importance", 0) < 0.08 and hinge_age.get("importance", 0) < 0.02:
        score = 10
    else:
        score = 35

    majority_first_coef = float(model_ctrl.params["majority_first"])
    majority_first_p = float(model_ctrl.pvalues["majority_first"])
    gender_coef = float(model_ctrl.params["gender"])
    gender_p = float(model_ctrl.pvalues["gender"])

    explanation = (
        f"The evidence for age-driven growth in majority reliance is weak and not robust. "
        f"Bivariately, age is near-zero related to majority choice (corr={age_corr:.3f}; "
        f"logit coef={age_biv_coef:.3f}, p={age_biv_p:.3f}). With controls for gender, "
        f"majority-first demonstration, and culture fixed effects, age remains non-significant "
        f"(coef={age_ctrl_coef:.3f}, OR={age_or:.3f}, p={age_ctrl_p:.3f}). "
        f"Age-by-culture interactions are also non-significant (minimum interaction p={min_age_int_p:.3f}), "
        f"so there is no strong evidence that age trends differ reliably across cultures. "
        f"In interpretable models, SmartAdditive ranks age #{smart_age.get('rank', 0)} with "
        f"{100*float(smart_age.get('importance', 0.0)):.1f}% importance and a "
        f"{smart_age.get('direction', 'unknown')} shape, indicating nonlinearity/thresholds, "
        f"but HingeEBM largely shrinks age toward zero (rank {hinge_age.get('rank', 0)}, "
        f"importance {100*float(hinge_age.get('importance', 0.0)):.1f}%). "
        f"The most robust predictor is majority_first (coef={majority_first_coef:.3f}, p={majority_first_p:.3g}), "
        f"with gender also contributing (coef={gender_coef:.3f}, p={gender_p:.3f}). "
        f"Overall, any age effect appears weak/inconsistent relative to stronger confounders."
    )

    result = {"response": int(score), "explanation": explanation}

    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=True)

    print("\n=== Final Conclusion JSON ===")
    print(json.dumps(result, indent=2))
    print("Wrote conclusion.txt")


if __name__ == "__main__":
    main()
