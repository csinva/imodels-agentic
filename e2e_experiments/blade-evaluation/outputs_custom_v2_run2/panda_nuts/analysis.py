import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor


def to_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return float(default)


def extract_param_stats(model, prefix: str) -> Tuple[float, float, str]:
    for name in model.params.index:
        if name.startswith(prefix):
            return to_float(model.params[name]), to_float(model.pvalues[name]), name
    return np.nan, np.nan, ""


def aggregate_effect(effects: Dict, base_name: str) -> Dict:
    keys = [k for k in effects.keys() if k == base_name or k.startswith(base_name + "_")]
    if not keys:
        return {"importance": 0.0, "rank": 0, "direction": "zero", "key": ""}

    ranked_nonzero = [
        (k, effects[k])
        for k in keys
        if to_float(effects[k].get("importance", 0.0), 0.0) > 0
    ]

    total_importance = sum(to_float(effects[k].get("importance", 0.0), 0.0) for k in keys)

    if ranked_nonzero:
        top_key, top_entry = sorted(
            ranked_nonzero,
            key=lambda t: to_float(t[1].get("importance", 0.0), 0.0),
            reverse=True,
        )[0]
        nonzero_ranks = [
            int(effects[k].get("rank", 0))
            for k in keys
            if int(effects[k].get("rank", 0)) > 0
        ]
        best_rank = min(nonzero_ranks) if nonzero_ranks else 0
        return {
            "importance": float(total_importance),
            "rank": int(best_rank),
            "direction": str(top_entry.get("direction", "zero")),
            "key": top_key,
        }

    return {"importance": float(total_importance), "rank": 0, "direction": "zero", "key": ""}


def variable_evidence_score(
    corr_val: float,
    p_simple: float,
    p_fe: float,
    smart_imp: float,
    hinge_imp: float,
) -> float:
    points = 0.0
    max_points = 6.5

    if np.isfinite(corr_val):
        if abs(corr_val) >= 0.25:
            points += 1.0
        elif abs(corr_val) >= 0.10:
            points += 0.5

    if np.isfinite(p_simple):
        if p_simple < 0.05:
            points += 1.5
        elif p_simple < 0.10:
            points += 0.75

    if np.isfinite(p_fe):
        if p_fe < 0.05:
            points += 1.5
        elif p_fe < 0.10:
            points += 0.75

    if smart_imp >= 0.10:
        points += 1.5
    elif smart_imp >= 0.03:
        points += 0.75

    if hinge_imp >= 0.10:
        points += 1.5
    elif hinge_imp >= 0.03:
        points += 0.75

    return max(0.0, min(100.0, 100.0 * points / max_points))


def main():
    with open("info.json", "r", encoding="utf-8") as f:
        info = json.load(f)

    question = info["research_questions"][0]
    print("Research question:")
    print(question)
    print()

    df = pd.read_csv("panda_nuts.csv")

    # Standardize categorical values.
    df["sex"] = df["sex"].astype(str).str.strip().str.lower()
    df["help"] = df["help"].astype(str).str.strip().str.lower().replace({"yes": "y", "no": "n"})
    df["hammer"] = df["hammer"].astype(str).str.strip()

    # DV: nut-cracking efficiency (nuts opened per second).
    df["efficiency"] = df["nuts_opened"] / df["seconds"].replace(0, np.nan)
    df = df.dropna(subset=["efficiency"]).copy()

    # Binary encodings for bivariate checks.
    df["sex_m"] = (df["sex"] == "m").astype(int)
    df["help_y"] = (df["help"] == "y").astype(int)

    print("Identified variables:")
    print("DV: efficiency = nuts_opened / seconds")
    print("Main IVs: age, sex, help")
    print("Controls: hammer type, chimpanzee identity")
    print()

    print("Step 1: Summary statistics")
    numeric_cols = ["efficiency", "age", "nuts_opened", "seconds", "chimpanzee", "sex_m", "help_y"]
    print(df[numeric_cols].describe().T)
    print()

    print("Categorical distributions:")
    print("sex counts:")
    print(df["sex"].value_counts(dropna=False))
    print("help counts:")
    print(df["help"].value_counts(dropna=False))
    print("hammer counts:")
    print(df["hammer"].value_counts(dropna=False))
    print()

    corr_matrix = df[numeric_cols].corr(numeric_only=True)
    print("Correlation matrix (numeric columns):")
    print(corr_matrix)
    print()

    corr_eff = corr_matrix["efficiency"].sort_values(ascending=False)
    print("Bivariate correlations with efficiency:")
    print(corr_eff)
    print()

    # Step 2: Controlled regression models.
    print("Step 2: OLS with controls")
    formula_simple = "efficiency ~ age + C(sex) + C(help) + C(hammer)"
    model_simple = smf.ols(formula_simple, data=df).fit()
    print("Model A (controls: sex, help, hammer):")
    print(model_simple.summary())
    print()

    formula_fe = "efficiency ~ age + C(sex) + C(help) + C(hammer) + C(chimpanzee)"
    model_fe = smf.ols(formula_fe, data=df).fit()
    print("Model B (+ chimpanzee fixed effects):")
    print(model_fe.summary())
    print()

    # Step 3: Interpretable models.
    print("Step 3: Interpretable additive/sparse models")
    # For interpretability models, keep substantive predictors + controls,
    # with categorical variables one-hot encoded.
    X = pd.get_dummies(
        df[["age", "sex", "help", "hammer", "chimpanzee"]],
        columns=["sex", "help", "hammer", "chimpanzee"],
        drop_first=True,
    )
    y = df["efficiency"]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X, y)
    smart_effects = smart.feature_effects()
    print("SmartAdditiveRegressor:")
    print(smart)
    print("Smart effects:")
    print(smart_effects)
    print()

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X, y)
    hinge_effects = hinge.feature_effects()
    print("HingeEBMRegressor:")
    print(hinge)
    print("Hinge effects:")
    print(hinge_effects)
    print()

    # Collect targeted stats for age/sex/help.
    age_coef_simple, age_p_simple, _ = extract_param_stats(model_simple, "age")
    sex_coef_simple, sex_p_simple, _ = extract_param_stats(model_simple, "C(sex)")
    help_coef_simple, help_p_simple, _ = extract_param_stats(model_simple, "C(help)")

    age_coef_fe, age_p_fe, _ = extract_param_stats(model_fe, "age")
    sex_coef_fe, sex_p_fe, _ = extract_param_stats(model_fe, "C(sex)")
    help_coef_fe, help_p_fe, _ = extract_param_stats(model_fe, "C(help)")

    smart_age = aggregate_effect(smart_effects, "age")
    smart_sex = aggregate_effect(smart_effects, "sex")
    smart_help = aggregate_effect(smart_effects, "help")

    hinge_age = aggregate_effect(hinge_effects, "age")
    hinge_sex = aggregate_effect(hinge_effects, "sex")
    hinge_help = aggregate_effect(hinge_effects, "help")

    # Convert correlation values.
    corr_age = to_float(corr_eff.get("age", np.nan), np.nan)
    corr_sex = to_float(corr_eff.get("sex_m", np.nan), np.nan)
    corr_help = to_float(corr_eff.get("help_y", np.nan), np.nan)

    age_score = variable_evidence_score(
        corr_age,
        age_p_simple,
        age_p_fe,
        to_float(smart_age["importance"], 0.0),
        to_float(hinge_age["importance"], 0.0),
    )
    sex_score = variable_evidence_score(
        corr_sex,
        sex_p_simple,
        sex_p_fe,
        to_float(smart_sex["importance"], 0.0),
        to_float(hinge_sex["importance"], 0.0),
    )
    help_score = variable_evidence_score(
        corr_help,
        help_p_simple,
        help_p_fe,
        to_float(smart_help["importance"], 0.0),
        to_float(hinge_help["importance"], 0.0),
    )

    overall_score = int(round((age_score + sex_score + help_score) / 3.0))
    overall_score = max(0, min(100, overall_score))

    explanation = (
        f"Using efficiency (nuts_opened/seconds) as the DV, age shows the strongest and most consistent signal: "
        f"bivariate r={corr_age:.3f}, OLS coef={age_coef_simple:.3f} (p={age_p_simple:.3g}) with hammer/sex/help controls, "
        f"and coef={age_coef_fe:.3f} (p={age_p_fe:.3g}) after chimpanzee fixed effects; "
        f"SmartAdditive ranks age #{smart_age['rank']} with {to_float(smart_age['importance'], 0.0):.1%} importance and "
        f"{smart_age['direction']} shape, while HingeEBM also ranks age #{hinge_age['rank']} at {to_float(hinge_age['importance'], 0.0):.1%}. "
        f"Sex is positive but less robust: OLS coef={sex_coef_simple:.3f} (p={sex_p_simple:.3g}) weakens with chimp controls "
        f"(coef={sex_coef_fe:.3f}, p={sex_p_fe:.3g}); SmartAdditive gives {to_float(smart_sex['importance'], 0.0):.1%} importance "
        f"(rank {smart_sex['rank']}), and HingeEBM gives {to_float(hinge_sex['importance'], 0.0):.1%}. "
        f"Help is mostly negative in OLS (coef={help_coef_simple:.3f}, p={help_p_simple:.3g}; FE coef={help_coef_fe:.3f}, p={help_p_fe:.3g}) "
        f"but is weaker/inconsistent in sparse modeling (SmartAdditive {to_float(smart_help['importance'], 0.0):.1%}, rank {smart_help['rank']}; "
        f"HingeEBM {to_float(hinge_help['importance'], 0.0):.1%}). "
        f"Confounders matter: hammer type and individual chimp differences explain substantial variance, which reduces certainty for sex/help. "
        f"Overall this supports a moderate, not universal, influence of age/sex/help on efficiency."
    )

    conclusion = {"response": overall_score, "explanation": explanation}
    with open("conclusion.txt", "w", encoding="utf-8") as f:
        json.dump(conclusion, f, ensure_ascii=True)

    print("Wrote conclusion.txt")
    print(conclusion)


if __name__ == "__main__":
    main()
