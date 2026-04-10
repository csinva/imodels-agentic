import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def direction_sign(direction: str) -> int:
    d = (direction or "").lower()
    if "positive" in d or "increasing" in d:
        return 1
    if "negative" in d or "decreasing" in d:
        return -1
    return 0


def get_effect(effects: dict, feature: str) -> dict:
    eff = effects.get(feature, {}) if effects else {}
    return {
        "direction": eff.get("direction", "zero"),
        "importance": safe_float(eff.get("importance", 0.0)),
        "rank": int(eff.get("rank", 0) or 0),
    }


def summarize_top_features(effects: dict, k: int = 3):
    rows = []
    for name, info in (effects or {}).items():
        imp = safe_float(info.get("importance", 0.0))
        rank = int(info.get("rank", 0) or 0)
        direction = info.get("direction", "zero")
        if imp > 0 and rank > 0:
            rows.append((rank, name, imp, direction))
    rows.sort(key=lambda x: x[0])
    return rows[:k]


def score_response(
    ols_coef,
    ols_p,
    bivar_diff,
    smart_human_imp,
    smart_human_dir,
    hinge_human_imp,
    hinge_human_dir,
):
    # Base scoring by controlled regression (primary evidence)
    if ols_p < 0.01:
        score = 90 if ols_coef > 0 else 10
    elif ols_p < 0.05:
        score = 80 if ols_coef > 0 else 20
    elif ols_p < 0.10:
        score = 65 if ols_coef > 0 else 35
    else:
        score = 50

    # Adjust for interpretable model support/contradiction
    smart_sign = direction_sign(smart_human_dir)
    hinge_sign = direction_sign(hinge_human_dir)

    if smart_human_imp >= 0.10 and smart_sign > 0:
        score += 10
    elif smart_human_imp >= 0.10 and smart_sign < 0:
        score -= 10
    elif smart_human_imp < 0.05:
        score -= 8

    if hinge_human_imp >= 0.10 and hinge_sign > 0:
        score += 10
    elif hinge_human_imp >= 0.10 and hinge_sign < 0:
        score -= 10
    elif hinge_human_imp == 0 or hinge_human_dir == "zero":
        score -= 8

    # If controlled effect is null but raw bivariate is positive, keep a weak-to-moderate score
    if ols_p >= 0.10 and bivar_diff > 0:
        score = min(score, 35)
        score = max(score, 18)

    # If controlled effect is null and both interpretable models show weak/zero human importance
    if (
        ols_p >= 0.10
        and smart_human_imp < 0.08
        and (hinge_human_imp == 0 or hinge_human_dir == "zero")
    ):
        score = 22 if bivar_diff > 0 else 8

    return int(max(0, min(100, round(score))))


def main():
    info = json.loads(Path("info.json").read_text())
    question = info.get("research_questions", ["Unknown question"])[0]

    df = pd.read_csv("amtl.csv")

    # Define DV and IV from the research question context
    # DV: frequency of AMTL (num_amtl / sockets)
    # IV: human status vs. non-human primates
    df = df.copy()
    df = df[df["sockets"] > 0].copy()
    df["amtl_rate"] = df["num_amtl"] / df["sockets"]
    df["is_human"] = (df["genus"] == "Homo sapiens").astype(int)

    print("=== Research Question ===")
    print(question)
    print("\nIdentified DV: amtl_rate (num_amtl / sockets)")
    print("Identified IV: is_human (1=Homo sapiens, 0=Pan/Pongo/Papio)")

    print("\n=== Step 1: Exploration ===")
    numeric_cols = [
        "amtl_rate",
        "num_amtl",
        "sockets",
        "age",
        "stdev_age",
        "prob_male",
        "is_human",
    ]
    print("Numeric summary:")
    print(df[numeric_cols].describe().T)

    print("\nGenus counts:")
    print(df["genus"].value_counts())

    print("\nTooth class counts:")
    print(df["tooth_class"].value_counts())

    # Bivariate results for primary question
    human_rate = df.loc[df["is_human"] == 1, "amtl_rate"]
    nonhuman_rate = df.loc[df["is_human"] == 0, "amtl_rate"]
    bivar_diff = float(human_rate.mean() - nonhuman_rate.mean())
    bivar_corr = float(df[["amtl_rate", "is_human"]].corr().iloc[0, 1])
    t_stat, t_p = stats.ttest_ind(human_rate, nonhuman_rate, equal_var=False, nan_policy="omit")

    print("\nBivariate human vs non-human comparison:")
    print(f"mean(amtl_rate | human=1)={human_rate.mean():.4f}")
    print(f"mean(amtl_rate | human=0)={nonhuman_rate.mean():.4f}")
    print(f"difference={bivar_diff:.4f}")
    print(f"corr(amtl_rate, is_human)={bivar_corr:.4f}")
    print(f"Welch t-test: t={t_stat:.3f}, p={t_p:.4g}")

    # Controls: age, sex proxy, tooth class; plus sockets and stdev_age as measurement controls
    tooth_dummies = pd.get_dummies(df["tooth_class"], prefix="tooth", drop_first=True, dtype=float)
    X_base = pd.concat(
        [df[["is_human", "age", "prob_male", "sockets", "stdev_age"]], tooth_dummies], axis=1
    )
    y = df["amtl_rate"]

    model_df = pd.concat([y, X_base], axis=1).dropna()
    y_model = model_df["amtl_rate"]
    X_model = model_df.drop(columns=["amtl_rate"])

    print("\n=== Step 2: Controlled OLS ===")
    X_ols = sm.add_constant(X_model)
    ols_model = sm.OLS(y_model, X_ols).fit()
    print(ols_model.summary())

    ols_coef = float(ols_model.params.get("is_human", np.nan))
    ols_p = float(ols_model.pvalues.get("is_human", np.nan))

    print("\n=== Step 3: Interpretable Models ===")
    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_model, y_model)
    smart_effects = smart.feature_effects()

    print("SmartAdditiveRegressor model:")
    print(smart)
    print("SmartAdditive feature effects:")
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_model, y_model)
    hinge_effects = hinge.feature_effects()

    print("\nHingeEBMRegressor model:")
    print(hinge)
    print("HingeEBM feature effects:")
    print(hinge_effects)

    human_smart = get_effect(smart_effects, "is_human")
    human_hinge = get_effect(hinge_effects, "is_human")

    # Pull some shape detail for age from SmartAdditive
    age_shape_note = ""
    if "age" in X_model.columns:
        age_idx = list(X_model.columns).index("age")
        if hasattr(smart, "shape_functions_") and age_idx in smart.shape_functions_:
            thresholds, intervals = smart.shape_functions_[age_idx]
            if len(thresholds) > 0 and len(intervals) >= 2:
                age_shape_note = (
                    f"Age shows a nonlinear pattern with thresholds around "
                    f"{thresholds[0]:.1f} and {thresholds[-1]:.1f} years; "
                    f"predicted AMTL is much higher at older ages."
                )

    top_smart = summarize_top_features(smart_effects, k=3)
    top_hinge = summarize_top_features(hinge_effects, k=3)

    response = score_response(
        ols_coef=ols_coef,
        ols_p=ols_p,
        bivar_diff=bivar_diff,
        smart_human_imp=human_smart["importance"],
        smart_human_dir=human_smart["direction"],
        hinge_human_imp=human_hinge["importance"],
        hinge_human_dir=human_hinge["direction"],
    )

    smart_top_text = "; ".join(
        [f"{name} (rank {rank}, importance {imp:.1%}, {direction})" for rank, name, imp, direction in top_smart]
    )
    hinge_top_text = "; ".join(
        [f"{name} (rank {rank}, importance {imp:.1%}, {direction})" for rank, name, imp, direction in top_hinge]
    )

    explanation = (
        f"Raw AMTL frequency is higher in humans (mean difference={bivar_diff:.3f}; "
        f"corr={bivar_corr:.3f}; Welch t-test p={t_p:.3g}), but after controlling for age, sex proxy, "
        f"tooth class, sockets, and age uncertainty, the human indicator is near zero in OLS "
        f"(coef={ols_coef:.4f}, p={ols_p:.3g}). In SmartAdditive, is_human has only modest influence "
        f"(importance={human_smart['importance']:.1%}, rank={human_smart['rank']}, direction={human_smart['direction']}), "
        f"while stronger drivers are {smart_top_text}. In HingeEBM, is_human is effectively excluded "
        f"(importance={human_hinge['importance']:.1%}, direction={human_hinge['direction']}), with top effects {hinge_top_text}. "
        f"{age_shape_note} Overall, the apparent human/non-human difference is largely explained by confounding, especially age, "
        f"so evidence that modern humans intrinsically have higher AMTL frequency is weak."
    )

    output = {"response": response, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(output))

    print("\n=== Step 4: Conclusion JSON ===")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
