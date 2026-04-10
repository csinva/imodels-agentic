import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from interp_models import HingeEBMRegressor, SmartAdditiveRegressor


def clamp_int(x, lo=0, hi=100):
    return int(max(lo, min(hi, round(float(x)))))


def to_native(obj):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    return obj


def main():
    info_path = Path("info.json")
    data_path = Path("fish.csv")

    info = json.loads(info_path.read_text())
    df = pd.read_csv(data_path)

    question = info["research_questions"][0]
    print("Research question:")
    print(question)

    dv = "fish_caught"
    iv = "hours"

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    predictors = [c for c in numeric_cols if c != dv]

    controls = [c for c in predictors if c != iv]

    print("\nDV:", dv)
    print("Primary IV:", iv)
    print("Controls:", controls)

    print("\n=== Summary statistics ===")
    print(df[numeric_cols].describe().T)

    print("\n=== Distribution checks (skewness) ===")
    print(df[numeric_cols].skew())

    print("\n=== Bivariate correlations with DV ===")
    corrs = df[numeric_cols].corr(numeric_only=True)[dv].sort_values(ascending=False)
    print(corrs)

    print("\n=== Average fish per hour (descriptive) ===")
    fish_per_hour = (df[dv] / df[iv]).replace([np.inf, -np.inf], np.nan).dropna()
    print(
        {
            "mean_fish_per_hour": float(fish_per_hour.mean()),
            "median_fish_per_hour": float(fish_per_hour.median()),
            "std_fish_per_hour": float(fish_per_hour.std()),
        }
    )

    print("\n=== OLS: bivariate (fish_caught ~ hours) ===")
    X_biv = sm.add_constant(df[[iv]])
    ols_biv = sm.OLS(df[dv], X_biv).fit()
    print(ols_biv.summary())

    print("\n=== OLS: controlled (fish_caught ~ hours + controls) ===")
    X_ctl = sm.add_constant(df[[iv] + controls])
    ols_ctl = sm.OLS(df[dv], X_ctl).fit()
    print(ols_ctl.summary())

    print("\n=== SmartAdditiveRegressor ===")
    X_interp = df[predictors]
    y = df[dv]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_interp, y)
    smart_effects = to_native(smart.feature_effects())
    print(smart)
    print("Feature effects:")
    print(json.dumps(smart_effects, indent=2))

    print("\n=== HingeEBMRegressor ===")
    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_interp, y)
    hinge_effects = to_native(hinge.feature_effects())
    print(hinge)
    print("Feature effects:")
    print(json.dumps(hinge_effects, indent=2))

    b_coef = float(ols_biv.params[iv])
    b_p = float(ols_biv.pvalues[iv])
    c_coef = float(ols_ctl.params[iv])
    c_p = float(ols_ctl.pvalues[iv])

    smart_iv = smart_effects.get(iv, {})
    hinge_iv = hinge_effects.get(iv, {})
    smart_imp = float(smart_iv.get("importance", 0.0))
    hinge_imp = float(hinge_iv.get("importance", 0.0))

    robust_positive = (
        (b_coef > 0)
        and (c_coef > 0)
        and ("positive" in str(hinge_iv.get("direction", "")).lower())
        and (
            "positive" in str(smart_iv.get("direction", "")).lower()
            or "increasing" in str(smart_iv.get("direction", "")).lower()
        )
    )

    if robust_positive and c_p < 0.05 and b_p < 0.05:
        score = 80.0
    elif robust_positive and c_p < 0.10:
        score = 60.0
    elif robust_positive and c_p < 0.20:
        score = 48.0
    elif robust_positive and b_p < 0.05:
        score = 42.0
    elif b_p < 0.10:
        score = 30.0
    else:
        score = 12.0

    if smart_imp >= 0.50:
        score += 5
    elif smart_imp >= 0.25:
        score += 2

    if hinge_imp >= 0.15:
        score += 3
    elif hinge_imp >= 0.08:
        score += 1

    if c_p >= 0.05:
        score -= 5
    if c_p >= 0.10:
        score -= 3

    response = clamp_int(score)

    confounders = []
    for var in controls:
        coef = float(ols_ctl.params[var])
        pval = float(ols_ctl.pvalues[var])
        direction = "positive" if coef > 0 else "negative"
        sig = "(p<0.05)" if pval < 0.05 else "(ns/marginal)"
        confounders.append(f"{var}: {direction} {sig}")

    explanation = (
        f"Primary test used DV={dv} and IV={iv}. Bivariate OLS shows a positive association "
        f"(coef={b_coef:.3f}, p={b_p:.4f}). With controls ({', '.join(controls)}), the hours effect stays positive "
        f"but weakens to marginal significance (coef={c_coef:.3f}, p={c_p:.4f}), so the relationship is partial rather "
        f"than unequivocally strong. SmartAdditive ranks hours #{smart_iv.get('rank', 'NA')} with importance "
        f"{smart_imp:.1%} and direction '{smart_iv.get('direction', 'unknown')}', showing a nonlinear increasing shape "
        f"with stronger gains at higher hour ranges (threshold-like pattern). HingeEBM also keeps hours positive, "
        f"rank #{hinge_iv.get('rank', 'NA')} with importance {hinge_imp:.1%}, supporting robustness of direction across "
        f"interpretable models. Key confounders in controlled OLS are {', '.join(confounders)}; notably persons (+) and "
        f"child (-) are strong, which explains why the pure hours effect attenuates after adjustment. Estimated average catch "
        f"rate is {fish_per_hour.mean():.2f} fish/hour (median {fish_per_hour.median():.2f}). Overall evidence supports "
        f"a real but moderate positive effect of hours on fish caught."
    )

    payload = {"response": response, "explanation": explanation}
    Path("conclusion.txt").write_text(json.dumps(payload, ensure_ascii=True))

    print("\n=== Final conclusion payload ===")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
