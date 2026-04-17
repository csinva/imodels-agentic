import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv("mortgage.csv")
print("Shape:", df.shape)
print(df.describe())
print("\nGender breakdown:")
print(df.groupby("female")["accept"].agg(["mean", "count"]))

# Bivariate correlation
print("\nCorrelation female <-> accept:", df["female"].corr(df["accept"]))

# OLS with controls
feature_cols = ["female", "black", "housing_expense_ratio", "self_employed",
                "married", "mortgage_credit", "consumer_credit", "bad_history",
                "PI_ratio", "loan_to_value", "denied_PMI"]
df_clean = df[feature_cols + ["accept"]].dropna()
X = sm.add_constant(df_clean[feature_cols])
model = sm.OLS(df_clean["accept"], X).fit()
print(model.summary())

female_coef = model.params["female"]
female_pval = model.pvalues["female"]
print(f"\nOLS female coef={female_coef:.4f}, p={female_pval:.4f}")

# Interpretable models
try:
    from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

    numeric_cols = feature_cols
    X_df = df_clean[numeric_cols]
    y = df_clean["accept"]

    smart = SmartAdditiveRegressor(n_rounds=200)
    smart.fit(X_df, y)
    print("\n=== SmartAdditiveRegressor ===")
    print(smart)
    smart_effects = smart.feature_effects()
    print(smart_effects)

    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_df, y)
    print("\n=== HingeEBMRegressor ===")
    print(hinge)
    hinge_effects = hinge.feature_effects()
    print(hinge_effects)

    smart_female = smart_effects.get("female", {})
    hinge_female = hinge_effects.get("female", {})
    smart_importance = smart_female.get("importance", 0)
    smart_direction = smart_female.get("direction", "unknown")
    hinge_importance = hinge_female.get("importance", 0)
    hinge_direction = hinge_female.get("direction", "unknown")

    # Rank features by SmartAdditive importance
    ranked = sorted(smart_effects.items(), key=lambda x: x[1].get("importance", 0), reverse=True)
    print("\nFeature importance ranking (SmartAdditive):")
    for feat, info in ranked:
        print(f"  {feat}: {info.get('importance', 0):.3f} ({info.get('direction', '')})")

    interp_info = (
        f"SmartAdditive: female importance={smart_importance:.3f}, direction={smart_direction}. "
        f"HingeEBM: female importance={hinge_importance:.3f}, direction={hinge_direction}. "
        f"Top features by SmartAdditive: {', '.join([f[0] for f in ranked[:5]])}."
    )
except Exception as e:
    print(f"Interp models error: {e}")
    smart_importance = 0
    hinge_importance = 0
    smart_direction = "unknown"
    interp_info = f"Interpretable models unavailable: {e}"

# Score
if female_pval < 0.05:
    if abs(female_coef) > 0.05:
        score = 70
    else:
        score = 50
else:
    if female_pval < 0.1:
        score = 35
    else:
        score = 15

# Adjust for interp model importance
avg_imp = (smart_importance + hinge_importance) / 2
if avg_imp > 0.1:
    score = min(score + 10, 100)
elif avg_imp < 0.02:
    score = max(score - 10, 0)

bivar_diff = df.groupby("female")["accept"].mean().get(1.0, 0.5) - df.groupby("female")["accept"].mean().get(0.0, 0.5)
print(f"\nBivariate accept rate diff (female - male): {bivar_diff:.4f}")

explanation = (
    f"Gender (female) effect on mortgage approval: OLS coefficient={female_coef:.4f}, p={female_pval:.4f}. "
    f"Bivariate accept rate difference (female minus male)={bivar_diff:.4f}. "
    f"{interp_info} "
    f"After controlling for creditworthiness variables (credit history, debt ratios, loan-to-value, etc.), "
    f"the female coefficient {'is statistically significant' if female_pval < 0.05 else 'is not statistically significant'} "
    f"({'positive' if female_coef > 0 else 'negative'} direction). "
    f"Key confounders include mortgage_credit, consumer_credit, bad_history, PI_ratio, and denied_PMI."
)

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)
print(f"\nconclusion.txt written: score={score}")
print(explanation)
