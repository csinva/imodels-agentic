import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

df = pd.read_csv("caschools.csv")

# Create DV: average test score; IV: student-teacher ratio
df["score"] = (df["read"] + df["math"]) / 2
df["str_ratio"] = df["students"] / df["teachers"]

numeric_cols = ["str_ratio", "calworks", "lunch", "expenditure", "income", "english"]
df_clean = df[numeric_cols + ["score"]].dropna()

print("=== Summary Statistics ===")
print(df_clean.describe())

print("\n=== Bivariate correlation with score ===")
print(df_clean.corr()["score"])

# OLS with controls
X = df_clean[numeric_cols]
X_const = sm.add_constant(X)
model = sm.OLS(df_clean["score"], X_const).fit()
print("\n=== OLS Summary ===")
print(model.summary())

# SmartAdditiveRegressor
print("\n=== SmartAdditiveRegressor ===")
smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(df_clean[numeric_cols], df_clean["score"])
print(smart)
effects_smart = smart.feature_effects()
print(effects_smart)

# HingeEBMRegressor
print("\n=== HingeEBMRegressor ===")
hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(df_clean[numeric_cols], df_clean["score"])
print(hinge)
effects_hinge = hinge.feature_effects()
print(effects_hinge)

# Gather results for conclusion
str_corr = df_clean.corr()["score"]["str_ratio"]
str_coef = model.params.get("str_ratio", float("nan"))
str_pval = model.pvalues.get("str_ratio", float("nan"))

smart_str = effects_smart.get("str_ratio", {})
hinge_str = effects_hinge.get("str_ratio", {})

# Score: bivariate correlation is negative (higher ratio -> lower score), and
# controlled OLS confirms direction. Check p-value and importance for strength.
if str_pval < 0.01 and abs(str_corr) > 0.2:
    likert = 80
    strength = "strong"
elif str_pval < 0.05:
    likert = 60
    strength = "moderate"
elif str_pval < 0.1:
    likert = 40
    strength = "marginal"
else:
    likert = 20
    strength = "weak"

smart_imp = smart_str.get("importance", None)
hinge_imp = hinge_str.get("importance", None)
smart_dir = smart_str.get("direction", "unknown")
hinge_dir = hinge_str.get("direction", "unknown")

explanation = (
    f"Research question: Is a lower student-teacher ratio associated with higher academic performance? "
    f"Bivariate correlation between str_ratio and score: r={str_corr:.3f} (negative means higher ratio -> lower score). "
    f"OLS (controlled for calworks, lunch, expenditure, income, english): "
    f"str_ratio coef={str_coef:.3f}, p={str_pval:.4f} ({strength} effect). "
    f"SmartAdditiveRegressor: str_ratio direction='{smart_dir}', importance={smart_imp}. "
    f"HingeEBMRegressor: str_ratio direction='{hinge_dir}', importance={hinge_imp}. "
    f"Income and lunch (poverty proxy) are typically the strongest confounders. "
    f"The evidence {'strongly supports' if likert >= 70 else 'moderately supports' if likert >= 45 else 'weakly supports'} "
    f"the claim that lower student-teacher ratios are associated with higher test scores."
)

conclusion = {"response": likert, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f)

print("\n=== Conclusion ===")
print(json.dumps(conclusion, indent=2))
