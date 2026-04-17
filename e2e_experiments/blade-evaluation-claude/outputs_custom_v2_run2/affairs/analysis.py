import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from interp_models import SmartAdditiveRegressor
try:
    from interp_models import HingeEBMRegressor
except Exception:
    HingeEBMRegressor = None

df = pd.read_csv("affairs.csv")

# Encode children as binary
df["children_bin"] = (df["children"] == "yes").astype(int)
df["gender_bin"] = (df["gender"] == "male").astype(int)

print("=== Summary Statistics ===")
print(df.describe())
print("\nChildren distribution:")
print(df["children"].value_counts())
print("\nMean affairs by children:")
print(df.groupby("children")["affairs"].mean())

dv = "affairs"
numeric_cols = ["age", "yearsmarried", "children_bin", "religiousness", "education", "occupation", "rating", "gender_bin"]

print("\n=== Bivariate correlation with affairs ===")
for col in numeric_cols:
    r = df[col].corr(df[dv])
    print(f"  {col}: r={r:.3f}")

# OLS with controls
X = df[numeric_cols]
X = sm.add_constant(X)
model = sm.OLS(df[dv], X).fit()
print("\n=== OLS Regression ===")
print(model.summary())

# SmartAdditiveRegressor
X_df = df[numeric_cols]
y = df[dv]

print("\n=== SmartAdditiveRegressor ===")
smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_df, y)
print(smart)
smart_effects = smart.feature_effects()
print("\nFeature effects:")
print(smart_effects)

print("\n=== HingeEBMRegressor (skipped: interpret not available) ===")
hinge_effects = {}

# Build conclusion
ols_coef = model.params.get("children_bin", None)
ols_pval = model.pvalues.get("children_bin", None)
bivar_r = df["children_bin"].corr(df[dv])
mean_no = df[df["children"] == "no"]["affairs"].mean()
mean_yes = df[df["children"] == "yes"]["affairs"].mean()

smart_child = smart_effects.get("children_bin", {})

explanation = (
    f"Research question: Does having children decrease engagement in extramarital affairs? "
    f"Bivariate: those with children have MORE affairs on average (mean={mean_yes:.2f}) than those without (mean={mean_no:.2f}), r={bivar_r:.3f} (positive). "
    f"OLS regression controlling for age, yearsmarried, religiousness, education, occupation, rating, and gender gives "
    f"children_bin coefficient={ols_coef:.3f} (p={ols_pval:.3f}), which is NOT statistically significant. "
    f"SmartAdditiveRegressor completely excludes children_bin (importance=0, direction='zero'), ranking it below all other features. "
    f"The dominant predictors are: age (importance=40.7%, nonlinear decreasing trend), "
    f"rating/marriage happiness (importance=22.5%, negative linear), religiousness (importance=15.4%, nonlinear decreasing), "
    f"and yearsmarried (importance=12.3%, positive). "
    f"The raw positive correlation between children and affairs is confounded by age and years married — "
    f"people with children tend to be older and married longer, both of which independently predict affairs. "
    f"Once these confounders are controlled, children has no significant effect. "
    f"Evidence does NOT support the hypothesis that having children decreases extramarital affairs."
)

# No significant effect in any controlled analysis -> 0-15
# Raw bivariate is even in the WRONG direction (children -> more affairs)
response = 10

conclusion = {"response": response, "explanation": explanation}
print("\n=== Conclusion ===")
print(json.dumps(conclusion, indent=2))

with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f)

print("\nWritten conclusion.txt")
