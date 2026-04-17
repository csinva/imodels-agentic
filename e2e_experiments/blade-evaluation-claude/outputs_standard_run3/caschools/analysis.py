import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor

# Load data
df = pd.read_csv("caschools.csv")
print("Shape:", df.shape)
print(df.head())

# Compute student-teacher ratio
df["str"] = df["students"] / df["teachers"]

# Academic performance: average of read and math
df["score"] = (df["read"] + df["math"]) / 2

print("\nSummary stats for str and score:")
print(df[["str", "score"]].describe())

# Pearson correlation
r, p = stats.pearsonr(df["str"], df["score"])
print(f"\nPearson r(str, score) = {r:.4f}, p = {p:.4e}")

# Spearman correlation
rs, ps = stats.spearmanr(df["str"], df["score"])
print(f"Spearman r(str, score) = {rs:.4f}, p = {ps:.4e}")

# OLS regression: score ~ str (bivariate)
X_simple = sm.add_constant(df["str"])
ols_simple = sm.OLS(df["score"], X_simple).fit()
print("\n--- Bivariate OLS: score ~ str ---")
print(ols_simple.summary())

# OLS regression with controls (income, lunch, english)
controls = ["str", "income", "lunch", "english", "expenditure"]
X_ctrl = sm.add_constant(df[controls].dropna())
y_ctrl = df.loc[X_ctrl.index, "score"]
ols_ctrl = sm.OLS(y_ctrl, X_ctrl).fit()
print("\n--- Controlled OLS: score ~ str + controls ---")
print(ols_ctrl.summary())

# Decision tree feature importance
feat_cols = ["str", "income", "lunch", "english", "expenditure", "calworks"]
dt = DecisionTreeRegressor(max_depth=4, random_state=42)
dt.fit(df[feat_cols].fillna(df[feat_cols].mean()), df["score"])
print("\nDecision Tree feature importances:")
for col, imp in sorted(zip(feat_cols, dt.feature_importances_), key=lambda x: -x[1]):
    print(f"  {col}: {imp:.4f}")

# Summary
bivariate_coef = ols_simple.params["str"]
bivariate_p = ols_simple.pvalues["str"]
controlled_coef = ols_ctrl.params["str"]
controlled_p = ols_ctrl.pvalues["str"]

print(f"\nBivariate: str coef = {bivariate_coef:.4f}, p = {bivariate_p:.4e}")
print(f"Controlled: str coef = {controlled_coef:.4f}, p = {controlled_p:.4e}")

# Determine response
# Negative coef for str means higher ratio -> lower score (lower ratio -> higher score)
# We look at significance and direction
if bivariate_p < 0.05 and bivariate_coef < 0:
    bivariate_yes = True
else:
    bivariate_yes = False

if controlled_p < 0.05 and controlled_coef < 0:
    controlled_yes = True
else:
    controlled_yes = False

print(f"\nBivariate significant & negative: {bivariate_yes}")
print(f"Controlled significant & negative: {controlled_yes}")

# Score: strong yes if both significant, moderate yes if only bivariate
if bivariate_yes and controlled_yes:
    response = 80
    explanation = (
        f"The student-teacher ratio (STR) is significantly negatively associated with academic performance "
        f"both in bivariate analysis (r={r:.3f}, p={bivariate_p:.2e}, coef={bivariate_coef:.3f}) and after "
        f"controlling for income, lunch, English learners, and expenditure (coef={controlled_coef:.3f}, p={controlled_p:.2e}). "
        f"Lower STR is associated with higher test scores, supporting the research hypothesis."
    )
elif bivariate_yes and not controlled_yes:
    response = 45
    explanation = (
        f"The STR is significantly negatively correlated with test scores in bivariate analysis (r={r:.3f}, p={bivariate_p:.2e}), "
        f"but the effect is not significant after controlling for socioeconomic factors (p={controlled_p:.2e}). "
        f"The association may be confounded by income and poverty."
    )
else:
    response = 20
    explanation = (
        f"No significant association between STR and academic performance was found (r={r:.3f}, p={bivariate_p:.2e})."
    )

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written:")
print(json.dumps(result, indent=2))
