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

# Compute student-teacher ratio
df["str"] = df["students"] / df["teachers"]

# Composite test score
df["score"] = (df["read"] + df["math"]) / 2

print("\nCorrelation of str with read, math, score:")
for col in ["read", "math", "score"]:
    r, p = stats.pearsonr(df["str"], df[col])
    print(f"  {col}: r={r:.4f}, p={p:.4e}")

# OLS regression: score ~ str (simple)
X_simple = sm.add_constant(df["str"])
model_simple = sm.OLS(df["score"], X_simple).fit()
print("\nSimple OLS (score ~ str):")
print(model_simple.summary().tables[1])

# OLS with controls: score ~ str + income + lunch + english + expenditure
controls = ["str", "income", "lunch", "english", "expenditure"]
X_ctrl = sm.add_constant(df[controls])
model_ctrl = sm.OLS(df["score"], X_ctrl).fit()
print("\nMultiple OLS (score ~ str + controls):")
print(model_ctrl.summary().tables[1])

# Decision tree to see feature importance
feature_cols = ["str", "income", "lunch", "english", "expenditure", "calworks"]
X_tree = df[feature_cols].values
y_tree = df["score"].values
tree = DecisionTreeRegressor(max_depth=4, random_state=0)
tree.fit(X_tree, y_tree)
importances = pd.Series(tree.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nDecision Tree feature importances:")
print(importances)

# Split into low vs high STR groups (median split)
median_str = df["str"].median()
low_str = df[df["str"] <= median_str]["score"]
high_str = df[df["str"] > median_str]["score"]
t_stat, t_p = stats.ttest_ind(low_str, high_str)
print(f"\nMedian STR split t-test: low_str mean={low_str.mean():.2f}, high_str mean={high_str.mean():.2f}")
print(f"  t={t_stat:.4f}, p={t_p:.4e}")

# Summary
str_coef = model_simple.params["str"]
str_pval = model_simple.pvalues["str"]
str_coef_ctrl = model_ctrl.params["str"]
str_pval_ctrl = model_ctrl.pvalues["str"]

r_simple, p_simple = stats.pearsonr(df["str"], df["score"])

print(f"\nSimple regression: coef={str_coef:.4f}, p={str_pval:.4e}")
print(f"Controlled regression: coef={str_coef_ctrl:.4f}, p={str_pval_ctrl:.4e}")

# Interpretation:
# Simple bivariate: significant negative association (lower STR -> higher scores)
# Controlled: NOT significant (p=0.35) — confounded by income, lunch eligibility (socioeconomic factors)
# Decision tree gives STR 0% importance when socioeconomic vars are available
# The bivariate association exists (p<0.001) but is largely explained by confounders.
# A moderate "Yes" score (~65) acknowledges the raw association while noting confounding.

explanation = (
    f"There is a statistically significant bivariate association between lower student-teacher ratio (STR) "
    f"and higher academic performance: r={r_simple:.3f}, p={p_simple:.2e}. "
    f"Simple OLS: each unit increase in STR is associated with a {abs(str_coef):.2f}-point decrease in "
    f"composite test scores (p={str_pval:.2e}). A median-split t-test confirms significantly higher scores "
    f"in low-STR districts (mean={low_str.mean():.1f} vs {high_str.mean():.1f}, p={t_p:.2e}). "
    f"However, after controlling for income, free-lunch eligibility, English learner %, and expenditure, "
    f"the STR coefficient becomes small and non-significant (coef={str_coef_ctrl:.3f}, p={str_pval_ctrl:.2f}). "
    f"Decision tree analysis gives STR 0% importance when socioeconomic variables are included. "
    f"This indicates the raw association is largely mediated by socioeconomic confounders. "
    f"There IS a significant association in the raw data, but it is not an independent predictor once "
    f"socioeconomic factors are accounted for."
)

# Score: bivariate association is significant -> Yes, but effect is confounded -> moderate Yes (~65)
response = 65

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
print(json.dumps(result, indent=2))
