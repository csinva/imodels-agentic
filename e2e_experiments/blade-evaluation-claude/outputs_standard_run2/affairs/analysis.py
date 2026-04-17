import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("affairs.csv")
print(df.head())
print(df.describe())
print(df["children"].value_counts())

# Basic group comparison
children_yes = df[df["children"] == "yes"]["affairs"]
children_no = df[df["children"] == "no"]["affairs"]

print(f"\nMean affairs (children=yes): {children_yes.mean():.4f}")
print(f"Mean affairs (children=no):  {children_no.mean():.4f}")
print(f"Median affairs (children=yes): {children_yes.median():.4f}")
print(f"Median affairs (children=no):  {children_no.median():.4f}")

# Mann-Whitney U test (non-parametric, since affairs is skewed/ordinal)
stat_mw, p_mw = stats.mannwhitneyu(children_yes, children_no, alternative="two-sided")
print(f"\nMann-Whitney U: stat={stat_mw:.2f}, p={p_mw:.4f}")

# t-test
stat_t, p_t = stats.ttest_ind(children_yes, children_no)
print(f"t-test: stat={stat_t:.4f}, p={p_t:.4f}")

# OLS regression controlling for confounders
df2 = df.copy()
df2["children_bin"] = (df2["children"] == "yes").astype(int)
df2["gender_bin"] = (df2["gender"] == "male").astype(int)

X = df2[["children_bin", "age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender_bin"]]
X = sm.add_constant(X)
y = df2["affairs"]

ols = sm.OLS(y, X).fit()
print("\nOLS Results:")
print(ols.summary())

children_coef = ols.params["children_bin"]
children_pval = ols.pvalues["children_bin"]
print(f"\nchildren coef: {children_coef:.4f}, p={children_pval:.4f}")

# Ridge regression feature importances
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df2[["children_bin", "age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender_bin"]])
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)
feat_names = ["children_bin", "age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender_bin"]
print("\nRidge coefficients (standardized):")
for name, coef in sorted(zip(feat_names, ridge.coef_), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {name}: {coef:.4f}")

# Decision Tree
dt = DecisionTreeRegressor(max_depth=3, random_state=42)
dt.fit(df2[["children_bin", "age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender_bin"]], y)
print("\nDecision Tree feature importances:")
for name, imp in sorted(zip(feat_names, dt.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {imp:.4f}")

# Conclusion
# children=yes has higher mean affairs (possibly due to confounders like years married)
# OLS controls for confounders
has_children_decreases = children_coef < 0 and children_pval < 0.05
print(f"\nchildren coef negative? {children_coef < 0}")
print(f"p < 0.05? {children_pval < 0.05}")

# The OLS coef for children indicates direction and significance
# Unadjusted: children_yes has slightly higher affairs
# Adjusted: check sign and significance
if children_coef < 0 and children_pval < 0.05:
    response = 75
    explanation = (
        f"After controlling for confounders (age, years married, religiousness, etc.), "
        f"having children is associated with fewer extramarital affairs "
        f"(OLS coef={children_coef:.3f}, p={children_pval:.4f}). "
        f"The effect is statistically significant, supporting that children decrease affairs."
    )
elif children_coef > 0 and children_pval < 0.05:
    response = 15
    explanation = (
        f"After controlling for confounders, having children is associated with MORE extramarital affairs "
        f"(OLS coef={children_coef:.3f}, p={children_pval:.4f}), opposite of the hypothesis."
    )
elif children_pval >= 0.05:
    # Not significant - direction from unadjusted
    if children_yes.mean() < children_no.mean():
        response = 30
        explanation = (
            f"Unadjusted: those with children have slightly fewer affairs (mean {children_yes.mean():.3f} vs {children_no.mean():.3f}), "
            f"but the OLS regression coefficient is not statistically significant (coef={children_coef:.3f}, p={children_pval:.4f}). "
            f"No strong evidence that children decrease affairs."
        )
    else:
        response = 20
        explanation = (
            f"No significant effect of children on affairs (OLS coef={children_coef:.3f}, p={children_pval:.4f}). "
            f"Unadjusted means: children_yes={children_yes.mean():.3f}, children_no={children_no.mean():.3f}. "
            f"Mann-Whitney p={p_mw:.4f}. No evidence children decrease affairs."
        )
else:
    response = 50
    explanation = "Mixed evidence."

print(f"\nResponse: {response}, Explanation: {explanation}")

conclusion = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f)
print("conclusion.txt written.")
