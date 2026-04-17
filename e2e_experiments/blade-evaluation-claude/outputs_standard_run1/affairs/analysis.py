import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeRegressor
import json

df = pd.read_csv("affairs.csv")

# Encode children
df['children_bin'] = (df['children'] == 'yes').astype(int)
df['gender_bin'] = (df['gender'] == 'male').astype(int)

# Summary stats by children
yes = df[df['children'] == 'yes']['affairs']
no = df[df['children'] == 'no']['affairs']
print(f"Children=yes: mean={yes.mean():.3f}, n={len(yes)}")
print(f"Children=no:  mean={no.mean():.3f}, n={len(no)}")

# t-test
t, p = stats.ttest_ind(yes, no)
print(f"t-test: t={t:.3f}, p={p:.4f}")

# Mann-Whitney (non-parametric, better for skewed/discrete)
u, p_mw = stats.mannwhitneyu(yes, no, alternative='less')
print(f"Mann-Whitney (children<no): U={u:.1f}, p={p_mw:.4f}")

# OLS controlling for confounders
X = df[['children_bin','age','yearsmarried','religiousness','education','occupation','rating','gender_bin']]
X = sm.add_constant(X)
y = df['affairs']
model = sm.OLS(y, X).fit()
print(model.summary())

children_coef = model.params['children_bin']
children_pval = model.pvalues['children_bin']
print(f"\nOLS children coef={children_coef:.3f}, p={children_pval:.4f}")

# Decision tree for feature importance
features = ['children_bin','age','yearsmarried','religiousness','education','occupation','rating','gender_bin']
Xf = df[features].values
yf = df['affairs'].values
dt = DecisionTreeRegressor(max_depth=4, random_state=42)
dt.fit(Xf, yf)
importances = dict(zip(features, dt.feature_importances_))
print("\nDecision tree feature importances:")
for k, v in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:.3f}")

# Conclusion
# Children have HIGHER mean affairs (1.44 vs 0.84), but this is likely confounded by years married / age
# In controlled OLS, check sign and significance of children_bin
if children_pval < 0.05:
    if children_coef < 0:
        response = 70
        explanation = (f"Controlling for age, years married, religiousness, education, occupation, "
                       f"marriage rating, and gender, having children is associated with significantly "
                       f"fewer affairs (coef={children_coef:.3f}, p={children_pval:.4f}). "
                       f"The unadjusted means (children=yes: {yes.mean():.2f}, no: {no.mean():.2f}) "
                       f"suggest confounding by years married. After controlling, children decreases affairs.")
    else:
        response = 25
        explanation = (f"Having children is statistically significant but associated with MORE affairs "
                       f"(coef={children_coef:.3f}, p={children_pval:.4f}) after controlling for confounders. "
                       f"Does not support the hypothesis that children decrease affairs.")
else:
    raw_diff = yes.mean() - no.mean()
    if raw_diff < 0:
        response = 40
        explanation = (f"Having children shows a raw trend toward fewer affairs (children=yes: {yes.mean():.2f}, "
                       f"no: {no.mean():.2f}), but the OLS controlled coefficient is not statistically significant "
                       f"(coef={children_coef:.3f}, p={children_pval:.4f}). Weak/no evidence.")
    else:
        response = 20
        explanation = (f"Having children does not decrease affairs. Unadjusted means: yes={yes.mean():.2f}, "
                       f"no={no.mean():.2f}. After controlling for confounders, the coefficient is "
                       f"not significant (coef={children_coef:.3f}, p={children_pval:.4f}).")

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)
print(f"\nConclusion: response={response}")
print(f"Explanation: {explanation}")
