import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json

df = pd.read_csv('affairs.csv')
df['children_binary'] = (df['children'] == 'yes').astype(int)

# Basic stats
no_children = df[df['children'] == 'no']['affairs']
yes_children = df[df['children'] == 'yes']['affairs']

print(f"No children - mean affairs: {no_children.mean():.3f}, n={len(no_children)}")
print(f"Has children - mean affairs: {yes_children.mean():.3f}, n={len(yes_children)}")

# T-test
t_stat, p_val = stats.ttest_ind(no_children, yes_children)
print(f"T-test: t={t_stat:.3f}, p={p_val:.4f}")

# Mann-Whitney U (more robust for non-normal data)
u_stat, p_mw = stats.mannwhitneyu(no_children, yes_children, alternative='two-sided')
print(f"Mann-Whitney U: U={u_stat:.1f}, p={p_mw:.4f}")

# OLS regression controlling for confounders
df['gender_binary'] = (df['gender'] == 'male').astype(int)
X = df[['children_binary', 'age', 'yearsmarried', 'religiousness', 'education', 'occupation', 'rating', 'gender_binary']]
X = sm.add_constant(X)
y = df['affairs']
model = sm.OLS(y, X).fit()
print(model.summary())

children_coef = model.params['children_binary']
children_pval = model.pvalues['children_binary']
print(f"\nChildren coefficient: {children_coef:.3f}, p={children_pval:.4f}")

# Proportion with any affairs
prop_no = (no_children > 0).mean()
prop_yes = (yes_children > 0).mean()
print(f"\nProportion having any affairs - no children: {prop_no:.3f}, yes children: {prop_yes:.3f}")

# Chi-square test on binary affairs
cont_table = pd.crosstab(df['children'], df['affairs'] > 0)
chi2, p_chi2, dof, _ = stats.chi2_contingency(cont_table)
print(f"Chi-square: chi2={chi2:.3f}, p={p_chi2:.4f}")

# Conclusion
# Children with yes have slightly higher mean affairs but we need to check direction
# The question is: does having children DECREASE affairs?
# If yes_children mean > no_children mean, children does NOT decrease affairs

mean_diff = yes_children.mean() - no_children.mean()
print(f"\nMean difference (children - no children): {mean_diff:.3f}")

# Build conclusion
# p-value from t-test and Mann-Whitney
# Direction matters: does having children decrease affairs?
# If children_coef in regression is negative and significant => decreases
# If positive/nonsignificant => does not decrease

if children_pval < 0.05 and children_coef < 0:
    response = 70
    explanation = (f"Having children is significantly associated with FEWER affairs "
                   f"(coef={children_coef:.3f}, p={children_pval:.4f}). "
                   f"Mean affairs: no children={no_children.mean():.3f}, children={yes_children.mean():.3f}. "
                   f"After controlling for confounders, children significantly decreases affair engagement.")
elif children_pval < 0.05 and children_coef > 0:
    response = 20
    explanation = (f"Having children is significantly associated with MORE affairs "
                   f"(coef={children_coef:.3f}, p={children_pval:.4f}). "
                   f"Mean affairs: no children={no_children.mean():.3f}, children={yes_children.mean():.3f}. "
                   f"Children does NOT decrease affairs; if anything it increases them.")
else:
    # Use raw means to decide direction for borderline
    if mean_diff < 0:
        response = 40
        explanation = (f"No significant relationship found (OLS p={children_pval:.4f}). "
                       f"Descriptively, those with children have slightly fewer affairs "
                       f"(mean diff={mean_diff:.3f}), but the effect is not statistically significant. "
                       f"Mann-Whitney p={p_mw:.4f}.")
    else:
        response = 30
        explanation = (f"No significant relationship (OLS p={children_pval:.4f}, MW p={p_mw:.4f}). "
                       f"Those with children have slightly MORE affairs (mean diff={mean_diff:.3f}), "
                       f"so no evidence children decreases affair engagement.")

print(f"\nResponse: {response}")
print(f"Explanation: {explanation}")

with open('conclusion.txt', 'w') as f:
    json.dump({"response": response, "explanation": explanation}, f)

print("conclusion.txt written.")
