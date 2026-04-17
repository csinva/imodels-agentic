import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('mortgage.csv')
# Drop rows where key variables are NaN
df = df.dropna(subset=['female', 'deny'])
print("Shape after dropping NaN:", df.shape)
print(df.describe())

# Approval rates by gender
female_deny = df[df['female'] == 1]['deny'].mean()
male_deny = df[df['female'] == 0]['deny'].mean()
print(f"\nDenial rate - Female: {female_deny:.4f}, Male: {male_deny:.4f}")
print(f"Acceptance rate - Female: {1-female_deny:.4f}, Male: {1-male_deny:.4f}")

# Chi-square test: gender vs deny
ct = pd.crosstab(df['female'], df['deny'])
print("\nCrosstab female vs deny:\n", ct)
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)
print(f"Chi2={chi2:.4f}, p={p_chi2:.4f}")

# T-test on deny rates
female_deny_vals = df[df['female'] == 1]['deny']
male_deny_vals = df[df['female'] == 0]['deny']
t_stat, p_ttest = stats.ttest_ind(female_deny_vals, male_deny_vals)
print(f"T-test: t={t_stat:.4f}, p={p_ttest:.4f}")

# OLS regression: deny ~ female
X_simple = sm.add_constant(df[['female']])
ols_simple = sm.OLS(df['deny'], X_simple).fit()
print("\nSimple OLS (deny ~ female):")
print(ols_simple.summary())

# Multiple regression controlling for creditworthiness variables
controls = ['housing_expense_ratio', 'self_employed', 'married', 'mortgage_credit',
            'consumer_credit', 'bad_history', 'PI_ratio', 'loan_to_value', 'denied_PMI', 'black']
X_multi = sm.add_constant(df[['female'] + controls].dropna())
y_multi = df.loc[X_multi.index, 'deny']
ols_multi = sm.OLS(y_multi, X_multi).fit()
print("\nMultiple OLS (deny ~ female + controls):")
print(ols_multi.summary())

female_coef = ols_multi.params['female']
female_pval = ols_multi.pvalues['female']
print(f"\nFemale coefficient: {female_coef:.4f}, p-value: {female_pval:.4f}")

# Logistic regression
lr = LogisticRegression(max_iter=1000)
X_lr = df[['female'] + controls].dropna()
y_lr = df.loc[X_lr.index, 'deny']
lr.fit(X_lr, y_lr)
female_idx = X_lr.columns.tolist().index('female')
print(f"\nLogistic regression female coef: {lr.coef_[0][female_idx]:.4f}")

# Summary
print(f"\n=== SUMMARY ===")
print(f"Unadjusted denial rate difference: female={female_deny:.4f}, male={male_deny:.4f}, diff={female_deny-male_deny:.4f}")
print(f"Chi-square p-value: {p_chi2:.4f}")
print(f"Simple OLS female coef: {ols_simple.params['female']:.4f}, p={ols_simple.pvalues['female']:.4f}")
print(f"Adjusted OLS female coef: {female_coef:.4f}, p={female_pval:.4f}")

# Determine response
# Unadjusted: females have lower denial rate (negative effect of female on denial)
# Check if significant after controls
if female_pval < 0.05:
    if female_coef < 0:
        # Female associated with lower denial (more approvals)
        response = 65
        explanation = (f"Gender does affect mortgage approval. Females have a lower denial rate ({female_deny:.3f}) "
                      f"vs males ({male_deny:.3f}). After controlling for creditworthiness variables (credit history, "
                      f"debt ratios, etc.), the female coefficient is {female_coef:.4f} (p={female_pval:.4f}), "
                      f"indicating females are significantly more likely to be approved even after controls. "
                      f"This suggests gender has a statistically significant effect on mortgage approval.")
    else:
        response = 65
        explanation = (f"Gender does affect mortgage approval. The female coefficient is {female_coef:.4f} "
                      f"(p={female_pval:.4f}) in the adjusted model, indicating a significant gender effect.")
else:
    response = 30
    explanation = (f"After controlling for creditworthiness, gender (female) does not significantly predict "
                  f"denial (coef={female_coef:.4f}, p={female_pval:.4f}). Unadjusted denial rates differ "
                  f"(female={female_deny:.3f}, male={male_deny:.3f}), but this disappears after controlling "
                  f"for financial factors, suggesting the raw difference is explained by other variables.")

print(f"\nResponse: {response}")
print(f"Explanation: {explanation}")

with open('conclusion.txt', 'w') as f:
    json.dump({"response": response, "explanation": explanation}, f)

print("\nconclustion.txt written.")
