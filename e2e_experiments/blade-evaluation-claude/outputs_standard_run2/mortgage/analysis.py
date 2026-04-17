import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load data
df = pd.read_csv("mortgage.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# Drop rows with NaN in female or deny
df_clean = df.dropna(subset=['female', 'deny'])
print(f"\nRows after dropping NaN in female/deny: {len(df_clean)}")

# Research question: Does gender affect mortgage approval?
# Key variables: female (0=male, 1=female), deny (1=denied, 0=accepted)

print("\n--- Approval rates by gender ---")
female_deny_rate = df_clean[df_clean['female'] == 1]['deny'].mean()
male_deny_rate = df_clean[df_clean['female'] == 0]['deny'].mean()
print(f"Female denial rate: {female_deny_rate:.4f} ({female_deny_rate*100:.2f}%)")
print(f"Male denial rate:   {male_deny_rate:.4f} ({male_deny_rate*100:.2f}%)")
print(f"Difference: {female_deny_rate - male_deny_rate:.4f}")

# Chi-square test
ct = pd.crosstab(df_clean['female'], df_clean['deny'])
print("\nCrosstab:\n", ct)
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)
print(f"\nChi-square: {chi2:.4f}, p-value: {p_chi2:.4f}, dof: {dof}")

# Point-biserial / t-test
female_group = df_clean[df_clean['female'] == 1]['deny']
male_group = df_clean[df_clean['female'] == 0]['deny']
t_stat, p_ttest = stats.ttest_ind(female_group, male_group)
print(f"\nT-test: t={t_stat:.4f}, p={p_ttest:.4f}")

# Logistic regression (unadjusted)
X_simple = df_clean[['female']].copy()
X_simple = sm.add_constant(X_simple)
y = df_clean['deny']
logit_simple = sm.Logit(y, X_simple).fit(disp=0)
print("\n--- Unadjusted Logistic Regression ---")
print(logit_simple.summary2())

# Logistic regression (adjusted for confounders)
controls = ['black', 'housing_expense_ratio', 'self_employed', 'married',
            'mortgage_credit', 'consumer_credit', 'bad_history', 'PI_ratio',
            'loan_to_value', 'denied_PMI']
X_full = df[['female'] + controls].copy().dropna()
y_full = df.loc[X_full.index, 'deny']
y_full = y_full.dropna()
X_full = X_full.loc[y_full.index]
X_full_const = sm.add_constant(X_full)
logit_full = sm.Logit(y_full, X_full_const).fit(disp=0)
print("\n--- Adjusted Logistic Regression ---")
print(logit_full.summary2())

female_coef = logit_full.params['female']
female_pval = logit_full.pvalues['female']
female_or = np.exp(female_coef)
print(f"\nFemale coefficient: {female_coef:.4f}, OR: {female_or:.4f}, p-value: {female_pval:.4f}")

# Decision tree for interpretable rules
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
X_dt = df[['female'] + controls].dropna()
y_dt = df.loc[X_dt.index, 'deny'].dropna()
X_dt = X_dt.loc[y_dt.index]
dt.fit(X_dt, y_dt)
feat_names = ['female'] + controls
importances = dict(zip(feat_names, dt.feature_importances_))
print("\nDecision Tree Feature Importances:")
for k, v in sorted(importances.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v:.4f}")

print(f"\nFemale feature importance in tree: {importances['female']:.4f}")

# Summary
print("\n=== SUMMARY ===")
print(f"Unadjusted denial rate difference (female - male): {female_deny_rate - male_deny_rate:.4f}")
print(f"Chi-square p-value: {p_chi2:.4f}")
print(f"T-test p-value: {p_ttest:.4f}")
print(f"Adjusted logistic regression female p-value: {female_pval:.4f}")
print(f"Adjusted OR for female: {female_or:.4f}")

# Determine response score
# If female coef is negative (females less likely to be denied) and significant -> gender matters
# If not significant -> no strong effect
alpha = 0.05
gender_significant = female_pval < alpha
direction = "females less likely denied" if female_coef < 0 else "females more likely denied"

if gender_significant:
    # Significant effect - direction matters
    # OR < 1 means females less likely denied (favorable treatment)
    # Score 60-80 if significant (moderate to strong effect on gender)
    abs_effect = abs(female_coef)
    score = min(80, 50 + int(abs_effect * 30))
else:
    # Not significant after controls - score 20-35
    score = 25

print(f"\nGender significant: {gender_significant}, Direction: {direction}")
print(f"Score: {score}")

explanation = (
    f"Raw denial rates: females {female_deny_rate*100:.1f}%, males {male_deny_rate*100:.1f}% "
    f"(diff={female_deny_rate-male_deny_rate:.3f}). "
    f"Chi-square p={p_chi2:.4f}. "
    f"After controlling for creditworthiness (black, credit scores, debt ratios, etc.), "
    f"the female coefficient in logistic regression is {female_coef:.4f} (OR={female_or:.3f}), "
    f"p={female_pval:.4f}. "
    f"{'The effect IS statistically significant' if gender_significant else 'The effect is NOT statistically significant'} "
    f"at alpha=0.05. "
    f"Direction: {direction}. "
    f"Gender does appear to have {'a statistically significant' if gender_significant else 'no statistically significant'} "
    f"effect on mortgage denial after controlling for financial characteristics."
)

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written:")
print(json.dumps(result, indent=2))
