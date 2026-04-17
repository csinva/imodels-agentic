import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load data
df = pd.read_csv('mortgage.csv')
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# Check columns
print("\nColumns:", df.columns.tolist())

# Target: deny (1=denied, 0=accepted)
# Gender: female (1=female, 0=male)

print("\n--- Denial rates by gender ---")
denial_by_gender = df.groupby('female')['deny'].agg(['mean', 'count', 'sum'])
denial_by_gender.index = ['Male', 'Female']
print(denial_by_gender)

female_denied = df[df['female'] == 1]['deny']
male_denied = df[df['female'] == 0]['deny']

print(f"\nFemale denial rate: {female_denied.mean():.4f} (n={len(female_denied)})")
print(f"Male denial rate: {male_denied.mean():.4f} (n={len(male_denied)})")

# Chi-square test
ct = pd.crosstab(df['female'], df['deny'])
chi2, p_chi2, dof, expected = stats.chi2_contingency(ct)
print(f"\nChi-square test: chi2={chi2:.4f}, p={p_chi2:.4f}")

# T-test
t_stat, p_ttest = stats.ttest_ind(female_denied, male_denied)
print(f"T-test: t={t_stat:.4f}, p={p_ttest:.4f}")

# Simple logistic regression: female -> deny
X_simple = df[['female']].dropna()
y_simple = df.loc[X_simple.index, 'deny']
X_sm = sm.add_constant(X_simple)
logit_simple = sm.Logit(y_simple, X_sm).fit(disp=0)
print("\n--- Simple Logistic Regression (female -> deny) ---")
print(logit_simple.summary2())

# Multivariate logistic regression controlling for confounders
features = ['female', 'black', 'housing_expense_ratio', 'self_employed',
            'married', 'mortgage_credit', 'consumer_credit', 'bad_history',
            'PI_ratio', 'loan_to_value', 'denied_PMI']
df_clean = df[features + ['deny']].dropna()
X_multi = sm.add_constant(df_clean[features])
logit_multi = sm.Logit(df_clean['deny'], X_multi).fit(disp=0)
print("\n--- Multivariate Logistic Regression ---")
print(logit_multi.summary2())

female_coef = logit_multi.params['female']
female_pval = logit_multi.pvalues['female']
print(f"\nFemale coefficient: {female_coef:.4f}, p-value: {female_pval:.4f}")

# Decision tree for interpretability
X_dt = df_clean[features]
y_dt = df_clean['deny']
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_dt, y_dt)
feat_imp = pd.Series(dt.feature_importances_, index=features).sort_values(ascending=False)
print("\n--- Decision Tree Feature Importances ---")
print(feat_imp)

# Summary
print("\n--- Summary ---")
print(f"Female denial rate: {female_denied.mean():.4f}")
print(f"Male denial rate: {male_denied.mean():.4f}")
print(f"Difference: {female_denied.mean() - male_denied.mean():.4f}")
print(f"Chi-square p-value: {p_chi2:.4f}")
print(f"Logit (simple) female p-value: {logit_simple.pvalues['female']:.4f}")
print(f"Logit (multi) female p-value: {female_pval:.4f}")
print(f"Logit (multi) female OR: {np.exp(female_coef):.4f}")

# Determine response score
# If female coefficient is negative (lower denial for females) and significant -> effect exists
# Check direction and significance
simple_pval = logit_simple.pvalues['female']
multi_pval = female_pval

# Female denial rate vs male
female_rate = female_denied.mean()
male_rate = male_denied.mean()
diff = female_rate - male_rate

# Build explanation
direction = "lower" if diff < 0 else "higher"
sig_simple = simple_pval < 0.05
sig_multi = multi_pval < 0.05

explanation = (
    f"Female denial rate: {female_rate:.3f}, Male denial rate: {male_rate:.3f} "
    f"(difference: {diff:.3f}, females have {direction} denial rates). "
    f"Chi-square p={p_chi2:.4f}. "
    f"Simple logistic regression: female coef={logit_simple.params['female']:.3f}, p={simple_pval:.4f}. "
    f"Multivariate logistic (controlling for credit, income, race, etc.): female coef={female_coef:.3f}, OR={np.exp(female_coef):.3f}, p={multi_pval:.4f}. "
    f"Decision tree: female feature importance={feat_imp['female']:.4f}. "
)

if sig_simple and not sig_multi:
    explanation += ("Simple regression shows a significant effect but it disappears after controlling for confounders, "
                    "suggesting the raw gender gap is explained by other variables.")
    score = 30
elif sig_multi:
    if female_coef < 0:
        explanation += "Gender has a significant effect even after controlling for confounders: females are less likely to be denied."
        score = 70
    else:
        explanation += "Gender has a significant effect even after controlling for confounders: females are more likely to be denied."
        score = 70
else:
    explanation += ("No statistically significant effect of gender on mortgage denial after controlling for confounders. "
                    "The raw difference is small and not significant in multivariate analysis.")
    score = 20

print(f"\nFinal score: {score}")
print(f"Explanation: {explanation}")

# Write conclusion
conclusion = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)
print("\nWritten conclusion.txt")
