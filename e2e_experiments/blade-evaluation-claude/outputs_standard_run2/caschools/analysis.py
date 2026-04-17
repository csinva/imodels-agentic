import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv('caschools.csv')

# Compute student-teacher ratio
df['str'] = df['students'] / df['teachers']

# Academic performance: average of read and math
df['score'] = (df['read'] + df['math']) / 2

print("=== Summary Statistics ===")
print(df[['str', 'read', 'math', 'score']].describe())

# Correlation
r_read, p_read = stats.pearsonr(df['str'], df['read'])
r_math, p_math = stats.pearsonr(df['str'], df['math'])
r_score, p_score = stats.pearsonr(df['str'], df['score'])

print(f"\nCorrelation STR vs read:  r={r_read:.4f}, p={p_read:.4e}")
print(f"Correlation STR vs math:  r={r_math:.4f}, p={p_math:.4e}")
print(f"Correlation STR vs score: r={r_score:.4f}, p={p_score:.4e}")

# Simple OLS: score ~ STR
X_simple = sm.add_constant(df['str'])
model_simple = sm.OLS(df['score'], X_simple).fit()
print("\n=== Simple OLS: score ~ STR ===")
print(model_simple.summary())

# Multiple regression controlling for confounders
controls = ['calworks', 'lunch', 'income', 'english', 'expenditure']
X_full = sm.add_constant(df[['str'] + controls])
model_full = sm.OLS(df['score'], X_full).fit()
print("\n=== Multiple OLS: score ~ STR + controls ===")
print(model_full.summary())

str_coef = model_full.params['str']
str_pval = model_full.pvalues['str']
print(f"\nSTR coefficient (controlled): {str_coef:.4f}, p={str_pval:.4e}")

# Ridge with standardized features for importance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['str'] + controls])
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, df['score'])
feat_names = ['str'] + controls
print("\n=== Ridge coefficients (standardized) ===")
for name, coef in zip(feat_names, ridge.coef_):
    print(f"  {name}: {coef:.4f}")

# Determine conclusion
# Simple regression shows negative correlation (higher STR -> lower scores)
# Multiple regression controls for socioeconomic confounders
simple_sig = p_score < 0.05
multi_sig = str_pval < 0.05
negative_effect = str_coef < 0

print(f"\n=== Conclusion ===")
print(f"Simple regression significant: {simple_sig}")
print(f"Multiple regression significant: {multi_sig}")
print(f"Negative effect (lower STR -> higher scores): {negative_effect}")

if multi_sig and negative_effect:
    response = 75
    explanation = (
        f"Yes, lower student-teacher ratio is associated with higher academic performance. "
        f"Simple correlation: r={r_score:.3f} (p={p_score:.2e}), indicating higher STR reduces scores. "
        f"In multiple regression controlling for socioeconomic factors (calworks, lunch, income, english, expenditure), "
        f"STR coefficient = {str_coef:.3f} (p={str_pval:.2e}), remaining statistically significant. "
        f"This suggests a modest but real negative effect of higher student-teacher ratios on test scores."
    )
elif simple_sig and not multi_sig:
    response = 40
    explanation = (
        f"The simple correlation between STR and scores is significant (r={r_score:.3f}, p={p_score:.2e}), "
        f"but after controlling for socioeconomic confounders the effect becomes non-significant "
        f"(STR coef={str_coef:.3f}, p={str_pval:.2e}), suggesting the relationship is largely confounded."
    )
else:
    response = 20
    explanation = (
        f"No significant relationship found between student-teacher ratio and academic performance. "
        f"Correlation r={r_score:.3f} (p={p_score:.2e})."
    )

result = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print(f"\nconclusion.txt written: response={response}")
print(f"explanation: {explanation}")
