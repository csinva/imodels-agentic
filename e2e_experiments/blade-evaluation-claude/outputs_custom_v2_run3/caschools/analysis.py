import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

# Load data
df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'caschools.csv'))

# Create student-teacher ratio and composite test score
df['str_ratio'] = df['students'] / df['teachers']
df['test_score'] = (df['read'] + df['math']) / 2

print("=== Summary Statistics ===")
print(df[['str_ratio', 'test_score', 'income', 'lunch', 'english', 'calworks', 'expenditure']].describe())

print("\n=== Bivariate correlation (str_ratio vs test_score) ===")
corr = df['str_ratio'].corr(df['test_score'])
print(f"Pearson r = {corr:.4f}")

# OLS with controls
numeric_cols = ['str_ratio', 'income', 'lunch', 'english', 'calworks', 'expenditure']
df_clean = df[numeric_cols + ['test_score']].dropna()

X = df_clean[numeric_cols]
X = sm.add_constant(X)
model = sm.OLS(df_clean['test_score'], X).fit()
print("\n=== OLS Regression ===")
print(model.summary())

# SmartAdditiveRegressor
print("\n=== SmartAdditiveRegressor ===")
smart = SmartAdditiveRegressor(n_rounds=200)
X_df = df_clean[numeric_cols]
smart.fit(X_df, df_clean['test_score'])
print(smart)
smart_effects = smart.feature_effects()
print("Feature effects:", smart_effects)

# HingeEBMRegressor
print("\n=== HingeEBMRegressor ===")
hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X_df, df_clean['test_score'])
print(hinge)
hinge_effects = hinge.feature_effects()
print("Feature effects:", hinge_effects)

# Summarize findings
str_ols_coef = model.params.get('str_ratio', None)
str_ols_pval = model.pvalues.get('str_ratio', None)
str_smart = smart_effects.get('str_ratio', {})
str_hinge = hinge_effects.get('str_ratio', {})

print(f"\n=== SUMMARY ===")
print(f"OLS coef for str_ratio: {str_ols_coef:.4f}, p={str_ols_pval:.4f}")
print(f"SmartAdditive: direction={str_smart.get('direction')}, importance={str_smart.get('importance'):.4f}, rank={str_smart.get('rank')}")
print(f"HingeEBM: direction={str_hinge.get('direction')}, importance={str_hinge.get('importance'):.4f}, rank={str_hinge.get('rank')}")

# Build conclusion
# Lower str_ratio -> higher test score means negative OLS coef for str_ratio
# Score: if significant negative effect, score 75-100
if str_ols_pval < 0.05 and str_ols_coef < 0:
    response = 75
    sig_text = "significant negative"
elif str_ols_pval < 0.05 and str_ols_coef > 0:
    response = 20
    sig_text = "significant positive (opposite direction)"
elif str_ols_pval < 0.1:
    response = 45
    sig_text = "marginally significant"
else:
    response = 20
    sig_text = "not significant"

# Boost if both interp models confirm
if str_smart.get('direction', '') in ('negative', 'nonlinear (decreasing trend)') and str_smart.get('importance', 0) > 0.05:
    response = min(100, response + 10)
if str_hinge.get('direction', '') in ('negative',) and str_hinge.get('importance', 0) > 0.05:
    response = min(100, response + 5)

# Get top predictors from smart model
ranked = sorted(smart_effects.items(), key=lambda x: x[1].get('importance', 0), reverse=True)
top_features = [(k, v['importance'], v['direction']) for k, v in ranked[:3]]

explanation = (
    f"Research question: Is lower student-teacher ratio associated with higher academic performance? "
    f"Bivariate correlation between str_ratio and test_score: r={corr:.3f}. "
    f"OLS with controls (income, lunch, english, calworks, expenditure): "
    f"str_ratio coef={str_ols_coef:.3f}, p={str_ols_pval:.4f} ({sig_text} effect). "
    f"SmartAdditiveRegressor: str_ratio direction='{str_smart.get('direction')}', importance={str_smart.get('importance'):.3f}, rank={str_smart.get('rank')}. "
    f"HingeEBMRegressor: str_ratio direction='{str_hinge.get('direction')}', importance={str_hinge.get('importance'):.3f}, rank={str_hinge.get('rank')}. "
    f"Top predictors (SmartAdditive): {top_features}. "
    f"Conclusion: The bivariate relationship shows that higher ratios correlate with lower scores, but controlling for socioeconomic factors (lunch, income, english) substantially alters the story. "
    f"Key confounders like lunch (poverty proxy) dominate importance rankings. "
    f"The direct effect of lower student-teacher ratio on higher performance is present but may be partially mediated by confounders."
)

result = {"response": response, "explanation": explanation}
print(f"\nFinal response: {response}")
print(f"Explanation: {explanation}")

with open(os.path.join(os.path.dirname(__file__), 'conclusion.txt'), 'w') as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
