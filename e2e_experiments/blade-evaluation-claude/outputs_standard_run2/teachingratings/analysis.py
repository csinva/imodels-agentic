import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder
import json

df = pd.read_csv("/home/chansingh/imodels-evolve/e2e_experiments/blade-evaluation-claude/outputs_standard_run2/teachingratings/teachingratings.csv")
print(df.shape)
print(df.head())
print(df.describe())

# Encode categorical variables
le = LabelEncoder()
df['minority_enc'] = le.fit_transform(df['minority'])
df['gender_enc'] = le.fit_transform(df['gender'])
df['credits_enc'] = le.fit_transform(df['credits'])
df['division_enc'] = le.fit_transform(df['division'])
df['native_enc'] = le.fit_transform(df['native'])
df['tenure_enc'] = le.fit_transform(df['tenure'])

# Basic correlation: beauty vs eval
r, p = stats.pearsonr(df['beauty'], df['eval'])
print(f"\nPearson r(beauty, eval) = {r:.4f}, p = {p:.4f}")

# OLS regression: eval ~ beauty (simple)
X_simple = sm.add_constant(df['beauty'])
model_simple = sm.OLS(df['eval'], X_simple).fit()
print("\nSimple OLS (eval ~ beauty):")
print(model_simple.summary())

# Multiple regression controlling for covariates
feature_cols = ['beauty', 'age', 'minority_enc', 'gender_enc', 'credits_enc',
                'division_enc', 'native_enc', 'tenure_enc', 'students']
X_full = sm.add_constant(df[feature_cols])
model_full = sm.OLS(df['eval'], X_full).fit()
print("\nMultiple OLS (eval ~ beauty + covariates):")
print(model_full.summary())

beauty_coef = model_full.params['beauty']
beauty_pval = model_full.pvalues['beauty']
print(f"\nBeauty coefficient: {beauty_coef:.4f}, p-value: {beauty_pval:.4f}")

# Determine response score
# Strong positive significant relationship -> high score
if beauty_pval < 0.05 and beauty_coef > 0:
    response = 80
    explanation = (f"Beauty has a statistically significant positive effect on teaching evaluations. "
                   f"Simple correlation: r={r:.3f}, p={p:.4f}. "
                   f"In multiple regression controlling for age, gender, minority status, credits, division, native English, tenure, and class size, "
                   f"beauty coefficient = {beauty_coef:.4f} (p={beauty_pval:.4f}). "
                   f"Higher beauty ratings are associated with higher teaching evaluation scores, consistent with Hamermesh & Parker (2005).")
elif beauty_pval < 0.05 and beauty_coef < 0:
    response = 20
    explanation = (f"Beauty has a statistically significant negative effect on teaching evaluations (unexpected). "
                   f"Coefficient = {beauty_coef:.4f}, p={beauty_pval:.4f}.")
else:
    response = 30
    explanation = (f"Beauty does not have a statistically significant effect on teaching evaluations after controlling for covariates. "
                   f"Coefficient = {beauty_coef:.4f}, p={beauty_pval:.4f}.")

result = {"response": response, "explanation": explanation}
with open("/home/chansingh/imodels-evolve/e2e_experiments/blade-evaluation-claude/outputs_standard_run2/teachingratings/conclusion.txt", "w") as f:
    json.dump(result, f)

print(f"\nConclusion: {result}")
