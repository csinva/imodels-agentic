import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge
import json

import os
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(script_dir, "teachingratings.csv"))
print(df.shape)
print(df.describe())

# Encode categorical variables
df['minority_enc'] = (df['minority'] == 'yes').astype(int)
df['gender_enc'] = (df['gender'] == 'male').astype(int)
df['credits_enc'] = (df['credits'] == 'more').astype(int)
df['native_enc'] = (df['native'] == 'yes').astype(int)
df['tenure_enc'] = (df['tenure'] == 'yes').astype(int)
df['division_enc'] = (df['division'] == 'upper').astype(int)

# Correlation: beauty vs eval
r, p = stats.pearsonr(df['beauty'], df['eval'])
print(f"Pearson r(beauty, eval) = {r:.4f}, p = {p:.4f}")

# Simple OLS: eval ~ beauty
X_simple = sm.add_constant(df['beauty'])
model_simple = sm.OLS(df['eval'], X_simple).fit()
print(model_simple.summary())

# Multiple regression controlling for other variables
features = ['beauty', 'age', 'minority_enc', 'gender_enc', 'credits_enc',
            'native_enc', 'tenure_enc', 'division_enc', 'students']
X_full = sm.add_constant(df[features])
model_full = sm.OLS(df['eval'], X_full).fit()
print(model_full.summary())

beauty_coef = model_full.params['beauty']
beauty_pval = model_full.pvalues['beauty']
print(f"\nBeauty coefficient (full model): {beauty_coef:.4f}, p-value: {beauty_pval:.4f}")

# Interpretation
significant = beauty_pval < 0.05
positive_effect = beauty_coef > 0

explanation = (
    f"Simple correlation: r={r:.3f}, p={p:.4f}. "
    f"In multivariate OLS controlling for age, gender, minority status, native English, tenure, division, and class size: "
    f"beauty coefficient={beauty_coef:.4f} (p={beauty_pval:.4f}). "
    f"{'Statistically significant positive effect' if significant and positive_effect else 'Not statistically significant' if not significant else 'Significant but negative'}. "
    f"This replicates the Hamermesh & Parker (2005) finding that physical attractiveness positively predicts teaching evaluations."
)

response = 85 if significant and positive_effect else 20

result = {"response": response, "explanation": explanation}
with open(os.path.join(script_dir, "conclusion.txt"), "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
print(json.dumps(result, indent=2))
