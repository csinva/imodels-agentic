import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import warnings
warnings.filterwarnings('ignore')

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('boxes.csv')
print("Shape:", df.shape)
print(df.describe())

# Create binary DV: chose majority option
df['majority_choice'] = (df['y'] == 2).astype(int)
print("\nMajority choice rate:", df['majority_choice'].mean())
print("\nMajority choice by age:\n", df.groupby('age')['majority_choice'].mean())
print("\nMajority choice by culture:\n", df.groupby('culture')['majority_choice'].mean())

# Bivariate correlation
from scipy import stats
corr, pval = stats.pearsonr(df['age'], df['majority_choice'])
print(f"\nBivariate corr age vs majority_choice: r={corr:.4f}, p={pval:.4f}")

# OLS with controls
print("\n--- OLS: majority_choice ~ age + gender + majority_first + culture ---")
feature_cols = ['age', 'gender', 'majority_first', 'culture']
X = df[feature_cols].copy()
X = sm.add_constant(X)
model = sm.OLS(df['majority_choice'], X).fit()
print(model.summary())

# Also try logistic regression
print("\n--- Logistic Regression ---")
logit = sm.Logit(df['majority_choice'], X).fit(disp=0)
print(logit.summary())

# OLS per culture to check interaction
print("\n--- Age effect per culture (OLS) ---")
for c in sorted(df['culture'].unique()):
    sub = df[df['culture'] == c]
    if len(sub) > 10:
        r, p = stats.pearsonr(sub['age'], sub['majority_choice'])
        print(f"  Culture {c} (n={len(sub)}): r={r:.3f}, p={p:.3f}")

# SmartAdditiveRegressor
print("\n--- SmartAdditiveRegressor ---")
numeric_cols = ['age', 'gender', 'majority_first', 'culture']
X_df = df[numeric_cols]
y = df['majority_choice']
smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_df, y)
print(smart)
effects = smart.feature_effects()
print("Feature effects:", effects)

# HingeEBMRegressor
print("\n--- HingeEBMRegressor ---")
hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X_df, y)
print(hinge)
hinge_effects = hinge.feature_effects()
print("Feature effects:", hinge_effects)

# Summarize findings
age_ols_coef = model.params['age']
age_ols_pval = model.pvalues['age']
age_smart = effects.get('age', {})
age_hinge = hinge_effects.get('age', {})

# Score: Does age significantly predict majority choice?
# Check significance, direction, robustness
sig = age_ols_pval < 0.05
moderate_sig = age_ols_pval < 0.10
direction = 'positive' if age_ols_coef > 0 else 'negative'

smart_importance = age_smart.get('importance', 0)
hinge_importance = age_hinge.get('importance', 0)
smart_rank = age_smart.get('rank', 0)

print(f"\n=== SUMMARY ===")
print(f"Age OLS coef: {age_ols_coef:.4f}, p={age_ols_pval:.4f}")
print(f"Age direction: {direction}")
print(f"SmartAdditive importance: {smart_importance:.3f}, rank: {smart_rank}")
print(f"HingeEBM importance: {hinge_importance:.3f}")

# Determine score
if sig and smart_importance > 0.1:
    score = 80
elif sig and smart_importance > 0.05:
    score = 70
elif sig:
    score = 60
elif moderate_sig:
    score = 40
else:
    # check bivariate
    if pval < 0.05:
        score = 35
    else:
        score = 15

# Adjust for cultural heterogeneity
culture_effects = df.groupby('culture').apply(lambda g: stats.pearsonr(g['age'], g['majority_choice'])[0] if len(g) > 10 else 0)
consistent_direction = (culture_effects > 0).sum() / len(culture_effects)
print(f"Fraction of cultures with positive age effect: {consistent_direction:.2f}")

if consistent_direction < 0.5:
    score = max(score - 20, 10)

smart_dir = age_smart.get('direction', '')
hinge_dir = age_hinge.get('direction', '')

explanation = (
    f"The research question asks whether children's reliance on majority preference increases with age across cultural contexts. "
    f"The DV is a binary indicator of choosing the majority option (y=2). "
    f"Bivariate correlation: r={corr:.3f}, p={pval:.4f}. "
    f"OLS with controls (gender, majority_first, culture): age coef={age_ols_coef:.4f}, p={age_ols_pval:.4f}, indicating a {direction} relationship. "
    f"SmartAdditiveRegressor ranks age with importance={smart_importance:.3f} (rank {smart_rank}), direction='{smart_dir}'. "
    f"HingeEBMRegressor: age importance={hinge_importance:.3f}, direction='{hinge_dir}'. "
    f"{consistent_direction*100:.0f}% of cultures show positive age-majority correlation, suggesting {'consistent' if consistent_direction > 0.6 else 'mixed'} cross-cultural pattern. "
    f"{'The effect is statistically significant and robust across models.' if sig else 'The effect is not statistically significant in controlled models.'}"
)

result = {"response": score, "explanation": explanation}
print("\nResult:", json.dumps(result, indent=2))

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\nWritten conclusion.txt")
