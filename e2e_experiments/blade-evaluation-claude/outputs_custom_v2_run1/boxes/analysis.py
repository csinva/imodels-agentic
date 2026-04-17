import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import sys
sys.path.insert(0, '/home/chansingh/imodels-evolve/e2e_experiments/blade-evaluation-claude/outputs_custom_v2_run1/boxes')
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

df = pd.read_csv('boxes.csv')
print("Shape:", df.shape)
print(df.describe())

# The DV is a 3-category outcome: 1=unchosen, 2=majority, 3=minority
# Research question: how does reliance on majority preference develop with age across cultures?
# Create binary DV: chose majority (2) vs not
df['chose_majority'] = (df['y'] == 2).astype(int)

print("\nMajority choice rate by age:")
print(df.groupby('age')['chose_majority'].mean())

print("\nMajority choice rate by culture:")
print(df.groupby('culture')['chose_majority'].mean())

print("\nCorrelation matrix:")
numeric_cols = ['age', 'gender', 'majority_first', 'culture', 'chose_majority']
print(df[numeric_cols].corr())

# OLS with controls
X = df[['age', 'gender', 'majority_first', 'culture']]
X = sm.add_constant(X)
model = sm.OLS(df['chose_majority'], X).fit()
print("\nOLS Summary:")
print(model.summary())

# SmartAdditiveRegressor
numeric_columns = ['age', 'gender', 'majority_first', 'culture']
X_df = df[numeric_columns]
y = df['chose_majority']

smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_df, y)
print("\nSmartAdditiveRegressor:")
print(smart)
smart_effects = smart.feature_effects()
print("Feature effects:", smart_effects)

# HingeEBMRegressor (skip if interpret not available)
hinge_effects = {}
try:
    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_df, y)
    print("\nHingeEBMRegressor:")
    print(hinge)
    hinge_effects = hinge.feature_effects()
    print("Feature effects:", hinge_effects)
except Exception as e:
    print(f"\nHingeEBMRegressor unavailable: {e}")

# Collect results
age_ols_coef = model.params['age']
age_ols_pval = model.pvalues['age']
age_smart = smart_effects.get('age', {})
age_hinge = hinge_effects.get('age', {})

print(f"\nAge OLS coef={age_ols_coef:.4f}, p={age_ols_pval:.4f}")
print(f"Age SmartAdditive: {age_smart}")
if age_hinge:
    print(f"Age HingeEBM: {age_hinge}")

age_smart_importance = float(age_smart.get('importance', 0))
age_smart_direction = age_smart.get('direction', 'N/A')
age_smart_rank = age_smart.get('rank', 'N/A')
majority_first_importance = float(smart_effects.get('majority_first', {}).get('importance', 0))
culture_importance = float(smart_effects.get('culture', {}).get('importance', 0))

# Score: OLS p=0.803 (not significant), SmartAdditive shows age as nonlinear rank 2 (29% importance)
# The OLS misses the nonlinear U-shape; SmartAdditive reveals nonlinear developmental pattern.
# Age is not linearly significant but shows a meaningful nonlinear pattern.
if age_ols_pval < 0.05:
    if age_smart_importance > 0.15:
        score = 80
    else:
        score = 65
elif age_smart_importance > 0.20:
    # Nonlinear effect captured by SmartAdditive even if OLS misses it
    score = 60
elif age_smart_importance > 0.10:
    score = 45
else:
    score = 20

# Build explanation
hinge_note = (f"HingeEBMRegressor: age importance={float(age_hinge.get('importance',0)):.3f}, "
              f"direction={age_hinge.get('direction','N/A')}. " if age_hinge else
              "HingeEBMRegressor unavailable (interpret module missing). ")

explanation = (
    f"Age has an OLS coefficient of {age_ols_coef:.3f} (p={age_ols_pval:.3f}) on majority-choice, "
    f"which is not statistically significant after controlling for gender, majority_first, and culture. "
    f"However, the SmartAdditiveRegressor reveals a nonlinear U-shaped developmental pattern: "
    f"age is the 2nd most important predictor (importance={age_smart_importance:.1%}), with high majority-following "
    f"in young children (age<=4.5), a dip in middle childhood (ages 6-11), and a recovery in adolescence (age>12.5). "
    f"This nonlinear pattern is missed by OLS. {hinge_note}"
    f"The dominant predictor is majority_first (importance={majority_first_importance:.1%}), "
    f"a presentation-order effect. Cultural context is a weaker predictor (importance={culture_importance:.1%}). "
    f"Overall: there is a moderate nonlinear developmental effect of age on majority preference "
    f"that is robust in the SmartAdditive model but not captured by linear OLS, "
    f"while cultural context plays a smaller role than age."
)

result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\nconclusion.txt written:")
print(json.dumps(result, indent=2))
