import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import statsmodels.api as sm

df = pd.read_csv('mortgage.csv')
print("Shape:", df.shape)
print(df.describe())

# DV: accept (1=approved), IV: female
print("\nAcceptance rate by gender:")
print(df.groupby('female')['accept'].mean())

print("\nCorrelation with accept:")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['Unnamed: 0', 'deny']]
print(df[numeric_cols].corr()['accept'])

# OLS with controls
feature_cols = [c for c in numeric_cols if c != 'accept']
X = df[feature_cols].dropna()
y = df.loc[X.index, 'accept']
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
print("\nOLS Summary:")
print(model.summary())

# Interpretable models
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

X_df = df[feature_cols].dropna()
y_s = df.loc[X_df.index, 'accept']

smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_df, y_s)
print("\nSmartAdditiveRegressor:")
print(smart)
smart_effects = smart.feature_effects()
print(smart_effects)

try:
    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_df, y_s)
    print("\nHingeEBMRegressor:")
    print(hinge)
    hinge_effects = hinge.feature_effects()
    print(hinge_effects)
except Exception as e:
    print(f"\nHingeEBMRegressor failed: {e}")
    hinge_effects = {}

# Gather results
female_ols_coef = model.params.get('female', None)
female_ols_pval = model.pvalues.get('female', None)
female_smart = smart_effects.get('female', {})
female_hinge = hinge_effects.get('female', {})

print(f"\nOLS female coef: {female_ols_coef:.4f}, p={female_ols_pval:.4f}")
print(f"SmartAdditive female: {female_smart}")
print(f"HingeEBM female: {female_hinge}")

# Bivariate rates
male_accept = df[df['female']==0]['accept'].mean()
female_accept = df[df['female']==1]['accept'].mean()
print(f"\nMale acceptance rate: {male_accept:.3f}")
print(f"Female acceptance rate: {female_accept:.3f}")

# Score: effect is small in bivariate, becomes even smaller with controls
# Determine score based on significance and magnitude
sig = female_ols_pval < 0.05 if female_ols_pval is not None else False
smart_imp = female_smart.get('importance', 0)
hinge_imp = female_hinge.get('importance', 0)

explanation = (
    f"Bivariate analysis shows virtually no gender difference in mortgage approval: "
    f"male acceptance rate={male_accept:.3f}, female acceptance rate={female_accept:.3f} (delta=0.0003). "
    f"OLS with controls (black, creditworthiness, debt ratios) yields female coef={female_ols_coef:.4f} "
    f"(p={female_ols_pval:.4f}), technically significant at p<0.05 but with a tiny effect size "
    f"(~3.7 percentage points). However, the SmartAdditiveRegressor completely excludes female from its model "
    f"(importance=0.000, direction=zero), suggesting this OLS result may be a spurious artifact. "
    f"The dominant predictors are denied_PMI (importance=38.5%), PI_ratio (19.8%, nonlinear decreasing), "
    f"loan_to_value (10.6%, nonlinear), bad_history (10.1%), and consumer_credit (8.7%). "
    f"Gender is inconsistent across models: marginally significant in OLS but completely zeroed out "
    f"in the more robust additive model. The evidence suggests gender has little to no meaningful effect "
    f"on mortgage approval once creditworthiness factors are controlled."
)

# Bivariate=zero, SmartAdditive=zero, OLS=marginal sig -> weak inconsistent effect
if sig and smart_imp > 0.05:
    score = 60
elif sig and smart_imp == 0:
    score = 25  # OLS sig but contradicted by SmartAdditive
elif sig:
    score = 40
elif female_ols_pval < 0.10:
    score = 20
else:
    score = 10

result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print(f"\nFinal score: {score}")
print("conclusion.txt written.")
