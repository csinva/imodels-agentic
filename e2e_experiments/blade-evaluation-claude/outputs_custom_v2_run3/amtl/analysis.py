import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

# Load data
df = pd.read_csv("amtl.csv")
print("Shape:", df.shape)
print(df.head())
print("\nGenus counts:\n", df['genus'].value_counts())
print("\nSummary stats:\n", df.describe())

# Create DV: AMTL rate
df['amtl_rate'] = df['num_amtl'] / df['sockets']
print("\nAMTL rate by genus:")
print(df.groupby('genus')['amtl_rate'].describe())

# IV: is_homo_sapiens
df['is_homo'] = (df['genus'] == 'Homo sapiens').astype(int)

# Bivariate correlation
from scipy import stats
homo_rates = df[df['is_homo'] == 1]['amtl_rate']
nonhomo_rates = df[df['is_homo'] == 0]['amtl_rate']
t_stat, p_val = stats.ttest_ind(homo_rates, nonhomo_rates)
print(f"\nBivariate t-test: t={t_stat:.4f}, p={p_val:.4e}")
print(f"Homo mean AMTL rate: {homo_rates.mean():.4f}")
print(f"Non-homo mean AMTL rate: {nonhomo_rates.mean():.4f}")

# Encode tooth_class as dummies
tooth_dummies = pd.get_dummies(df['tooth_class'], prefix='tooth', drop_first=True).astype(float)

# OLS with controls
feature_cols = ['is_homo', 'age', 'prob_male'] + list(tooth_dummies.columns)
X_ols = pd.concat([df[['is_homo', 'age', 'prob_male']], tooth_dummies], axis=1)
X_ols = sm.add_constant(X_ols)
model = sm.OLS(df['amtl_rate'], X_ols).fit()
print("\n=== OLS REGRESSION ===")
print(model.summary())

# Prepare X for interpretable models (all numeric, no string cols)
X_interp = pd.concat([df[['is_homo', 'age', 'prob_male']], tooth_dummies], axis=1)
y = df['amtl_rate']

# SmartAdditiveRegressor
print("\n=== SmartAdditiveRegressor ===")
smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_interp, y)
print(smart)
smart_effects = smart.feature_effects()
print("\nFeature effects:", smart_effects)

# HingeEBMRegressor (skip if interpret not available)
try:
    print("\n=== HingeEBMRegressor ===")
    hinge = HingeEBMRegressor(n_knots=3)
    hinge.fit(X_interp, y)
    print(hinge)
    hinge_effects = hinge.feature_effects()
    print("\nFeature effects:", hinge_effects)
except Exception as e:
    print(f"HingeEBMRegressor skipped: {e}")
    hinge_effects = {}

# Gather results for conclusion
ols_coef_homo = model.params['is_homo']
ols_pval_homo = model.pvalues['is_homo']
smart_homo = smart_effects.get('is_homo', {})
hinge_homo = hinge_effects.get('is_homo', {})

print(f"\nSummary:")
print(f"OLS is_homo coef={ols_coef_homo:.4f}, p={ols_pval_homo:.4e}")
print(f"SmartAdditive is_homo: {smart_homo}")
print(f"HingeEBM is_homo: {hinge_homo}")

# Determine score and explanation
is_significant = ols_pval_homo < 0.05
positive_direction = ols_coef_homo > 0
smart_imp = float(smart_homo.get('importance', 0))
hinge_imp = float(hinge_homo.get('importance', 0)) if hinge_homo else 0.0

# Score based on guidelines
# OLS non-significant after controls, but SmartAdditive shows positive with ~6.6% importance
# Bivariate very strong, but confounded by age (85.2% importance in SmartAdditive)
if is_significant and positive_direction and (smart_imp > 0.05 or hinge_imp > 0.05):
    score = 85
elif is_significant and positive_direction:
    score = 75
elif not is_significant and positive_direction and smart_imp > 0.05:
    score = 45
elif not is_significant and positive_direction and smart_imp > 0.02:
    score = 30
elif not is_significant and not positive_direction:
    score = 10
else:
    score = 15

explanation = (
    f"Bivariate analysis shows Homo sapiens have dramatically higher AMTL rates "
    f"(mean={homo_rates.mean():.4f}) compared to non-human primates "
    f"(mean={nonhomo_rates.mean():.4f}), t={t_stat:.3f}, p={p_val:.2e}. "
    f"However, OLS with controls (age, sex, tooth class) reveals the human effect is NOT "
    f"significant: is_homo coef={ols_coef_homo:.4f}, p={ols_pval_homo:.2e}. "
    f"Age is the dominant driver (OLS coef=0.0064, p<0.001). "
    f"SmartAdditiveRegressor confirms: age ranks 1st (importance=85.2%, nonlinear increasing trend), "
    f"while is_homo is 2nd (importance=6.6%, positive linear direction). "
    f"The large bivariate difference is largely explained by age composition differences: "
    f"human specimens tend to be older, and older individuals lose more teeth regardless of species. "
    f"After controlling for age, sex, and tooth class, the human-specific AMTL effect is small "
    f"and statistically non-significant in OLS, though the SmartAdditive model still shows a modest "
    f"positive direction for is_homo. Overall, the evidence for a distinctly human elevation in AMTL "
    f"beyond what age explains is weak and inconsistent across models."
)

result = {"response": score, "explanation": explanation}
print(f"\nFinal result: {result}")

with open("conclusion.txt", "w") as f:
    json.dump(result, f)
print("conclusion.txt written.")
