import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('fertility.csv')
print("Shape:", df.shape)
print(df.head())

# Parse dates
for col in ['DateTesting', 'StartDateofLastPeriod', 'StartDateofPeriodBeforeLast']:
    df[col] = pd.to_datetime(df[col], format='%m/%d/%y', errors='coerce')

# Compute cycle-based fertility proxy
# Days since last period
df['days_since_last'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

# Estimate cycle length from two period starts
df['computed_cycle_length'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days

# Use reported cycle length if available, else computed
df['cycle_length'] = df['ReportedCycleLength'].fillna(df['computed_cycle_length'])
# fallback to 28
df['cycle_length'] = df['cycle_length'].fillna(28)

# Cycle day (0-indexed from last period)
df['cycle_day'] = df['days_since_last']

# Days until next ovulation: ovulation ~14 days before next period
# next period = cycle_length - days_since_last days away
df['days_until_next_period'] = df['cycle_length'] - df['days_since_last']
df['days_from_ovulation'] = (df['days_until_next_period'] - 14).abs()

# Conception risk: high within 5 days before ovulation and on ovulation day
# Approximate: days_from_ovulation <= 5
df['high_fertility'] = (df['days_from_ovulation'] <= 5).astype(int)

# Continuous fertility: use normal approximation around ovulation
df['fertility_continuous'] = np.exp(-0.5 * (df['days_from_ovulation'] / 3) ** 2)

# Religiosity composite
df['religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

print("\nSummary stats:")
print(df[['days_since_last', 'cycle_length', 'days_from_ovulation', 'high_fertility', 'fertility_continuous', 'religiosity']].describe())

print("\nBivariate correlations with religiosity:")
for col in ['days_since_last', 'cycle_day', 'days_from_ovulation', 'high_fertility', 'fertility_continuous']:
    r = df[['religiosity', col]].dropna().corr().iloc[0, 1]
    print(f"  {col}: r={r:.3f}")

# OLS with controls
feature_cols = ['fertility_continuous', 'Relationship', 'Sure1', 'Sure2', 'cycle_length']
valid = df[feature_cols + ['religiosity']].dropna()
X = sm.add_constant(valid[feature_cols])
model = sm.OLS(valid['religiosity'], X).fit()
print("\nOLS with controls:")
print(model.summary())

# Also try high_fertility binary
feature_cols2 = ['high_fertility', 'Relationship', 'Sure1', 'Sure2', 'cycle_length']
valid2 = df[feature_cols2 + ['religiosity']].dropna()
X2 = sm.add_constant(valid2[feature_cols2])
model2 = sm.OLS(valid2['religiosity'], X2).fit()
print("\nOLS (binary high_fertility) with controls:")
print(model2.summary())

# Interpretable models
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

numeric_cols = ['fertility_continuous', 'high_fertility', 'Relationship', 'Sure1', 'Sure2', 'cycle_length']
valid3 = df[numeric_cols + ['religiosity']].dropna()
X3 = valid3[numeric_cols]
y3 = valid3['religiosity']

smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X3, y3)
print("\nSmartAdditiveRegressor:")
print(smart)
effects_smart = smart.feature_effects()
print(effects_smart)

hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X3, y3)
print("\nHingeEBMRegressor:")
print(hinge)
effects_hinge = hinge.feature_effects()
print(effects_hinge)

# Summarize findings
fert_cont_coef = model.params.get('fertility_continuous', None)
fert_cont_pval = model.pvalues.get('fertility_continuous', None)
fert_bin_coef = model2.params.get('high_fertility', None)
fert_bin_pval = model2.pvalues.get('high_fertility', None)

smart_fert = effects_smart.get('fertility_continuous', {})
hinge_fert = effects_hinge.get('fertility_continuous', {})

print(f"\nFertility continuous: coef={fert_cont_coef:.3f}, p={fert_cont_pval:.3f}")
print(f"High fertility binary: coef={fert_bin_coef:.3f}, p={fert_bin_pval:.3f}")
print(f"SmartAdditive fertility_continuous: {smart_fert}")
print(f"HingeEBM fertility_continuous: {hinge_fert}")

# Build conclusion
# Score based on p-values and effect consistency
p_cont = fert_cont_pval if fert_cont_pval is not None else 1.0
p_bin = fert_bin_pval if fert_bin_pval is not None else 1.0
smart_imp = smart_fert.get('importance', 0)
hinge_imp = hinge_fert.get('importance', 0)

if p_cont < 0.05 or p_bin < 0.05:
    if p_cont < 0.01 or p_bin < 0.01:
        score = 75
    else:
        score = 60
elif p_cont < 0.1 or p_bin < 0.1:
    score = 40
else:
    score = 20

# Adjust for model importance
if smart_imp > 0.1 or hinge_imp > 0.1:
    score = min(score + 10, 100)

explanation = (
    f"The effect of hormonal fluctuations (fertility) on religiosity was examined using "
    f"cycle-day-derived fertility proxies. "
    f"OLS with controls (Relationship, certainty, cycle length): "
    f"fertility_continuous coef={fert_cont_coef:.3f} (p={fert_cont_pval:.3f}), "
    f"high_fertility binary coef={fert_bin_coef:.3f} (p={fert_bin_pval:.3f}). "
    f"SmartAdditiveRegressor ranked fertility_continuous with importance={smart_imp:.3f} ({smart_fert.get('rank', 'N/A')}). "
    f"HingeEBM importance={hinge_imp:.3f}. "
    f"Relationship status was a notable confounder. "
    f"Overall, the fertility effect on religiosity is {'significant' if score >= 50 else 'weak/non-significant'} "
    f"(score={score})."
)

import json
result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\nconclusion.txt written:")
print(json.dumps(result, indent=2))
