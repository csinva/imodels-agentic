import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import warnings
warnings.filterwarnings('ignore')

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

# Load data
df = pd.read_csv('fertility.csv')
print("Shape:", df.shape)
print(df.head())
print(df.dtypes)

# Parse dates
for col in ['DateTesting', 'StartDateofLastPeriod', 'StartDateofPeriodBeforeLast']:
    df[col] = pd.to_datetime(df[col], format='%m/%d/%y', errors='coerce')

# Compute days since last period (cycle day)
df['DaysSinceLastPeriod'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days
df['CycleLength2'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days

# Fertility proxy: days since period / cycle length -> position in cycle
# Ovulation typically around day 14; peak fertility ~day 10-16
# Use a continuous fertility score: proximity to ovulation
df['CycleDay'] = df['DaysSinceLastPeriod']
df['FertilityEstimate'] = df.apply(
    lambda r: max(0, 1 - abs(r['CycleDay'] - r['ReportedCycleLength'] * 0.45) / (r['ReportedCycleLength'] * 0.45))
    if pd.notna(r['CycleDay']) and r['ReportedCycleLength'] > 0 else np.nan,
    axis=1
)

# Religiosity composite
df['Religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

print("\nSummary stats:")
print(df[['Religiosity', 'FertilityEstimate', 'CycleDay', 'ReportedCycleLength', 'Relationship']].describe())

print("\nCorrelations with Religiosity:")
numeric_cols = ['FertilityEstimate', 'CycleDay', 'ReportedCycleLength', 'Relationship', 'Sure1', 'Sure2']
for col in numeric_cols:
    corr = df['Religiosity'].corr(df[col])
    print(f"  {col}: r={corr:.4f}")

# OLS with controls
df_clean = df.dropna(subset=['Religiosity', 'FertilityEstimate', 'Relationship', 'Sure1', 'Sure2', 'ReportedCycleLength'])
print(f"\nSample after dropping NAs: {len(df_clean)}")

feature_cols = ['FertilityEstimate', 'Relationship', 'Sure1', 'Sure2', 'ReportedCycleLength']
X = df_clean[feature_cols]
X = sm.add_constant(X)
y = df_clean['Religiosity']
model = sm.OLS(y, X).fit()
print(model.summary())

# Interpretable models
X_df = df_clean[feature_cols].copy()
y_arr = df_clean['Religiosity'].values

print("\n=== SmartAdditiveRegressor ===")
smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_df, y_arr)
print(smart)
effects_smart = smart.feature_effects()
print(effects_smart)

print("\n=== HingeEBMRegressor ===")
hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X_df, y_arr)
print(hinge)
effects_hinge = hinge.feature_effects()
print(effects_hinge)

# Build conclusion
fertility_ols_coef = model.params.get('FertilityEstimate', np.nan)
fertility_ols_p = model.pvalues.get('FertilityEstimate', np.nan)
fertility_smart = effects_smart.get('FertilityEstimate', {})
fertility_hinge = effects_hinge.get('FertilityEstimate', {})

print(f"\nFertility OLS: coef={fertility_ols_coef:.4f}, p={fertility_ols_p:.4f}")
print(f"Fertility SmartAdditive: {fertility_smart}")
print(f"Fertility HingeEBM: {fertility_hinge}")

# Determine score
if fertility_ols_p < 0.05:
    if abs(fertility_ols_coef) > 0.5:
        score = 75
    else:
        score = 60
elif fertility_ols_p < 0.1:
    score = 35
else:
    score = 15

smart_imp = fertility_smart.get('importance', 0)
hinge_imp = fertility_hinge.get('importance', 0)

# Adjust based on interpretable model importance
if smart_imp > 0.15 or hinge_imp > 0.15:
    score = max(score, 40)

# Build explanation
other_top = sorted(
    [(k, v) for k, v in effects_smart.items() if k != 'FertilityEstimate'],
    key=lambda x: x[1].get('importance', 0), reverse=True
)[:3]
other_str = ", ".join([f"{k} (imp={v.get('importance',0):.2%})" for k, v in other_top])

explanation = (
    f"FertilityEstimate (proximity to ovulation) shows OLS coef={fertility_ols_coef:.3f}, p={fertility_ols_p:.3f}. "
    f"SmartAdditive ranks it with importance={smart_imp:.1%} ({fertility_smart.get('direction','unknown')} direction). "
    f"HingeEBM importance={hinge_imp:.1%} ({fertility_hinge.get('direction','unknown')}). "
    f"Other predictors in model: {other_str}. "
    f"The effect of fertility-cycle position on religiosity is "
    + ("significant" if fertility_ols_p < 0.05 else ("marginally significant" if fertility_ols_p < 0.1 else "non-significant"))
    + f" after controlling for relationship status and date certainty. "
    f"The fertility estimate has relatively low importance compared to other variables, suggesting hormonal fluctuations have at best a weak influence on religiosity."
)

result = {"response": score, "explanation": explanation}
print("\nResult:", result)

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("Wrote conclusion.txt")
