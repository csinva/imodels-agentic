import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
import warnings
warnings.filterwarnings('ignore')

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

# Load data
df = pd.read_csv('fertility.csv')

# Parse dates and compute fertility proxy (days to next ovulation)
# Ovulation typically occurs ~14 days before next period
# days_since_last_period = DateTesting - StartDateofLastPeriod
# cycle_phase_normalized = days_since_last_period / ReportedCycleLength
# High fertility = mid-cycle (around 0.5 of cycle)

df['DateTesting'] = pd.to_datetime(df['DateTesting'], format='%m/%d/%y')
df['StartDateofLastPeriod'] = pd.to_datetime(df['StartDateofLastPeriod'], format='%m/%d/%y')
df['StartDateofPeriodBeforeLast'] = pd.to_datetime(df['StartDateofPeriodBeforeLast'], format='%m/%d/%y')

# Compute cycle length from actual dates (more accurate than self-report)
df['ComputedCycleLength'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days

# Days since last period started
df['DaysSinceLastPeriod'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

# Use reported cycle length if available, else computed
cycle_len = df['ReportedCycleLength'].fillna(df['ComputedCycleLength'])
df['CycleLength'] = cycle_len

# Fertility index: estimated days until next ovulation (ovulation = cycle_len - 14 days from start)
# Higher value = closer to ovulation = higher fertility
df['DaysToOvulation'] = (df['CycleLength'] - 14) - df['DaysSinceLastPeriod']
# Normalize to 0-1 fertility window; higher = more fertile (closer to ovulation)
df['FertilityProxy'] = 1 - (np.abs(df['DaysToOvulation']) / (df['CycleLength'] / 2))
df['FertilityProxy'] = df['FertilityProxy'].clip(-1, 1)

# Composite religiosity score (average of 3 items)
df['Religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

print("=== Dataset Overview ===")
print(df[['FertilityProxy', 'DaysSinceLastPeriod', 'CycleLength', 'Religiosity', 'Relationship']].describe())

print("\n=== Correlation with Religiosity ===")
corr_cols = ['FertilityProxy', 'DaysSinceLastPeriod', 'CycleLength', 'Relationship', 'Sure1', 'Sure2']
for col in corr_cols:
    valid = df[[col, 'Religiosity']].dropna()
    r = valid[col].corr(valid['Religiosity'])
    print(f"  {col}: r = {r:.4f} (n={len(valid)})")

# Drop rows with missing key variables
analysis_df = df[['FertilityProxy', 'DaysSinceLastPeriod', 'CycleLength',
                   'Relationship', 'Sure1', 'Sure2', 'Religiosity']].dropna()
print(f"\nAnalysis sample size: {len(analysis_df)}")

# OLS with controls
print("\n=== OLS Regression (DV = Religiosity) ===")
feature_cols = ['FertilityProxy', 'DaysSinceLastPeriod', 'CycleLength', 'Relationship', 'Sure1', 'Sure2']
X = analysis_df[feature_cols]
X = sm.add_constant(X)
model = sm.OLS(analysis_df['Religiosity'], X).fit()
print(model.summary())

# SmartAdditiveRegressor
print("\n=== SmartAdditiveRegressor ===")
X_df = analysis_df[feature_cols]
y = analysis_df['Religiosity']
smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_df, y)
print(smart)
smart_effects = smart.feature_effects()
print(smart_effects)

# HingeEBMRegressor
print("\n=== HingeEBMRegressor ===")
hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X_df, y)
print(hinge)
hinge_effects = hinge.feature_effects()
print(hinge_effects)

# Extract key results
fertility_ols_coef = model.params.get('FertilityProxy', None)
fertility_ols_pval = model.pvalues.get('FertilityProxy', None)
fertility_smart = smart_effects.get('FertilityProxy', {})
fertility_hinge = hinge_effects.get('FertilityProxy', {})

days_ols_coef = model.params.get('DaysSinceLastPeriod', None)
days_ols_pval = model.pvalues.get('DaysSinceLastPeriod', None)
days_smart = smart_effects.get('DaysSinceLastPeriod', {})

print(f"\nFertilityProxy OLS: coef={fertility_ols_coef:.4f}, p={fertility_ols_pval:.4f}")
print(f"DaysSinceLastPeriod OLS: coef={days_ols_coef:.4f}, p={days_ols_pval:.4f}")
print(f"SmartAdditive FertilityProxy: {fertility_smart}")
print(f"SmartAdditive DaysSinceLastPeriod: {days_smart}")
print(f"HingeEBM FertilityProxy: {fertility_hinge}")

# Determine score
fertility_p = fertility_ols_pval if fertility_ols_pval is not None else 1.0
days_p = days_ols_pval if days_ols_pval is not None else 1.0
fertility_imp = fertility_smart.get('importance', 0)
days_imp = days_smart.get('importance', 0)

# Combined fertility signal importance
combined_fertility_imp = max(fertility_imp, days_imp)

if (fertility_p < 0.05 or days_p < 0.05) and combined_fertility_imp > 0.1:
    score = 75
elif (fertility_p < 0.1 or days_p < 0.1) and combined_fertility_imp > 0.05:
    score = 55
elif (fertility_p < 0.2 or days_p < 0.2) or combined_fertility_imp > 0.05:
    score = 35
else:
    score = 15

explanation = (
    f"The research question asks whether hormonal fluctuations associated with fertility affect women's religiosity. "
    f"Fertility was operationalized as FertilityProxy (estimated closeness to ovulation based on cycle timing). "
    f"OLS regression controlling for cycle length, relationship status, and date-reporting certainty showed: "
    f"FertilityProxy coef={fertility_ols_coef:.3f} (p={fertility_ols_pval:.3f}), "
    f"DaysSinceLastPeriod coef={days_ols_coef:.3f} (p={days_ols_pval:.3f}). "
    f"SmartAdditiveRegressor ranked FertilityProxy importance={fertility_imp:.3f} and DaysSinceLastPeriod importance={days_imp:.3f}. "
    f"HingeEBM FertilityProxy: {fertility_hinge}. "
    f"The overall evidence {'supports' if score >= 50 else 'does not strongly support'} a meaningful relationship "
    f"between fertility-related hormonal fluctuations and religiosity, "
    f"though the effect is {'statistically significant' if min(fertility_p, days_p) < 0.05 else 'not statistically significant at p<0.05'}. "
    f"Relationship status was also in the model as a potential confounder."
)

result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print(f"\n=== CONCLUSION ===")
print(json.dumps(result, indent=2))
print("conclusion.txt written.")
