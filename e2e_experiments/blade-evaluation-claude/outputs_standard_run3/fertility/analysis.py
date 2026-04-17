import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

# Load data
df = pd.read_csv('fertility.csv')
print("Shape:", df.shape)
print(df.describe())

# Parse dates
for col in ['DateTesting', 'StartDateofLastPeriod', 'StartDateofPeriodBeforeLast']:
    df[col] = pd.to_datetime(df[col], format='%m/%d/%y', errors='coerce')

# Compute cycle day: days since last period start
df['DaysSincePeriod'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

# Fill missing ReportedCycleLength with median
df['CycleLength'] = df['ReportedCycleLength'].fillna(df['ReportedCycleLength'].median())

# Compute fertility: high near ovulation (~day 14), using backward counting
# Days until next ovulation estimated; ovulation ~ CycleLength - 14 days from start
# Fertility peaks at ovulation, estimated as cycle day relative to cycle length
df['CycleDay'] = df['DaysSincePeriod']
df['FertilityEstimate'] = df['CycleDay'] / df['CycleLength']

# High fertility: roughly days 10-17 of a 28-day cycle = cycle fraction 0.35-0.61
# Use a continuous fertility measure based on distance from expected ovulation day
df['DaysToOvulation'] = df['CycleLength'] - 14 - df['CycleDay']
df['FertilityScore'] = np.exp(-0.5 * (df['DaysToOvulation'] / 5) ** 2)  # Gaussian around ovulation

# Religiosity composite
df['Religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

print("\nFertility score stats:")
print(df['FertilityScore'].describe())
print("\nReligiosity stats:")
print(df['Religiosity'].describe())

# Drop rows with missing key values
df_clean = df.dropna(subset=['FertilityScore', 'Religiosity', 'CycleDay'])
print(f"\nClean sample size: {len(df_clean)}")

# --- Statistical Tests ---

# 1. Pearson correlation: FertilityScore vs Religiosity
r, p = stats.pearsonr(df_clean['FertilityScore'], df_clean['Religiosity'])
print(f"\nPearson r(FertilityScore, Religiosity) = {r:.4f}, p = {p:.4f}")

# 2. High vs Low fertility groups
median_fert = df_clean['FertilityScore'].median()
high_fert = df_clean[df_clean['FertilityScore'] >= median_fert]['Religiosity']
low_fert = df_clean[df_clean['FertilityScore'] < median_fert]['Religiosity']
t_stat, t_p = stats.ttest_ind(high_fert, low_fert)
print(f"T-test (high vs low fertility): t={t_stat:.4f}, p={t_p:.4f}")
print(f"High fertility mean religiosity: {high_fert.mean():.4f}")
print(f"Low fertility mean religiosity: {low_fert.mean():.4f}")

# 3. OLS regression: Religiosity ~ FertilityScore + controls
X = df_clean[['FertilityScore', 'Relationship', 'Sure1', 'Sure2']].copy()
X = sm.add_constant(X)
y = df_clean['Religiosity']
model = sm.OLS(y, X).fit()
print("\nOLS regression summary:")
print(model.summary())
print(f"\nFertilityScore coef: {model.params['FertilityScore']:.4f}, p={model.pvalues['FertilityScore']:.4f}")

# 4. Spearman correlation (non-parametric)
rho, p_spear = stats.spearmanr(df_clean['FertilityScore'], df_clean['Religiosity'])
print(f"\nSpearman rho(FertilityScore, Religiosity) = {rho:.4f}, p = {p_spear:.4f}")

# 5. Alternative: use cycle day directly
r2, p2 = stats.pearsonr(df_clean['CycleDay'].dropna(),
                         df_clean.loc[df_clean['CycleDay'].notna(), 'Religiosity'])
print(f"\nPearson r(CycleDay, Religiosity) = {r2:.4f}, p = {p2:.4f}")

# --- Interpretable Model ---
features = ['FertilityScore', 'CycleDay', 'CycleLength', 'Relationship', 'Sure1', 'Sure2']
df_model = df_clean[features + ['Religiosity']].dropna()
X_m = df_model[features].values
y_m = df_model['Religiosity'].values

ridge = Ridge(alpha=1.0)
ridge.fit(X_m, y_m)
print("\nRidge coefficients:")
for feat, coef in zip(features, ridge.coef_):
    print(f"  {feat}: {coef:.4f}")

dt = DecisionTreeRegressor(max_depth=3, random_state=42)
dt.fit(X_m, y_m)
print("\nDecision Tree feature importances:")
for feat, imp in sorted(zip(features, dt.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.4f}")

# --- Conclusion ---
# Key findings
fert_p = model.pvalues['FertilityScore']
fert_coef = model.params['FertilityScore']

significant = fert_p < 0.05
direction = "positive" if fert_coef > 0 else "negative"

explanation = (
    f"Research question: Does fertility (hormonal fluctuations in menstrual cycle) affect women's religiosity? "
    f"Fertility was estimated using a Gaussian around the expected ovulation day (cycle_length - 14 days). "
    f"Pearson correlation between fertility score and religiosity: r={r:.4f}, p={p:.4f}. "
    f"T-test (high vs low fertility groups): t={t_stat:.4f}, p={t_p:.4f}. "
    f"OLS regression (controlling for relationship status and date certainty): "
    f"FertilityScore coef={fert_coef:.4f}, p={fert_p:.4f}. "
    f"Spearman rho={rho:.4f}, p={p_spear:.4f}. "
    f"The effect of fertility on religiosity is {'statistically significant' if significant else 'NOT statistically significant'} "
    f"(p={'<0.05' if significant else '>0.05'}). "
    f"The direction is {direction}. "
    f"Overall, the evidence {'supports' if significant else 'does not support'} a meaningful relationship between "
    f"hormonal fluctuations associated with fertility and women's religiosity in this dataset (n={len(df_model)})."
)

# Score: 0=strong No, 100=strong Yes
# Based on significance and effect size
if significant and abs(r) > 0.15:
    response = 70
elif significant:
    response = 55
elif p < 0.1:
    response = 35
else:
    response = 15

print(f"\n=== CONCLUSION ===")
print(f"Response score: {response}")
print(f"Explanation: {explanation}")

result = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\nconclustion.txt written successfully.")
