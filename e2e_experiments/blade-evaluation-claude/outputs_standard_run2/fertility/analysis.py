import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('fertility.csv')
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# Parse dates
for col in ['DateTesting', 'StartDateofLastPeriod', 'StartDateofPeriodBeforeLast']:
    df[col] = pd.to_datetime(df[col], format='%m/%d/%y', errors='coerce')

# Compute cycle day: days since start of last period
df['DaysSinceLastPeriod'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

# Compute cycle length from actual dates where available
df['ActualCycleLength'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days

# Use reported cycle length as fallback, actual where available
df['CycleLength'] = df['ActualCycleLength'].where(
    df['ActualCycleLength'].between(20, 45),
    df['ReportedCycleLength']
)
# Still missing? use 28
df['CycleLength'] = df['CycleLength'].fillna(28)

# Fertility estimate: based on Gangestad & Thornhill (2008)
# Peak fertility ~14 days before next period = CycleLength - DaysSinceLastPeriod - 14 days before next period
# Days until next period
df['DaysUntilNextPeriod'] = df['CycleLength'] - df['DaysSinceLastPeriod']

# Continuous fertility proxy: proximity to ovulation (day CycleLength - 14 from last period)
df['DaysFromOvulation'] = abs(df['DaysSinceLastPeriod'] - (df['CycleLength'] - 14))

# High fertility = within 5 days of ovulation
df['HighFertility'] = (df['DaysFromOvulation'] <= 5).astype(int)

# Fertility score (higher = more fertile)
df['FertilityScore'] = np.exp(-df['DaysFromOvulation'] / 3.0)

# Composite religiosity
df['Religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

# Drop rows with missing key data
clean = df.dropna(subset=['DaysSinceLastPeriod', 'Religiosity', 'FertilityScore'])
print(f"\nClean rows: {len(clean)}")

print("\nHigh vs Low fertility religiosity:")
high = clean[clean['HighFertility'] == 1]['Religiosity']
low = clean[clean['HighFertility'] == 0]['Religiosity']
print(f"High fertility (n={len(high)}): mean={high.mean():.3f}, std={high.std():.3f}")
print(f"Low fertility  (n={len(low)}): mean={low.mean():.3f}, std={low.std():.3f}")

t_stat, p_val = stats.ttest_ind(high, low)
print(f"T-test: t={t_stat:.3f}, p={p_val:.4f}")

# Correlation: continuous fertility score vs religiosity
r_pearson, p_pearson = stats.pearsonr(clean['FertilityScore'], clean['Religiosity'])
r_spearman, p_spearman = stats.spearmanr(clean['FertilityScore'], clean['Religiosity'])
print(f"\nPearson r={r_pearson:.3f}, p={p_pearson:.4f}")
print(f"Spearman r={r_spearman:.3f}, p={p_spearman:.4f}")

# Regression with controls
X = clean[['FertilityScore', 'Relationship', 'CycleLength']].copy()
X = sm.add_constant(X)
y = clean['Religiosity']
model = sm.OLS(y, X).fit()
print("\nOLS Regression:")
print(model.summary())

# Also check individual religiosity items
for col in ['Rel1', 'Rel2', 'Rel3']:
    sub = clean[['FertilityScore', col]].dropna()
    r, p = stats.pearsonr(sub['FertilityScore'], sub[col])
    print(f"{col}: r={r:.3f}, p={p:.4f}")

# Decision tree for interpretability
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

features = ['FertilityScore', 'DaysSinceLastPeriod', 'DaysFromOvulation', 'Relationship', 'CycleLength']
Xf = clean[features].fillna(clean[features].median())
yf = clean['Religiosity']

dt = DecisionTreeRegressor(max_depth=3, random_state=42)
dt.fit(Xf, yf)
print("\nDecision Tree feature importances:")
for f, imp in sorted(zip(features, dt.feature_importances_), key=lambda x: -x[1]):
    print(f"  {f}: {imp:.3f}")

# Summary
fertility_coef = model.params.get('FertilityScore', np.nan)
fertility_pval = model.pvalues.get('FertilityScore', np.nan)
print(f"\nFertility coef={fertility_coef:.4f}, p={fertility_pval:.4f}")
print(f"Pearson correlation (fertility vs religiosity): r={r_pearson:.3f}, p={p_pearson:.4f}")
print(f"T-test (high vs low fertility): t={t_stat:.3f}, p={p_val:.4f}")

# Determine conclusion
# Multiple tests: if any significant at p<0.05, there may be an effect
significant = p_pearson < 0.05 or p_val < 0.05 or fertility_pval < 0.05

# Effect direction
direction = "positive" if r_pearson > 0 else "negative"

# Score based on evidence
if p_pearson < 0.01 or p_val < 0.01:
    score = 70
elif p_pearson < 0.05 or p_val < 0.05:
    score = 55
else:
    score = 25  # no significant effect

explanation = (
    f"The analysis tested whether hormonal fluctuations (fertility) affect women's religiosity. "
    f"Fertility was estimated from cycle day data (proximity to ovulation). "
    f"Pearson correlation between fertility score and composite religiosity: r={r_pearson:.3f}, p={p_pearson:.4f}. "
    f"T-test (high vs low fertility groups): t={t_stat:.3f}, p={p_val:.4f}. "
    f"OLS regression coefficient for fertility: {fertility_coef:.4f}, p={fertility_pval:.4f}. "
    f"The effect is {'statistically significant' if significant else 'not statistically significant'} "
    f"and in the {direction} direction. "
    f"Overall, the evidence {'supports' if significant else 'does not support'} a meaningful effect of "
    f"fertility-related hormonal fluctuations on religiosity in this dataset."
)

result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\nconclusion.txt written:")
print(json.dumps(result, indent=2))
