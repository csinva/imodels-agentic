import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('reading.csv')

print("Shape:", df.shape)
print("\nBasic stats for speed:")
print(df['speed'].describe())

# Focus on dyslexia individuals
dyslexia_df = df[df['dyslexia_bin'] == 1].copy()
print(f"\nDyslexia individuals: {len(dyslexia_df)} rows")

# Cap extreme outliers (99th percentile)
speed_cap = df['speed'].quantile(0.99)
dyslexia_df = dyslexia_df[dyslexia_df['speed'] < speed_cap].copy()
print(f"After removing outliers: {len(dyslexia_df)} rows")

# Split by reader_view
rv_on = dyslexia_df[dyslexia_df['reader_view'] == 1]['speed']
rv_off = dyslexia_df[dyslexia_df['reader_view'] == 0]['speed']

print(f"\nDyslexia group - Reader View ON: n={len(rv_on)}, mean={rv_on.mean():.2f}, median={rv_on.median():.2f}")
print(f"Dyslexia group - Reader View OFF: n={len(rv_off)}, mean={rv_off.mean():.2f}, median={rv_off.median():.2f}")

# T-test
t_stat, p_val = stats.ttest_ind(rv_on, rv_off)
print(f"\nT-test: t={t_stat:.3f}, p={p_val:.4f}")

# Mann-Whitney U (non-parametric)
u_stat, p_mw = stats.mannwhitneyu(rv_on, rv_off, alternative='two-sided')
print(f"Mann-Whitney U: U={u_stat:.1f}, p={p_mw:.4f}")

# OLS regression: speed ~ reader_view + dyslexia + interaction, on full dataset
df_clean = df[df['speed'] < df['speed'].quantile(0.99)].copy()
df_clean['dyslexia_x_rv'] = df_clean['dyslexia_bin'] * df_clean['reader_view']

cols = ['reader_view', 'dyslexia_bin', 'dyslexia_x_rv', 'num_words']
df_model = df_clean[cols].dropna()
y = np.log1p(df_clean.loc[df_model.index, 'speed'])
X = sm.add_constant(df_model)
model = sm.OLS(y, X).fit()
print("\nOLS results (log speed ~ reader_view * dyslexia):")
print(model.summary().tables[1])

# Effect: reader_view coefficient for dyslexia subgroup
rv_coef = model.params['reader_view']
interaction_coef = model.params['dyslexia_x_rv']
dyslexia_rv_effect = rv_coef + interaction_coef
print(f"\nReader view effect for dyslexia group (reader_view + interaction): {dyslexia_rv_effect:.4f}")

# Decision
speed_increase = rv_on.mean() - rv_off.mean()
pct_change = speed_increase / rv_off.mean() * 100
print(f"\nMean speed difference (ON - OFF): {speed_increase:.2f} words/min ({pct_change:.1f}%)")

# Score: combine statistical significance and effect direction
# Higher score if significant AND positive effect (speed increases with reader view for dyslexia)
if p_val < 0.05 and speed_increase > 0:
    score = 75
elif p_val < 0.10 and speed_increase > 0:
    score = 60
elif p_val < 0.05 and speed_increase < 0:
    score = 20
elif speed_increase > 0:
    score = 40
else:
    score = 25

# Check Mann-Whitney too
if p_mw < 0.05 and speed_increase > 0:
    score = max(score, 70)
elif p_mw >= 0.05:
    score = min(score, 45)

print(f"\nFinal score: {score}")

explanation = (
    f"Among dyslexia individuals (n={len(dyslexia_df)}), reader view {'increased' if speed_increase > 0 else 'decreased'} "
    f"mean reading speed by {abs(pct_change):.1f}% "
    f"(reader_view ON: {rv_on.mean():.1f} vs OFF: {rv_off.mean():.1f} words/min). "
    f"T-test p={p_val:.4f}, Mann-Whitney p={p_mw:.4f}. "
    f"{'The effect is statistically significant.' if p_val < 0.05 else 'The effect is not statistically significant (p>=0.05).'} "
    f"OLS interaction term (reader_view x dyslexia) coefficient: {interaction_coef:.4f}."
)

import json
result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    f.write(json.dumps(result))

print("\nconclusion.txt written.")
print(json.dumps(result, indent=2))
