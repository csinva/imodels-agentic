import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('hurricane.csv')
print("Shape:", df.shape)
print(df.describe())
print("\nCorrelation masfem vs alldeaths:", df['masfem'].corr(df['alldeaths']))
print("Correlation gender_mf vs alldeaths:", df['gender_mf'].corr(df['alldeaths']))

# Basic stats by gender
print("\nDeaths by gender (0=male, 1=female):")
print(df.groupby('gender_mf')['alldeaths'].describe())

# T-test: do female-named hurricanes cause more deaths?
male = df[df['gender_mf'] == 0]['alldeaths']
female = df[df['gender_mf'] == 1]['alldeaths']
t_stat, p_val = stats.ttest_ind(female, male)
print(f"\nT-test (female vs male deaths): t={t_stat:.3f}, p={p_val:.4f}")
print(f"Mean deaths - female: {female.mean():.1f}, male: {male.mean():.1f}")

# Correlation test: masfem vs alldeaths
r, p_corr = stats.pearsonr(df['masfem'], df['alldeaths'])
print(f"\nPearson r (masfem vs alldeaths): r={r:.3f}, p={p_corr:.4f}")

r_sp, p_sp = stats.spearmanr(df['masfem'], df['alldeaths'])
print(f"Spearman r (masfem vs alldeaths): r={r_sp:.3f}, p={p_sp:.4f}")

# OLS regression controlling for storm severity (category, wind, min pressure, ndam)
cols = ['masfem', 'category', 'wind', 'min', 'ndam', 'alldeaths']
df_clean = df[cols].dropna()
X = df_clean[['masfem', 'category', 'wind', 'min', 'ndam']].copy()
X = sm.add_constant(X)
y = np.log1p(df_clean['alldeaths'])
model = sm.OLS(y, X).fit()
print("\nOLS (log deaths ~ masfem + severity controls):")
print(model.summary())

# OLS without controls
X2 = sm.add_constant(df_clean[['masfem']])
model2 = sm.OLS(np.log1p(df_clean['alldeaths']), X2).fit()
print("\nOLS (log deaths ~ masfem only):")
print(f"  coef={model2.params['masfem']:.4f}, p={model2.pvalues['masfem']:.4f}")

# Decision tree feature importances with severity vars
X_feat = df_clean[['masfem', 'category', 'wind', 'min', 'ndam']].copy()
dt = DecisionTreeRegressor(max_depth=4, random_state=42)
dt.fit(X_feat, np.log1p(df_clean['alldeaths']))
print("\nDecision Tree feature importances:")
for name, imp in sorted(zip(X_feat.columns, dt.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.3f}")

# Summary of findings
masfem_coef = model.params['masfem']
masfem_pval = model.pvalues['masfem']
raw_corr_p = p_corr
print(f"\n--- Summary ---")
print(f"Raw correlation p-value: {raw_corr_p:.4f}")
print(f"Controlled regression: coef={masfem_coef:.4f}, p={masfem_pval:.4f}")

# Determine score
# The question is whether feminine names -> less precaution -> more deaths
# We assess if masfem is positively associated with alldeaths
significant_raw = p_corr < 0.05
significant_controlled = masfem_pval < 0.05
positive_direction = r > 0

print(f"Raw correlation positive direction: {positive_direction}")
print(f"Raw significant: {significant_raw}")
print(f"Controlled significant: {significant_controlled}")

# Scoring: effect is in expected direction but significance is debated in literature
# The original paper found significance, but replications raised doubts
if significant_controlled and positive_direction:
    score = 65
    explanation = (f"Controlled OLS shows masfem coef={masfem_coef:.3f} (p={masfem_pval:.3f}), "
                   f"suggesting feminine-named hurricanes are associated with more deaths after controlling for severity. "
                   f"Raw correlation r={r:.3f} (p={raw_corr_p:.3f}). However, effect size is modest and "
                   f"the result is sensitive to model specification and outliers.")
elif not significant_controlled and positive_direction and r > 0.1:
    score = 40
    explanation = (f"Raw correlation between masfem and deaths: r={r:.3f} (p={raw_corr_p:.3f}). "
                   f"Controlled regression: coef={masfem_coef:.3f} (p={masfem_pval:.3f}). "
                   f"Positive trend in expected direction but not statistically significant after controlling for storm severity. "
                   f"Weak evidence that feminine names lead to less precautionary behavior.")
else:
    score = 30
    explanation = (f"Raw correlation r={r:.3f} (p={raw_corr_p:.3f}), controlled regression coef={masfem_coef:.3f} (p={masfem_pval:.3f}). "
                   f"No consistent significant relationship found between name femininity and deaths after controlling for severity.")

import json
result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)
print(f"\nWritten conclusion.txt with score={score}")
