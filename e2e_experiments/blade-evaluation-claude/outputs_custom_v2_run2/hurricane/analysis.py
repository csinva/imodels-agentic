import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('hurricane.csv')
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nSummary stats:")
print(df[['masfem', 'gender_mf', 'alldeaths', 'category', 'min', 'wind', 'ndam']].describe())

# Research question: Do more feminine hurricane names lead to fewer precautionary measures
# (proxied by more deaths)? IV=masfem/gender_mf, DV=alldeaths

print("\n--- Bivariate correlations ---")
numeric_cols = ['masfem', 'gender_mf', 'alldeaths', 'category', 'min', 'wind', 'ndam', 'elapsedyrs']
corr = df[numeric_cols].corr()
print("Correlation with alldeaths:")
print(corr['alldeaths'].sort_values(ascending=False))

# Log-transform deaths (highly skewed)
df['log_deaths'] = np.log1p(df['alldeaths'])
df['log_ndam'] = np.log1p(df['ndam'])

print("\n--- OLS: masfem -> log(deaths), no controls ---")
X_biv = sm.add_constant(df[['masfem']])
m_biv = sm.OLS(df['log_deaths'], X_biv).fit()
print(m_biv.summary())

print("\n--- OLS: masfem -> log(deaths), with severity controls ---")
feature_cols = ['masfem', 'min', 'wind', 'log_ndam', 'elapsedyrs', 'category']
X_ctrl = sm.add_constant(df[feature_cols].dropna())
y_ctrl = df['log_deaths'].loc[X_ctrl.index]
m_ctrl = sm.OLS(y_ctrl, X_ctrl).fit()
print(m_ctrl.summary())

print("\n--- OLS: gender_mf (binary) -> log(deaths), with severity controls ---")
feature_cols2 = ['gender_mf', 'min', 'wind', 'log_ndam', 'elapsedyrs', 'category']
X_ctrl2 = sm.add_constant(df[feature_cols2].dropna())
y_ctrl2 = df['log_deaths'].loc[X_ctrl2.index]
m_ctrl2 = sm.OLS(y_ctrl2, X_ctrl2).fit()
print(m_ctrl2.summary())

# Interpretable models
print("\n--- SmartAdditiveRegressor ---")
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

numeric_columns = ['masfem', 'min', 'wind', 'log_ndam', 'elapsedyrs', 'category']
X_interp = df[numeric_columns].dropna()
y_interp = df['log_deaths'].loc[X_interp.index]

smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_interp, y_interp)
print(smart)
smart_effects = smart.feature_effects()
print("\nSmartAdditive feature effects:")
print(smart_effects)

print("\n--- HingeEBMRegressor ---")
hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X_interp, y_interp)
print(hinge)
hinge_effects = hinge.feature_effects()
print("\nHingeEBM feature effects:")
print(hinge_effects)

# Summarize findings
masfem_biv_coef = m_biv.params['masfem']
masfem_biv_p = m_biv.pvalues['masfem']
masfem_ctrl_coef = m_ctrl.params['masfem']
masfem_ctrl_p = m_ctrl.pvalues['masfem']

smart_masfem = smart_effects.get('masfem', {})
hinge_masfem = hinge_effects.get('masfem', {})

print(f"\n=== SUMMARY ===")
print(f"Bivariate: masfem coef={masfem_biv_coef:.4f}, p={masfem_biv_p:.4f}")
print(f"Controlled: masfem coef={masfem_ctrl_coef:.4f}, p={masfem_ctrl_p:.4f}")
print(f"SmartAdditive masfem: {smart_masfem}")
print(f"HingeEBM masfem: {hinge_masfem}")

# Determine score
# The hypothesis: feminine names -> fewer precautions -> more deaths
# If masfem (femininity) positively predicts deaths, it supports the hypothesis
if masfem_ctrl_p < 0.05 and masfem_ctrl_coef > 0:
    score = 75
    direction = "positive and significant"
elif masfem_ctrl_p < 0.10 and masfem_ctrl_coef > 0:
    score = 55
    direction = "positive but marginal"
elif masfem_ctrl_p < 0.05 and masfem_ctrl_coef < 0:
    score = 20
    direction = "negative and significant (opposite direction)"
else:
    score = 30
    direction = "not significant"

# Adjust based on bivariate
if masfem_biv_p < 0.05 and masfem_biv_coef > 0:
    score = min(score + 10, 100)

# Adjust for interpretable model agreement
smart_imp = smart_masfem.get('importance', 0)
smart_dir = smart_masfem.get('direction', '')
if smart_imp > 0.05 and 'positive' in str(smart_dir).lower():
    score = min(score + 5, 100)

print(f"\nFinal score: {score}")

# Build explanation
explanation = (
    f"The research question asks whether more feminine hurricane names lead to fewer precautionary measures "
    f"(proxied by higher deaths). Bivariate OLS shows masfem coef={masfem_biv_coef:.3f} (p={masfem_biv_p:.3f}), "
    f"direction={direction}. "
    f"In the fully controlled model (min pressure, wind, damage, category, elapsed years): "
    f"masfem coef={masfem_ctrl_coef:.3f} (p={masfem_ctrl_p:.3f}). "
    f"SmartAdditiveRegressor: masfem importance={smart_masfem.get('importance', 0):.3f}, direction='{smart_masfem.get('direction', 'unknown')}', rank={smart_masfem.get('rank', 'N/A')}. "
    f"HingeEBM: masfem importance={hinge_masfem.get('importance', 0):.3f}, direction='{hinge_masfem.get('direction', 'unknown')}'. "
    f"Storm severity (damage, wind, pressure) dominates predictions. "
    f"The effect of name femininity on deaths is {direction} after controlling for storm severity, "
    f"providing {'moderate' if score >= 50 else 'weak'} support for the hypothesis that feminine names lead to fewer precautions."
)

import json
result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
print(json.dumps(result, indent=2))
