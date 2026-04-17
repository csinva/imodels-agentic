import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

# Load data
df = pd.read_csv('reading.csv')
print(f"Shape: {df.shape}")
print(df[['reader_view', 'dyslexia_bin', 'dyslexia', 'speed']].describe())

# Focus on dyslexic individuals (the question asks about dyslexia subgroup)
dyslexic = df[df['dyslexia_bin'] == 1].copy()
print(f"\nDyslexic participants: {len(dyslexic)} rows")
print(f"Reader view in dyslexic group - mean speed: reader_view=0: {dyslexic[dyslexic['reader_view']==0]['speed'].median():.1f}, reader_view=1: {dyslexic[dyslexic['reader_view']==1]['speed'].median():.1f}")

# Bivariate: correlation between reader_view and speed for dyslexic users
from scipy import stats
rv0 = dyslexic[dyslexic['reader_view']==0]['speed'].dropna()
rv1 = dyslexic[dyslexic['reader_view']==1]['speed'].dropna()
t_stat, p_val = stats.ttest_ind(rv0, rv1)
print(f"\nBivariate t-test (dyslexic): t={t_stat:.3f}, p={p_val:.4f}")
print(f"Mean speed without reader_view: {rv0.mean():.1f}, with: {rv1.mean():.1f}")

# Also check on full dataset with interaction
print("\n--- Full dataset bivariate ---")
corr = df['reader_view'].corr(df['speed'])
print(f"Correlation reader_view vs speed (all): {corr:.4f}")
print(f"Mean speed: reader_view=0: {df[df['reader_view']==0]['speed'].mean():.1f}, reader_view=1: {df[df['reader_view']==1]['speed'].mean():.1f}")

# OLS on dyslexic subgroup with controls
print("\n--- OLS on dyslexic subgroup ---")
dyslexic_clean = dyslexic.dropna(subset=['speed', 'reader_view', 'age', 'gender', 'num_words', 'Flesch_Kincaid'])
feature_cols = ['reader_view', 'age', 'gender', 'num_words', 'Flesch_Kincaid', 'retake_trial']
dyslexic_clean = dyslexic_clean[feature_cols + ['speed']].replace([np.inf, -np.inf], np.nan).dropna()
X_dys = dyslexic_clean[feature_cols].copy()
X_dys = sm.add_constant(X_dys)
y_dys = dyslexic_clean['speed']
model_dys = sm.OLS(y_dys, X_dys).fit()
print(model_dys.summary())

# OLS on full dataset with interaction term
print("\n--- OLS on full dataset with dyslexia interaction ---")
feat_full_base = ['reader_view', 'dyslexia_bin', 'age', 'gender', 'num_words', 'Flesch_Kincaid', 'retake_trial', 'speed']
df_clean = df.dropna(subset=feat_full_base).replace([np.inf, -np.inf], np.nan).dropna(subset=feat_full_base)
df_clean = df_clean.copy()
df_clean['rv_x_dyslexia'] = df_clean['reader_view'] * df_clean['dyslexia_bin']
feat_full = ['reader_view', 'dyslexia_bin', 'rv_x_dyslexia', 'age', 'gender', 'num_words', 'Flesch_Kincaid', 'retake_trial']
X_full = df_clean[feat_full]
X_full = sm.add_constant(X_full)
y_full = df_clean['speed']
model_full = sm.OLS(y_full, X_full).fit()
print(model_full.summary())

# SmartAdditiveRegressor on dyslexic subgroup
print("\n--- SmartAdditiveRegressor on dyslexic subgroup ---")
num_cols = ['reader_view', 'age', 'gender', 'num_words', 'Flesch_Kincaid', 'retake_trial']
X_smart = dyslexic_clean[num_cols].copy()
y_smart = dyslexic_clean['speed'].values
smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_smart, y_smart)
print(smart)
effects_smart = smart.feature_effects()
print("Feature effects:", effects_smart)

# HingeEBMRegressor on dyslexic subgroup
print("\n--- HingeEBMRegressor on dyslexic subgroup ---")
hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X_smart, y_smart)
print(hinge)
effects_hinge = hinge.feature_effects()
print("Feature effects:", effects_hinge)

# Summarize
rv_coef = model_dys.params['reader_view']
rv_pval = model_dys.pvalues['reader_view']
rv_smart = effects_smart.get('reader_view', {})
rv_hinge = effects_hinge.get('reader_view', {})

print(f"\n=== SUMMARY ===")
print(f"OLS (dyslexic): reader_view coef={rv_coef:.2f}, p={rv_pval:.4f}")
print(f"SmartAdditive: {rv_smart}")
print(f"HingeEBM: {rv_hinge}")

# Determine score
# Check if reader_view has significant positive effect for dyslexic
if rv_pval < 0.05 and rv_coef > 0:
    base_score = 75
elif rv_pval < 0.1 and rv_coef > 0:
    base_score = 55
elif rv_pval < 0.05:
    base_score = 40
else:
    base_score = 20

# Boost if confirmed by interpretable models
smart_imp = rv_smart.get('importance', 0)
hinge_imp = rv_hinge.get('importance', 0)
smart_dir = rv_smart.get('direction', 'zero')
hinge_dir = rv_hinge.get('direction', 'zero')

if 'positive' in str(smart_dir) and smart_imp > 0.05:
    base_score = min(base_score + 10, 100)
if 'positive' in str(hinge_dir) and hinge_imp > 0.05:
    base_score = min(base_score + 5, 100)

response_score = base_score

explanation = (
    f"The research question asks whether Reader View improves reading speed for dyslexic individuals. "
    f"Dyslexic subgroup analysis (n={len(dyslexic_clean)}): mean speed without reader_view={rv0.mean():.1f} wpm, "
    f"with reader_view={rv1.mean():.1f} wpm. Bivariate t-test: t={t_stat:.3f}, p={p_val:.4f}. "
    f"OLS with controls (age, gender, num_words, Flesch_Kincaid, retake_trial): reader_view coef={rv_coef:.2f}, p={rv_pval:.4f}. "
    f"SmartAdditiveRegressor: reader_view direction='{smart_dir}', importance={smart_imp:.3f}, rank={rv_smart.get('rank','N/A')}. "
    f"HingeEBMRegressor: reader_view direction='{hinge_dir}', importance={hinge_imp:.3f}. "
    f"Interaction model (full data): reader_view*dyslexia interaction coef={model_full.params.get('rv_x_dyslexia',0):.2f}, p={model_full.pvalues.get('rv_x_dyslexia',1):.4f}. "
    f"Overall: the effect of Reader View on reading speed for dyslexic individuals is "
    f"{'significant and positive' if rv_pval < 0.05 and rv_coef > 0 else 'weak or not statistically significant at p<0.05'}."
)

result = {"response": response_score, "explanation": explanation}
print(f"\nFinal result: {result}")

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("Wrote conclusion.txt")
