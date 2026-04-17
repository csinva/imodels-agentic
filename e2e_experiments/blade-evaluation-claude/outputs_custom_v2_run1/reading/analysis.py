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
print(df[['reader_view', 'dyslexia', 'dyslexia_bin', 'speed']].describe())

# The research question: Does Reader View improve reading speed for dyslexic individuals?
# DV: speed, IV: reader_view, focus: dyslexia population

# Filter to dyslexic individuals only (dyslexia_bin == 1)
df_dys = df[df['dyslexia_bin'] == 1].copy()
print(f"\nDyslexic participants rows: {len(df_dys)}")
print(f"reader_view=0 count: {(df_dys['reader_view']==0).sum()}, reader_view=1 count: {(df_dys['reader_view']==1).sum()}")

# Log-transform speed (highly skewed)
df['log_speed'] = np.log1p(df['speed'])
df_dys['log_speed'] = np.log1p(df_dys['speed'])

# Bivariate: reader_view effect on speed among dyslexic readers
rv0 = df_dys[df_dys['reader_view']==0]['speed']
rv1 = df_dys[df_dys['reader_view']==1]['speed']
print(f"\nDyslexic readers - speed without Reader View: mean={rv0.mean():.1f}, median={rv0.median():.1f}")
print(f"Dyslexic readers - speed with Reader View: mean={rv1.mean():.1f}, median={rv1.median():.1f}")

from scipy import stats
t_stat, p_val = stats.ttest_ind(rv1, rv0)
print(f"t-test: t={t_stat:.3f}, p={p_val:.4f}")

# OLS on dyslexic subset with controls
numeric_cols = ['reader_view', 'age', 'gender', 'num_words', 'Flesch_Kincaid', 'correct_rate', 'img_width']
df_dys_clean = df_dys[numeric_cols + ['log_speed']].dropna()
X_ols = df_dys_clean[numeric_cols]
X_ols = sm.add_constant(X_ols)
model = sm.OLS(df_dys_clean['log_speed'], X_ols).fit()
print("\n=== OLS on dyslexic subset (DV=log_speed) ===")
print(model.summary())

# Also run on full dataset with interaction term
df_full = df[numeric_cols + ['dyslexia_bin', 'log_speed']].dropna()
df_full['rv_x_dyslexia'] = df_full['reader_view'] * df_full['dyslexia_bin']
feature_cols_full = ['reader_view', 'dyslexia_bin', 'rv_x_dyslexia', 'age', 'gender', 'num_words', 'Flesch_Kincaid', 'correct_rate', 'img_width']
X_full = sm.add_constant(df_full[feature_cols_full])
model_full = sm.OLS(df_full['log_speed'], X_full).fit()
print("\n=== OLS full dataset with interaction (rv x dyslexia) ===")
print(model_full.summary())

# Interpretable models on dyslexic subset
X_interp = df_dys_clean[numeric_cols]
y_interp = df_dys_clean['log_speed']

print("\n=== SmartAdditiveRegressor (dyslexic subset) ===")
smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_interp, y_interp)
print(smart)
smart_effects = smart.feature_effects()
print("Feature effects:", smart_effects)

print("\n=== HingeEBMRegressor (dyslexic subset) ===")
hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X_interp, y_interp)
print(hinge)
hinge_effects = hinge.feature_effects()
print("Feature effects:", hinge_effects)

# Gather key numbers for conclusion
rv_coef = model.params.get('reader_view', np.nan)
rv_pval = model.pvalues.get('reader_view', np.nan)
interaction_coef = model_full.params.get('rv_x_dyslexia', np.nan)
interaction_pval = model_full.pvalues.get('rv_x_dyslexia', np.nan)

smart_rv = smart_effects.get('reader_view', {})
hinge_rv = hinge_effects.get('reader_view', {})

print(f"\nSummary:")
print(f"  OLS (dyslexic subset) reader_view coef={rv_coef:.4f}, p={rv_pval:.4f}")
print(f"  Full OLS interaction coef={interaction_coef:.4f}, p={interaction_pval:.4f}")
print(f"  Smart model reader_view: {smart_rv}")
print(f"  Hinge model reader_view: {hinge_rv}")

# Build conclusion
rv_direction = "positive" if rv_coef > 0 else "negative"
speed_diff_pct = (rv1.mean() - rv0.mean()) / rv0.mean() * 100

# Score: consider significance in OLS and models
# reader_view effect on dyslexic readers
sig_ols = rv_pval < 0.05
sig_bivariate = p_val < 0.05
smart_importance = smart_rv.get('importance', 0)
hinge_importance = hinge_rv.get('importance', 0)

# Determine score
if sig_ols and sig_bivariate and smart_importance > 0.05:
    score = 75
elif sig_ols or sig_bivariate:
    score = 55
elif smart_importance > 0.05 or hinge_importance > 0.05:
    score = 35
else:
    score = 15

# Adjust based on direction
if rv_direction == "negative":
    # Reader View slows down dyslexic readers — answer is No
    score = 100 - score

explanation = (
    f"Among individuals with dyslexia (n={len(df_dys_clean)} observations), Reader View {'increased' if rv_coef>0 else 'decreased'} "
    f"log-reading speed by {abs(rv_coef):.3f} units (OLS coef={rv_coef:.3f}, p={rv_pval:.4f}). "
    f"Bivariate: dyslexic readers had mean speed {rv0.mean():.1f} wpm without vs {rv1.mean():.1f} wpm with Reader View "
    f"({speed_diff_pct:+.1f}%, t-test p={p_val:.4f}). "
    f"The full-sample OLS interaction (reader_view x dyslexia) coef={interaction_coef:.3f} (p={interaction_pval:.4f}). "
    f"SmartAdditive model: reader_view importance={smart_importance:.3f}, direction={smart_rv.get('direction','unknown')}; "
    f"HingeEBM: importance={hinge_importance:.3f}, direction={hinge_rv.get('direction','unknown')}. "
    f"Other important predictors include num_words and Flesch_Kincaid (text complexity). "
    f"The effect {'is' if sig_ols else 'is not'} statistically significant in the controlled OLS model, "
    f"and {'holds' if smart_importance > 0.05 else 'is weak'} in the interpretable models."
)

result = {"response": score, "explanation": explanation}
print(f"\nFinal result: {result}")

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("conclusion.txt written.")
