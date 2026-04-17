import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

df = pd.read_csv('reading.csv')

print("Shape:", df.shape)
print(df[['reader_view', 'dyslexia_bin', 'speed']].describe())

# Focus on dyslexia individuals
dyslexia_df = df[df['dyslexia_bin'] == 1].copy()
print(f"\nDyslexia participants: {len(dyslexia_df)} rows")
print(dyslexia_df.groupby('reader_view')['speed'].describe())

# Bivariate: reader_view effect on speed in dyslexia group
rv0 = dyslexia_df[dyslexia_df['reader_view'] == 0]['speed']
rv1 = dyslexia_df[dyslexia_df['reader_view'] == 1]['speed']
print(f"\nDyslexia group - reader_view=0 mean speed: {rv0.mean():.2f}, median: {rv0.median():.2f}")
print(f"Dyslexia group - reader_view=1 mean speed: {rv1.mean():.2f}, median: {rv1.median():.2f}")

from scipy import stats
t_stat, t_p = stats.ttest_ind(rv1, rv0)
print(f"t-test: t={t_stat:.3f}, p={t_p:.4f}")

# OLS with controls on dyslexia subset
numeric_cols = ['reader_view', 'age', 'dyslexia', 'gender', 'retake_trial', 'Flesch_Kincaid', 'num_words', 'img_width']
dyslexia_df_clean = dyslexia_df[numeric_cols + ['speed']].dropna()

X = sm.add_constant(dyslexia_df_clean[numeric_cols])
model = sm.OLS(dyslexia_df_clean['speed'], X).fit()
print("\nOLS on dyslexia group:")
print(model.summary())

# Also OLS on full dataset with interaction term
full_df = df[numeric_cols + ['dyslexia_bin', 'speed']].dropna().copy()
full_df['rv_x_dyslexia'] = full_df['reader_view'] * full_df['dyslexia_bin']
X_full = sm.add_constant(full_df[numeric_cols + ['dyslexia_bin', 'rv_x_dyslexia']])
model_full = sm.OLS(full_df['speed'], X_full).fit()
print("\nOLS on full dataset with interaction:")
print(model_full.summary())

# Interpretable models on dyslexia subset
print("\n=== SmartAdditiveRegressor on dyslexia group ===")
X_interp = dyslexia_df_clean[numeric_cols]
y_interp = dyslexia_df_clean['speed']

smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_interp, y_interp)
print(smart)
smart_effects = smart.feature_effects()
print("Feature effects:", smart_effects)

print("\n=== HingeEBMRegressor on dyslexia group ===")
hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X_interp, y_interp)
print(hinge)
hinge_effects = hinge.feature_effects()
print("Feature effects:", hinge_effects)

# Gather results
rv_coef = model.params.get('reader_view', model.params.get('const', 0))
rv_coef = model.params['reader_view']
rv_pval = model.pvalues['reader_view']

smart_rv = smart_effects.get('reader_view', {})
smart_rv_importance = smart_rv.get('importance', 0)
smart_rv_direction = smart_rv.get('direction', 'unknown')
smart_rv_rank = smart_rv.get('rank', 0)

hinge_rv = hinge_effects.get('reader_view', {})
hinge_rv_importance = hinge_rv.get('importance', 0)
hinge_rv_direction = hinge_rv.get('direction', 'unknown')

# Interaction term in full model
interaction_coef = model_full.params.get('rv_x_dyslexia', 0)
interaction_pval = model_full.pvalues.get('rv_x_dyslexia', 1.0)

mean_diff = rv1.mean() - rv0.mean()
pct_diff = (mean_diff / rv0.mean()) * 100

print(f"\n--- Summary ---")
print(f"Mean speed diff (reader_view=1 - 0) in dyslexia group: {mean_diff:.2f} ({pct_diff:.1f}%)")
print(f"OLS coef for reader_view (dyslexia group): {rv_coef:.3f}, p={rv_pval:.4f}")
print(f"Interaction coef (full model): {interaction_coef:.3f}, p={interaction_pval:.4f}")
print(f"SmartAdditive: reader_view importance={smart_rv_importance:.3f}, direction={smart_rv_direction}, rank={smart_rv_rank}")
print(f"HingeEBM: reader_view importance={hinge_rv_importance:.3f}, direction={hinge_rv_direction}")

# Score determination
# Evaluate significance and effect size
if rv_pval < 0.05 and mean_diff > 0:
    base_score = 75
elif rv_pval < 0.05 and mean_diff < 0:
    base_score = 20
elif rv_pval < 0.1:
    base_score = 50
else:
    base_score = 20

# Boost if interpretable models confirm
if smart_rv_direction in ('positive', 'nonlinear (increasing trend)') and smart_rv_importance > 0.05:
    base_score = min(base_score + 10, 100)
if hinge_rv_direction in ('positive',) and hinge_rv_importance > 0.05:
    base_score = min(base_score + 5, 100)

score = base_score

explanation = (
    f"The research question asks whether Reader View improves reading speed for dyslexic individuals. "
    f"Among {len(dyslexia_df)} dyslexia-positive rows, mean speed with Reader View={rv1.mean():.1f} vs without={rv0.mean():.1f} "
    f"(diff={mean_diff:.1f}, {pct_diff:.1f}%). "
    f"OLS (controlled for age, dyslexia severity, gender, retake_trial, Flesch_Kincaid, num_words, img_width) yields "
    f"reader_view coef={rv_coef:.3f} (p={rv_pval:.4f}). "
    f"Full-dataset interaction term (reader_view x dyslexia_bin) coef={interaction_coef:.3f} (p={interaction_pval:.4f}). "
    f"SmartAdditiveRegressor: reader_view importance={smart_rv_importance:.3f}, direction='{smart_rv_direction}', rank={smart_rv_rank}. "
    f"HingeEBMRegressor: reader_view importance={hinge_rv_importance:.3f}, direction='{hinge_rv_direction}'. "
    f"t-test in dyslexia group: t={t_stat:.3f}, p={t_p:.4f}."
)

result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print(f"\nFinal score: {score}")
print(f"Conclusion written to conclusion.txt")
