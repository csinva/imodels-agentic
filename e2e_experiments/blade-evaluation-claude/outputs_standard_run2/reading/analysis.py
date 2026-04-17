import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('reading.csv')
print(f"Shape: {df.shape}")
print(df[['reader_view', 'dyslexia_bin', 'dyslexia', 'speed']].describe())

# Focus: does reader_view improve speed for dyslexic individuals?
dyslexic = df[df['dyslexia_bin'] == 1]
non_dyslexic = df[df['dyslexia_bin'] == 0]

print(f"\nDyslexic rows: {len(dyslexic)}, Non-dyslexic rows: {len(non_dyslexic)}")

# Speed distributions
dys_rv1 = dyslexic[dyslexic['reader_view'] == 1]['speed'].dropna()
dys_rv0 = dyslexic[dyslexic['reader_view'] == 0]['speed'].dropna()
print(f"\nDyslexic + Reader View ON:  n={len(dys_rv1)}, mean={dys_rv1.mean():.2f}, median={dys_rv1.median():.2f}")
print(f"Dyslexic + Reader View OFF: n={len(dys_rv0)}, mean={dys_rv0.mean():.2f}, median={dys_rv0.median():.2f}")

# T-test for dyslexic group
t_stat, p_val = stats.ttest_ind(dys_rv1, dys_rv0)
print(f"\nT-test (dyslexic, RV=1 vs RV=0): t={t_stat:.3f}, p={p_val:.4f}")

# Mann-Whitney U (non-parametric)
u_stat, p_mw = stats.mannwhitneyu(dys_rv1, dys_rv0, alternative='two-sided')
print(f"Mann-Whitney U: U={u_stat:.1f}, p={p_mw:.4f}")

# Also check non-dyslexic group
nd_rv1 = non_dyslexic[non_dyslexic['reader_view'] == 1]['speed'].dropna()
nd_rv0 = non_dyslexic[non_dyslexic['reader_view'] == 0]['speed'].dropna()
t2, p2 = stats.ttest_ind(nd_rv1, nd_rv0)
print(f"\nNon-dyslexic + Reader View ON:  mean={nd_rv1.mean():.2f}")
print(f"Non-dyslexic + Reader View OFF: mean={nd_rv0.mean():.2f}")
print(f"T-test (non-dyslexic, RV=1 vs RV=0): t={t2:.3f}, p={p2:.4f}")

# OLS regression: speed ~ reader_view * dyslexia_bin + controls
df_model = df.copy()
df_model['log_speed'] = np.log1p(df_model['speed'])
df_model['interaction'] = df_model['reader_view'] * df_model['dyslexia_bin']

features = ['reader_view', 'dyslexia_bin', 'interaction', 'age', 'num_words']
df_model = df_model.dropna(subset=features + ['log_speed'])
X = sm.add_constant(df_model[features])
y = df_model['log_speed']
model = sm.OLS(y, X).fit()
print("\nOLS regression (log_speed ~ reader_view * dyslexia_bin + controls):")
print(model.summary().tables[1])

# Effect size for dyslexic group
effect = dys_rv1.mean() - dys_rv0.mean()
pooled_std = np.sqrt((dys_rv1.std()**2 + dys_rv0.std()**2) / 2)
cohens_d = effect / pooled_std
print(f"\nEffect size (Cohen's d) for dyslexic group: {cohens_d:.3f}")
print(f"Mean speed change for dyslexic: {effect:.2f} wpm ({effect/dys_rv0.mean()*100:.1f}%)")

# Conclusion
# Reader view ON increases speed for dyslexic individuals
# Statistical significance determines score
alpha = 0.05
significant = p_val < alpha or p_mw < alpha
interaction_coef = model.params.get('interaction', 0)
interaction_pval = model.pvalues.get('interaction', 1)

print(f"\nSignificant main effect (dyslexic group)? {significant} (t-test p={p_val:.4f}, MW p={p_mw:.4f})")
print(f"Interaction term coef={interaction_coef:.4f}, p={interaction_pval:.4f}")

# Determine response score
# Check if reader_view improves (increases) speed for dyslexic individuals
if effect > 0 and significant:
    response = 70
    explanation = (f"Reader View is associated with higher reading speed in dyslexic individuals "
                   f"(mean speed: RV=1: {dys_rv1.mean():.1f} vs RV=0: {dys_rv0.mean():.1f} wpm, "
                   f"+{effect:.1f} wpm, {effect/dys_rv0.mean()*100:.1f}% increase). "
                   f"T-test p={p_val:.4f}, Mann-Whitney p={p_mw:.4f}. "
                   f"Interaction term in OLS: coef={interaction_coef:.4f}, p={interaction_pval:.4f}. "
                   f"Cohen's d={cohens_d:.3f}. The effect is statistically significant, supporting "
                   f"that Reader View improves reading speed for dyslexic individuals.")
elif effect > 0 and not significant:
    response = 40
    explanation = (f"Reader View shows a positive but non-significant trend for dyslexic individuals "
                   f"(mean speed: RV=1: {dys_rv1.mean():.1f} vs RV=0: {dys_rv0.mean():.1f} wpm). "
                   f"T-test p={p_val:.4f}, Mann-Whitney p={p_mw:.4f}. Not statistically significant.")
else:
    response = 20
    explanation = (f"Reader View does not improve reading speed for dyslexic individuals "
                   f"(mean speed: RV=1: {dys_rv1.mean():.1f} vs RV=0: {dys_rv0.mean():.1f} wpm). "
                   f"T-test p={p_val:.4f}. No significant improvement.")

import json
result = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)
print(f"\nconclusion.txt written with response={response}")
