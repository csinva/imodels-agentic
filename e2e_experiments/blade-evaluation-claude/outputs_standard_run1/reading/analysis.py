import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('reading.csv')
print("Shape:", df.shape)
print(df[['reader_view', 'dyslexia', 'dyslexia_bin', 'speed']].describe())

# Focus on dyslexic individuals only
dyslexic = df[df['dyslexia_bin'] == 1].copy()
non_dyslexic = df[df['dyslexia_bin'] == 0].copy()

print(f"\nDyslexic: {len(dyslexic)}, Non-dyslexic: {len(non_dyslexic)}")

# Cap extreme outliers (top 1%) for speed
speed_cap = df['speed'].quantile(0.99)
df['speed_capped'] = df['speed'].clip(upper=speed_cap)
dyslexic_c = df[(df['dyslexia_bin'] == 1)].copy()
non_dyslexic_c = df[(df['dyslexia_bin'] == 0)].copy()

# --- Primary analysis: reader_view effect on speed for dyslexic individuals ---
dys_rv1 = dyslexic_c[dyslexic_c['reader_view'] == 1]['speed_capped']
dys_rv0 = dyslexic_c[dyslexic_c['reader_view'] == 0]['speed_capped']

print(f"\nDyslexic + Reader View ON:  n={len(dys_rv1)}, mean={dys_rv1.mean():.2f}, median={dys_rv1.median():.2f}")
print(f"Dyslexic + Reader View OFF: n={len(dys_rv0)}, mean={dys_rv0.mean():.2f}, median={dys_rv0.median():.2f}")

t_stat, p_val = stats.ttest_ind(dys_rv1, dys_rv0)
print(f"\nT-test (dyslexic, reader_view 1 vs 0): t={t_stat:.3f}, p={p_val:.4f}")

# Mann-Whitney U (non-parametric)
u_stat, p_mw = stats.mannwhitneyu(dys_rv1, dys_rv0, alternative='two-sided')
print(f"Mann-Whitney U test: U={u_stat:.1f}, p={p_mw:.4f}")

# --- Control for page_id (pages differ in length, affecting speed) ---
# OLS with interaction: speed ~ reader_view * dyslexia_bin + controls
df_model = df.dropna(subset=['speed_capped', 'reader_view', 'dyslexia_bin', 'age', 'num_words'])
df_model['log_speed'] = np.log1p(df_model['speed_capped'])
df_model['interaction'] = df_model['reader_view'] * df_model['dyslexia_bin']

X = df_model[['reader_view', 'dyslexia_bin', 'interaction', 'age', 'num_words']]
X = sm.add_constant(X)
y = df_model['log_speed']
model = sm.OLS(y, X).fit()
print("\n--- OLS Regression (log speed) ---")
print(model.summary().tables[1])

rv_coef = model.params.get('reader_view', np.nan)
interaction_coef = model.params.get('interaction', np.nan)
interaction_pval = model.pvalues.get('interaction', np.nan)
rv_pval = model.pvalues.get('reader_view', np.nan)

print(f"\nreader_view coef={rv_coef:.4f}, p={rv_pval:.4f}")
print(f"interaction (reader_view x dyslexia) coef={interaction_coef:.4f}, p={interaction_pval:.4f}")

# --- Summary ---
mean_diff = dys_rv1.mean() - dys_rv0.mean()
pct_change = 100 * mean_diff / dys_rv0.mean()
print(f"\nMean speed difference (reader view on - off) for dyslexic: {mean_diff:.2f} ({pct_change:.1f}%)")
print(f"Primary t-test p-value: {p_val:.4f}")
print(f"Mann-Whitney p-value: {p_mw:.4f}")

# Determine response score
# Check if reader_view increases speed for dyslexic individuals
# Consider both the direction of effect and significance
if p_val < 0.05 and mean_diff > 0:
    response = 75
    explanation = (
        f"Reader View significantly increases reading speed for individuals with dyslexia. "
        f"Dyslexic readers with Reader View ON had mean speed {dys_rv1.mean():.1f} vs {dys_rv0.mean():.1f} without "
        f"(+{pct_change:.1f}%, t-test p={p_val:.4f}, Mann-Whitney p={p_mw:.4f}). "
        f"OLS interaction term (reader_view x dyslexia_bin): coef={interaction_coef:.4f}, p={interaction_pval:.4f}."
    )
elif p_val < 0.1 and mean_diff > 0:
    response = 60
    explanation = (
        f"Reader View shows a marginal positive effect on reading speed for dyslexic individuals "
        f"(mean speed {dys_rv1.mean():.1f} vs {dys_rv0.mean():.1f}, +{pct_change:.1f}%, t-test p={p_val:.4f}). "
        f"Effect is not significant at the 0.05 level."
    )
elif p_val < 0.05 and mean_diff < 0:
    response = 20
    explanation = (
        f"Reader View significantly DECREASES reading speed for dyslexic individuals "
        f"(mean speed {dys_rv1.mean():.1f} vs {dys_rv0.mean():.1f}, {pct_change:.1f}%, t-test p={p_val:.4f})."
    )
else:
    response = 40
    explanation = (
        f"Reader View does not significantly improve reading speed for individuals with dyslexia. "
        f"Dyslexic readers: mean speed {dys_rv1.mean():.1f} (reader view ON) vs {dys_rv0.mean():.1f} (OFF), "
        f"difference {pct_change:.1f}%, t-test p={p_val:.4f}, Mann-Whitney p={p_mw:.4f}. "
        f"OLS interaction coef={interaction_coef:.4f}, p={interaction_pval:.4f}."
    )

conclusion = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print(f"\nConclusion written: response={response}")
print(f"Explanation: {explanation}")
