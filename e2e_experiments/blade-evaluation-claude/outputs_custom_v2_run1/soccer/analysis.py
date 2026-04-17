import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('soccer.csv')
print(f"Shape: {df.shape}")
print(df[['redCards', 'rater1', 'rater2', 'games', 'yellowCards', 'meanIAT', 'meanExp']].describe())

# Create average skin tone
df['skinTone'] = (df['rater1'].fillna(df['rater2']) + df['rater2'].fillna(df['rater1'])) / 2
df['skinTone'] = df['skinTone'].where(df['rater1'].notna() & df['rater2'].notna(),
                                       df['rater1'].fillna(df['rater2']))

# Bivariate correlation
valid = df[['redCards', 'skinTone', 'games', 'yellowCards', 'meanIAT', 'meanExp']].dropna()
print(f"\nBivariate corr (skinTone vs redCards): {valid['skinTone'].corr(valid['redCards']):.4f}")
print(f"N valid rows: {len(valid)}")

# Red card rate by skin tone group
df['skinGroup'] = pd.cut(df['skinTone'], bins=[-0.01, 0.25, 0.75, 1.01],
                          labels=['light', 'medium', 'dark'])
grouped = df.groupby('skinGroup')['redCards'].agg(['mean', 'sum', 'count'])
print("\nRed cards by skin group:")
print(grouped)

# OLS with controls
feature_cols = ['skinTone', 'games', 'yellowCards', 'meanIAT', 'meanExp']
reg_df = valid[['redCards'] + feature_cols].dropna()
X = sm.add_constant(reg_df[feature_cols])
model = sm.OLS(reg_df['redCards'], X).fit()
print("\nOLS Summary:")
print(model.summary())

# Interpretable models
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

numeric_cols = feature_cols
X_df = reg_df[numeric_cols]
y = reg_df['redCards']

smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_df, y)
print("\nSmartAdditiveRegressor:")
print(smart)
effects_smart = smart.feature_effects()
print(effects_smart)

hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X_df, y)
print("\nHingeEBMRegressor:")
print(hinge)
effects_hinge = hinge.feature_effects()
print(effects_hinge)

# Extract key results
skin_coef = model.params['skinTone']
skin_pval = model.pvalues['skinTone']
skin_smart = effects_smart.get('skinTone', {})
skin_hinge = effects_hinge.get('skinTone', {})

light_rate = grouped.loc['light', 'mean'] if 'light' in grouped.index else None
dark_rate = grouped.loc['dark', 'mean'] if 'dark' in grouped.index else None

print(f"\nSkin coef={skin_coef:.4f}, p={skin_pval:.4f}")
print(f"Light rate={light_rate:.4f}, Dark rate={dark_rate:.4f}")
print(f"SmartAdditive skinTone: {skin_smart}")
print(f"HingeEBM skinTone: {skin_hinge}")

# Score determination
if skin_pval < 0.05 and skin_coef > 0:
    base_score = 72
elif skin_pval < 0.1 and skin_coef > 0:
    base_score = 55
elif skin_coef > 0:
    base_score = 35
else:
    base_score = 15

# Adjust for interpretable model agreement
smart_dir = skin_smart.get('direction', '')
hinge_dir = skin_hinge.get('direction', '')
if 'positive' in smart_dir or 'increasing' in smart_dir:
    base_score = min(base_score + 5, 100)
if 'positive' in hinge_dir or 'increasing' in hinge_dir:
    base_score = min(base_score + 5, 100)

response = base_score

explanation = (
    f"Players with darker skin tone show a {'positive' if skin_coef > 0 else 'negative'} association with red cards "
    f"(OLS coef={skin_coef:.4f}, p={skin_pval:.4f}). "
    f"Bivariate rates: light skin={light_rate:.4f}, dark skin={dark_rate:.4f} red cards per dyad. "
    f"SmartAdditive model: skinTone direction='{smart_dir}', importance={skin_smart.get('importance', 'N/A'):.3f}, rank={skin_smart.get('rank', 'N/A')}. "
    f"HingeEBM model: skinTone direction='{hinge_dir}', importance={skin_hinge.get('importance', 'N/A'):.3f}. "
    f"After controlling for games played, yellow cards, implicit bias (meanIAT), and explicit bias (meanExp), "
    f"the skin tone effect {'persists significantly' if skin_pval < 0.05 else 'is marginal/non-significant'}. "
    f"yellowCards is a strong confounder (more disciplinary events overall). "
    f"The evidence {'supports' if skin_pval < 0.05 else 'weakly supports'} darker-skinned players receiving more red cards."
)

result = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print(f"\nFinal score: {response}")
print(f"Conclusion written to conclusion.txt")
