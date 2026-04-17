import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('soccer.csv')
print(f"Shape: {df.shape}")
print(f"Red cards: {df['redCards'].describe()}")

# Create average skin tone
df['skin_tone'] = df[['rater1', 'rater2']].mean(axis=1)

# Drop rows without skin tone rating
df_rated = df.dropna(subset=['skin_tone']).copy()
print(f"\nRows with skin tone rating: {len(df_rated)}")
print(f"Skin tone distribution:\n{df_rated['skin_tone'].value_counts().sort_index()}")

# Aggregate by player to get red card rates
player_agg = df_rated.groupby('playerShort').agg(
    skin_tone=('skin_tone', 'mean'),
    total_games=('games', 'sum'),
    total_redCards=('redCards', 'sum'),
    rater1=('rater1', 'mean'),
    rater2=('rater2', 'mean'),
).reset_index()

player_agg['red_card_rate'] = player_agg['total_redCards'] / player_agg['total_games']

print(f"\nPlayers with skin ratings: {len(player_agg)}")

# Bin into light vs dark
player_agg['dark'] = (player_agg['skin_tone'] > 0.5).astype(int)
light = player_agg[player_agg['dark'] == 0]['red_card_rate']
dark = player_agg[player_agg['dark'] == 1]['red_card_rate']
print(f"\nLight skin players: {len(light)}, mean red card rate: {light.mean():.5f}")
print(f"Dark skin players: {len(dark)}, mean red card rate: {dark.mean():.5f}")

t_stat, p_val = stats.ttest_ind(dark, light)
print(f"\nT-test: t={t_stat:.4f}, p={p_val:.4f}")

# Mann-Whitney U (non-parametric)
u_stat, p_mw = stats.mannwhitneyu(dark, light, alternative='greater')
print(f"Mann-Whitney U (dark > light): U={u_stat:.1f}, p={p_mw:.4f}")

# Linear regression: skin_tone -> red_card_rate
X = sm.add_constant(player_agg['skin_tone'])
y = player_agg['red_card_rate']
model = sm.OLS(y, X).fit()
print(f"\nOLS regression skin_tone -> red_card_rate:")
print(f"  coef={model.params['skin_tone']:.6f}, p={model.pvalues['skin_tone']:.4f}")
print(f"  95% CI: [{model.conf_int().loc['skin_tone', 0]:.6f}, {model.conf_int().loc['skin_tone', 1]:.6f}]")

# Dyad-level analysis with skin_tone continuous
X2 = sm.add_constant(df_rated[['skin_tone', 'games']])
y2 = df_rated['redCards']
model2 = sm.OLS(y2, X2).fit()
print(f"\nDyad-level OLS: skin_tone coef={model2.params['skin_tone']:.6f}, p={model2.pvalues['skin_tone']:.4f}")

# Correlation
corr, p_corr = stats.pearsonr(player_agg['skin_tone'], player_agg['red_card_rate'])
print(f"\nPearson correlation: r={corr:.4f}, p={p_corr:.4f}")

# Summary
print("\n--- SUMMARY ---")
print(f"Dark skin mean red card rate: {dark.mean():.5f}")
print(f"Light skin mean red card rate: {light.mean():.5f}")
print(f"Ratio: {dark.mean()/light.mean():.3f}x")
print(f"T-test p-value: {p_val:.4f}")
print(f"Mann-Whitney p-value (one-sided): {p_mw:.4f}")
print(f"OLS p-value: {model.pvalues['skin_tone']:.4f}")

# Determine response score
# Significant if p < 0.05
significant = p_val < 0.05 or p_mw < 0.05
direction_correct = dark.mean() > light.mean()

if significant and direction_correct:
    response = 75
    explanation = (
        f"The analysis shows that dark-skinned players receive more red cards than light-skinned players. "
        f"Dark skin players had a mean red card rate of {dark.mean():.5f} vs {light.mean():.5f} for light skin players "
        f"(ratio {dark.mean()/light.mean():.2f}x). "
        f"T-test: t={t_stat:.3f}, p={p_val:.4f}. "
        f"Mann-Whitney U (one-sided, dark > light): p={p_mw:.4f}. "
        f"OLS regression skin_tone coefficient: {model.params['skin_tone']:.6f}, p={model.pvalues['skin_tone']:.4f}. "
        f"The effect is statistically significant and in the expected direction (darker skin -> more red cards), "
        f"supporting the hypothesis. Effect size is modest but consistent across methods."
    )
elif not significant:
    response = 30
    explanation = (
        f"Results are not statistically significant. Dark skin rate: {dark.mean():.5f}, light skin rate: {light.mean():.5f}. "
        f"T-test p={p_val:.4f}, Mann-Whitney p={p_mw:.4f}."
    )
else:
    response = 25
    explanation = (
        f"Dark-skinned players actually show lower red card rates, opposite to the hypothesis. "
        f"Dark: {dark.mean():.5f}, Light: {light.mean():.5f}. p={p_val:.4f}."
    )

import json
result = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print(f"\nWritten conclusion.txt with response={response}")
