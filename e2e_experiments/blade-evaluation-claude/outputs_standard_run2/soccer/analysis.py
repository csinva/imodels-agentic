import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('soccer.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Create average skin tone measure
df['skin_tone'] = (df['rater1'] + df['rater2']) / 2
df_skin = df.dropna(subset=['skin_tone', 'redCards'])
print(f"\nRows with skin tone data: {len(df_skin)}")
print(f"\nSkin tone distribution:\n{df_skin['skin_tone'].value_counts().sort_index()}")
print(f"\nRedCards distribution:\n{df_skin['redCards'].describe()}")

# Aggregate by player (mean skin tone, total red cards, total games)
player_df = df_skin.groupby('playerShort').agg(
    skin_tone=('skin_tone', 'mean'),
    redCards=('redCards', 'sum'),
    games=('games', 'sum')
).reset_index()
player_df['red_card_rate'] = player_df['redCards'] / player_df['games']
print(f"\nPlayer-level dataset: {len(player_df)} players")

# Split into dark (>= 0.5) vs light (< 0.5)
dark = player_df[player_df['skin_tone'] >= 0.5]['red_card_rate']
light = player_df[player_df['skin_tone'] < 0.5]['red_card_rate']
print(f"\nDark skin players (skin_tone >= 0.5): n={len(dark)}, mean rate={dark.mean():.5f}")
print(f"Light skin players (skin_tone < 0.5): n={len(light)}, mean rate={light.mean():.5f}")

# T-test
t_stat, p_val = stats.ttest_ind(dark, light)
print(f"\nT-test: t={t_stat:.4f}, p={p_val:.4f}")

# Mann-Whitney U test (non-parametric)
u_stat, p_mw = stats.mannwhitneyu(dark, light, alternative='greater')
print(f"Mann-Whitney U (dark > light): U={u_stat:.1f}, p={p_mw:.4f}")

# OLS regression on dyad-level with skin_tone as predictor
df_model = df_skin[['skin_tone', 'redCards', 'games']].dropna()
X = sm.add_constant(df_model['skin_tone'])
y = df_model['redCards']
ols = sm.OLS(y, X).fit()
print(f"\nOLS on dyad-level redCards ~ skin_tone:")
print(f"  coef={ols.params['skin_tone']:.5f}, p={ols.pvalues['skin_tone']:.4f}")

# Poisson regression (more appropriate for count outcome)
from statsmodels.discrete.discrete_model import Poisson
df_pois = df_skin[['skin_tone', 'redCards', 'games']].dropna()
df_pois = df_pois[df_pois['games'] > 0]
X_p = sm.add_constant(df_pois['skin_tone'])
offset = np.log(df_pois['games'])
poisson_model = Poisson(df_pois['redCards'], X_p, offset=offset).fit(disp=False)
print(f"\nPoisson regression (offset=log(games)):")
print(f"  skin_tone coef={poisson_model.params['skin_tone']:.5f}, p={poisson_model.pvalues['skin_tone']:.4f}")
print(f"  IRR={np.exp(poisson_model.params['skin_tone']):.4f}")

# Feature importance via linear regression on player level
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

lr = LinearRegression()
X_lr = player_df[['skin_tone', 'games']].dropna()
y_lr = player_df.loc[X_lr.index, 'red_card_rate']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_lr)
lr.fit(X_scaled, y_lr)
print(f"\nLinear Regression coefficients (standardized):")
for feat, coef in zip(['skin_tone', 'games'], lr.coef_):
    print(f"  {feat}: {coef:.5f}")

# Summary stats
print(f"\n=== SUMMARY ===")
print(f"Dark skin mean red card rate: {dark.mean():.5f}")
print(f"Light skin mean red card rate: {light.mean():.5f}")
print(f"Ratio (dark/light): {dark.mean()/light.mean():.3f}")
print(f"T-test p-value: {p_val:.4f}")
print(f"Mann-Whitney p-value (one-sided): {p_mw:.4f}")
poisson_p = poisson_model.pvalues['skin_tone']
poisson_coef = poisson_model.params['skin_tone']
irr = np.exp(poisson_coef)

# Determine response score
# Poisson result is the most appropriate model
if poisson_p < 0.05 and poisson_coef > 0:
    response = 75
    explanation = (
        f"Yes, dark-skinned players receive more red cards. "
        f"Dark skin players (skin_tone>=0.5) had a mean red card rate of {dark.mean():.5f} vs "
        f"{light.mean():.5f} for light-skinned players (ratio={dark.mean()/light.mean():.3f}x). "
        f"Poisson regression (controlling for games played) shows a significant positive effect: "
        f"coef={poisson_coef:.4f}, IRR={irr:.4f}, p={poisson_p:.4f}. "
        f"T-test p={p_val:.4f}, Mann-Whitney one-sided p={p_mw:.4f}. "
        f"The evidence consistently supports that darker skin tone is associated with higher red card rates."
    )
elif p_val < 0.05:
    response = 60
    explanation = (
        f"Weak yes. T-test significant (p={p_val:.4f}) but Poisson not (p={poisson_p:.4f}). "
        f"Dark rate={dark.mean():.5f}, light rate={light.mean():.5f}."
    )
else:
    response = 25
    explanation = (
        f"No significant evidence. T-test p={p_val:.4f}, Poisson p={poisson_p:.4f}. "
        f"Dark rate={dark.mean():.5f}, light rate={light.mean():.5f}."
    )

import json
result = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)
print(f"\nWritten conclusion.txt with response={response}")
print(f"Explanation: {explanation}")
