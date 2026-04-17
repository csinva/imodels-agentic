import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("/home/chansingh/imodels-evolve/e2e_experiments/blade-evaluation-claude/outputs_standard_run3/crofoot/crofoot.csv")
print("Shape:", df.shape)
print(df.describe())
print("\nWin rate:", df['win'].mean())

# Derived features
df['rel_size'] = df['n_focal'] / df['n_other']          # relative group size
df['log_rel_size'] = np.log(df['rel_size'])
df['loc_advantage'] = df['dist_other'] - df['dist_focal']  # positive = focal closer to home
df['rel_male'] = df['m_focal'] / df['m_other']

print("\n--- Correlation with win ---")
for col in ['rel_size', 'log_rel_size', 'loc_advantage', 'dist_focal', 'dist_other', 'rel_male']:
    r, p = stats.pointbiserialr(df[col], df['win'])
    print(f"  {col}: r={r:.3f}, p={p:.4f}")

# Split by win/loss to inspect group size and location
wins = df[df['win'] == 1]
losses = df[df['win'] == 0]

print("\n--- Group size: winners vs losers ---")
t_size, p_size = stats.ttest_ind(wins['rel_size'], losses['rel_size'])
print(f"  Mean rel_size (win): {wins['rel_size'].mean():.3f}, (loss): {losses['rel_size'].mean():.3f}")
print(f"  t={t_size:.3f}, p={p_size:.4f}")

print("\n--- Location advantage: winners vs losers ---")
t_loc, p_loc = stats.ttest_ind(wins['loc_advantage'], losses['loc_advantage'])
print(f"  Mean loc_advantage (win): {wins['loc_advantage'].mean():.1f}, (loss): {losses['loc_advantage'].mean():.1f}")
print(f"  t={t_loc:.3f}, p={p_loc:.4f}")

# Logistic regression with statsmodels for p-values
X = df[['log_rel_size', 'loc_advantage']].copy()
X = sm.add_constant(X)
y = df['win']

logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=0)
print("\n--- Logistic Regression (statsmodels) ---")
print(result.summary())

# Extract p-values
pval_size = result.pvalues['log_rel_size']
pval_loc = result.pvalues['loc_advantage']
print(f"\np-value rel_size: {pval_size:.4f}")
print(f"p-value loc_advantage: {pval_loc:.4f}")

# Interpret results
both_significant = (pval_size < 0.05) and (pval_loc < 0.05)
size_sig = pval_size < 0.05
loc_sig = pval_loc < 0.05

print(f"\nRelative group size significant: {size_sig}")
print(f"Contest location significant: {loc_sig}")

# Score: both factors matter -> high yes
if both_significant:
    score = 85
    explanation = (
        f"Both relative group size (log_rel_size coef={result.params['log_rel_size']:.3f}, "
        f"p={pval_size:.4f}) and contest location (loc_advantage coef={result.params['loc_advantage']:.4f}, "
        f"p={pval_loc:.4f}) are statistically significant predictors of winning. "
        f"Larger focal groups and contests closer to the focal group's home range both increase win probability."
    )
elif size_sig and not loc_sig:
    score = 55
    explanation = (
        f"Relative group size is significant (p={pval_size:.4f}) but location advantage is not (p={pval_loc:.4f}). "
        f"Only group size influences win probability in this sample."
    )
elif loc_sig and not size_sig:
    score = 55
    explanation = (
        f"Contest location is significant (p={pval_loc:.4f}) but relative group size is not (p={pval_size:.4f}). "
        f"Only location influences win probability in this sample."
    )
else:
    score = 20
    explanation = (
        f"Neither relative group size (p={pval_size:.4f}) nor location advantage (p={pval_loc:.4f}) "
        f"reaches statistical significance in logistic regression. Weak evidence for either factor."
    )

conclusion = {"response": score, "explanation": explanation}
with open("/home/chansingh/imodels-evolve/e2e_experiments/blade-evaluation-claude/outputs_standard_run3/crofoot/conclusion.txt", "w") as f:
    json.dump(conclusion, f)

print("\nconclusion.txt written:")
print(json.dumps(conclusion, indent=2))
