import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json

df = pd.read_csv("crofoot.csv")
print("Shape:", df.shape)
print(df.describe())
print("\nWin rate:", df['win'].mean())

# Feature engineering
df['size_ratio'] = df['n_focal'] / df['n_other']
df['log_size_ratio'] = np.log(df['size_ratio'])
df['location_advantage'] = df['dist_other'] - df['dist_focal']  # positive = focal closer to home
df['focal_home_advantage'] = (df['dist_focal'] < df['dist_other']).astype(int)

# --- Group size effect ---
wins = df[df['win'] == 1]['size_ratio']
losses = df[df['win'] == 0]['size_ratio']
t_stat, p_size = stats.ttest_ind(wins, losses)
print(f"\nSize ratio (wins vs losses): {wins.mean():.3f} vs {losses.mean():.3f}, p={p_size:.4f}")

# --- Location effect ---
wins_loc = df[df['win'] == 1]['location_advantage']
losses_loc = df[df['win'] == 0]['location_advantage']
t_loc, p_loc = stats.ttest_ind(wins_loc, losses_loc)
print(f"Location advantage (wins vs losses): {wins_loc.mean():.1f} vs {losses_loc.mean():.1f}, p={p_loc:.4f}")

# Home advantage test
contingency = pd.crosstab(df['focal_home_advantage'], df['win'])
print("\nContingency table (home advantage vs win):\n", contingency)
chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
print(f"Chi2={chi2:.3f}, p={p_chi:.4f}")

# --- Logistic regression ---
df['const'] = 1.0
X = df[['log_size_ratio', 'location_advantage']]
X = sm.add_constant(X)
logit_model = sm.Logit(df['win'], X).fit(disp=0)
print("\nLogistic regression summary:")
print(logit_model.summary())

p_size_logit = logit_model.pvalues['log_size_ratio']
p_loc_logit = logit_model.pvalues['location_advantage']
print(f"\nLog size ratio p-value: {p_size_logit:.4f}")
print(f"Location advantage p-value: {p_loc_logit:.4f}")

# Both significant?
both_significant = (p_size_logit < 0.05) and (p_loc_logit < 0.05)
either_significant = (p_size_logit < 0.05) or (p_loc_logit < 0.05)

# Score: both significant = ~85, one = ~55, neither = ~15
if both_significant:
    score = 85
    explanation = (
        f"Both relative group size (log size ratio, p={p_size_logit:.4f}) and contest location "
        f"(distance advantage, p={p_loc_logit:.4f}) significantly predict win probability in logistic regression. "
        f"Larger focal groups win more (size ratio wins={wins.mean():.2f} vs losses={losses.mean():.2f}), "
        f"and focal groups fight better when closer to their home range center "
        f"(location advantage wins={wins_loc.mean():.1f}m vs losses={losses_loc.mean():.1f}m). "
        f"Both factors jointly influence contest outcomes."
    )
elif p_size_logit < 0.05:
    score = 60
    explanation = (
        f"Relative group size significantly predicts win probability (p={p_size_logit:.4f}) but "
        f"contest location does not reach significance alone (p={p_loc_logit:.4f}). "
        f"Partial support for the research question."
    )
elif p_loc_logit < 0.05:
    score = 60
    explanation = (
        f"Contest location significantly predicts win probability (p={p_loc_logit:.4f}) but "
        f"relative group size is not significant alone (p={p_size_logit:.4f}). "
        f"Partial support for the research question."
    )
else:
    score = 20
    explanation = (
        f"Neither relative group size (p={p_size_logit:.4f}) nor contest location (p={p_loc_logit:.4f}) "
        f"significantly predicts win probability. Little evidence for the proposed relationships."
    )

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\n--- CONCLUSION ---")
print(json.dumps(result, indent=2))
