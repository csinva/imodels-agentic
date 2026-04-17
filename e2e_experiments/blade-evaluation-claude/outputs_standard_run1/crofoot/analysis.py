import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json

df = pd.read_csv('crofoot.csv')

print("Shape:", df.shape)
print(df.describe())

# Derived features: relative group size and contest location
df['rel_size'] = df['n_focal'] / df['n_other']
df['loc_advantage'] = df['dist_other'] - df['dist_focal']  # positive = focal closer to own range

print("\nCorrelation with win:")
print(df[['win','rel_size','loc_advantage','dist_focal','dist_other','n_focal','n_other']].corr()['win'])

# Statistical tests
wins = df[df['win']==1]
losses = df[df['win']==0]

t_size, p_size = stats.ttest_ind(wins['rel_size'], losses['rel_size'])
t_loc, p_loc = stats.ttest_ind(wins['loc_advantage'], losses['loc_advantage'])
print(f"\nRelative size t-test: t={t_size:.3f}, p={p_size:.4f}")
print(f"Location advantage t-test: t={t_loc:.3f}, p={p_loc:.4f}")

# Logistic regression with both predictors
X = df[['rel_size', 'loc_advantage']]
X = sm.add_constant(X)
y = df['win']
logit_model = sm.Logit(y, X).fit(disp=0)
print("\nLogistic regression summary:")
print(logit_model.summary())

# Individual logistic regressions
X_size = sm.add_constant(df[['rel_size']])
m_size = sm.Logit(y, X_size).fit(disp=0)
X_loc = sm.add_constant(df[['loc_advantage']])
m_loc = sm.Logit(y, X_loc).fit(disp=0)

p_size_logit = m_size.pvalues['rel_size']
p_loc_logit = m_loc.pvalues['loc_advantage']
coef_size = m_size.params['rel_size']
coef_loc = m_loc.params['loc_advantage']

print(f"\nSize-only logit: coef={coef_size:.3f}, p={p_size_logit:.4f}")
print(f"Location-only logit: coef={coef_loc:.4f}, p={p_loc_logit:.4f}")

# Decision tree for interpretability
from sklearn.tree import DecisionTreeClassifier, export_text
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(df[['rel_size','loc_advantage']], y)
print("\nDecision tree:")
print(export_text(dt, feature_names=['rel_size','loc_advantage']))
print("Feature importances:", dict(zip(['rel_size','loc_advantage'], dt.feature_importances_)))

# Both factors are significant - determine score
# rel_size: significant (p < 0.05), loc_advantage: significant (p < 0.05)
both_significant = (p_size_logit < 0.05) and (p_loc_logit < 0.05)
size_sig = p_size_logit < 0.05
loc_sig = p_loc_logit < 0.05

print(f"\nSize significant: {size_sig} (p={p_size_logit:.4f})")
print(f"Location significant: {loc_sig} (p={p_loc_logit:.4f})")

# Research question asks about BOTH relative group size AND contest location
# Score reflects whether both influence win probability
if both_significant:
    score = 85
    explanation = (f"Both relative group size (logit coef={coef_size:.2f}, p={p_size_logit:.4f}) "
                   f"and contest location advantage (logit coef={coef_loc:.4f}, p={p_loc_logit:.4f}) "
                   f"significantly predict win probability. Larger relative size and being closer to own "
                   f"home range (higher loc_advantage) both increase winning odds. "
                   f"Decision tree confirms both features are used. Strong Yes.")
elif size_sig or loc_sig:
    score = 60
    explanation = (f"At least one factor is significant. "
                   f"Relative size: p={p_size_logit:.4f}, Location: p={p_loc_logit:.4f}.")
else:
    score = 20
    explanation = "Neither factor reaches significance."

print(f"\nScore: {score}")
print(f"Explanation: {explanation}")

result = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)
print("\nWrote conclusion.txt")
