import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("panda_nuts.csv")
print(df.head())
print(df.describe())
print(df.dtypes)
print(df['sex'].value_counts())
print(df['help'].value_counts())

# Efficiency: nuts per second
df['efficiency'] = df['nuts_opened'] / df['seconds']

print("\nEfficiency stats:")
print(df['efficiency'].describe())

# Encode categoricals
df['sex_enc'] = (df['sex'] == 'm').astype(int)
df['help_enc'] = (df['help'] == 'y').astype(int)

# Correlation of age with efficiency
r_age, p_age = stats.pearsonr(df['age'], df['efficiency'])
print(f"\nAge vs efficiency: r={r_age:.3f}, p={p_age:.4f}")

# T-test: sex vs efficiency
male_eff = df[df['sex'] == 'm']['efficiency']
female_eff = df[df['sex'] == 'f']['efficiency']
t_sex, p_sex = stats.ttest_ind(male_eff, female_eff)
print(f"Sex vs efficiency: t={t_sex:.3f}, p={p_sex:.4f}")
print(f"Male mean={male_eff.mean():.4f}, Female mean={female_eff.mean():.4f}")

# T-test: help vs efficiency
help_y = df[df['help'] == 'y']['efficiency']
help_n = df[df['help'] == 'N']['efficiency']
t_help, p_help = stats.ttest_ind(help_y, help_n)
print(f"Help vs efficiency: t={t_help:.3f}, p={p_help:.4f}")
print(f"Help-yes mean={help_y.mean():.4f}, Help-no mean={help_n.mean():.4f}")

# OLS regression
X = df[['age', 'sex_enc', 'help_enc']].copy()
X = sm.add_constant(X)
model = sm.OLS(df['efficiency'], X).fit()
print("\nOLS summary:")
print(model.summary())

# Ridge regression for feature importance
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[['age', 'sex_enc', 'help_enc']])
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, df['efficiency'])
print("\nRidge coefficients (standardized):")
for name, coef in zip(['age', 'sex_enc', 'help_enc'], ridge.coef_):
    print(f"  {name}: {coef:.4f}")

# Summary findings
print("\n=== Summary ===")
print(f"Age: r={r_age:.3f}, p={p_age:.4f} -> {'significant' if p_age < 0.05 else 'not significant'}")
print(f"Sex: p={p_sex:.4f} -> {'significant' if p_sex < 0.05 else 'not significant'}")
print(f"Help: p={p_help:.4f} -> {'significant' if p_help < 0.05 else 'not significant'}")

# Determine overall response score
# Research question: How do age, sex, and help influence nut-cracking efficiency?
# Score based on how many factors are significant and magnitude of effects
sig_factors = []
if p_age < 0.05:
    sig_factors.append(f"age (r={r_age:.2f}, p={p_age:.4f})")
if p_sex < 0.05:
    sig_factors.append(f"sex (p={p_sex:.4f})")
if p_help < 0.05:
    sig_factors.append(f"help (p={p_help:.4f})")

ols_pvals = model.pvalues[1:]  # exclude intercept
any_sig = any(p < 0.05 for p in ols_pvals)

# Score: if all 3 factors are significant -> high score; if none -> low
n_sig = sum([p_age < 0.05, p_sex < 0.05, p_help < 0.05])
if n_sig == 3:
    score = 90
elif n_sig == 2:
    score = 75
elif n_sig == 1:
    score = 55
else:
    score = 20

explanation = (
    f"Analysis of nut-cracking efficiency (nuts/second) in {len(df)} observations. "
    f"Age correlation with efficiency: r={r_age:.3f}, p={p_age:.4f}. "
    f"Sex effect (male vs female): t={t_sex:.3f}, p={p_sex:.4f} "
    f"(male mean={male_eff.mean():.3f}, female mean={female_eff.mean():.3f}). "
    f"Help effect (help vs no help): t={t_help:.3f}, p={p_help:.4f} "
    f"(help-yes mean={help_y.mean():.3f}, help-no mean={help_n.mean():.3f}). "
    f"OLS regression R^2={model.rsquared:.3f}. "
    f"Significant factors: {sig_factors if sig_factors else 'none'}. "
    f"All three predictors (age, sex, help) {'jointly influence' if any_sig else 'do not clearly influence'} efficiency."
)

result = {"response": score, "explanation": explanation}
print(f"\nConclusion: {json.dumps(result, indent=2)}")

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("conclusion.txt written.")
