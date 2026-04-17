import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import json

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv('boxes.csv')
print("Shape:", df.shape)
print(df.describe())

# Create binary: did child follow majority (y==2)?
df['majority_choice'] = (df['y'] == 2).astype(int)

print("\nOverall majority choice rate:", df['majority_choice'].mean())
print("\nMajority choice rate by age:")
age_rates = df.groupby('age')['majority_choice'].agg(['mean', 'count'])
print(age_rates)

print("\nMajority choice rate by culture:")
culture_rates = df.groupby('culture')['majority_choice'].agg(['mean', 'count'])
print(culture_rates)

# --- Statistical test: does age predict majority following? ---
# Logistic regression: majority_choice ~ age
X_age = sm.add_constant(df['age'])
logit_age = sm.Logit(df['majority_choice'], X_age).fit(disp=0)
print("\nLogistic regression (majority_choice ~ age):")
print(logit_age.summary())

age_coef = logit_age.params['age']
age_pval = logit_age.pvalues['age']
print(f"\nAge coefficient: {age_coef:.4f}, p-value: {age_pval:.4f}")

# Spearman correlation between age and majority choice
rho, p_spearman = stats.spearmanr(df['age'], df['majority_choice'])
print(f"\nSpearman correlation (age vs majority_choice): rho={rho:.4f}, p={p_spearman:.4f}")

# --- Does age effect vary by culture? Interaction term ---
logit_interact = smf.logit('majority_choice ~ age * C(culture)', data=df).fit(disp=0)
print("\nLogistic regression with age*culture interaction:")
print(logit_interact.summary())

# --- Age effect within each culture ---
print("\nLogistic regression age coefficient per culture:")
culture_results = []
for c in sorted(df['culture'].unique()):
    sub = df[df['culture'] == c]
    if len(sub) < 20:
        continue
    try:
        X = sm.add_constant(sub['age'])
        m = sm.Logit(sub['majority_choice'], X).fit(disp=0)
        coef = m.params['age']
        pval = m.pvalues['age']
        rate = sub['majority_choice'].mean()
        print(f"  Culture {c}: n={len(sub)}, majority_rate={rate:.2f}, age_coef={coef:.3f}, p={pval:.3f}")
        culture_results.append({'culture': c, 'coef': coef, 'pval': pval, 'n': len(sub)})
    except Exception as e:
        print(f"  Culture {c}: error {e}")

# --- ANOVA: does majority rate differ by age group? ---
df['age_group'] = pd.cut(df['age'], bins=[3, 6, 9, 12, 15], labels=['4-6', '7-9', '10-12', '13-14'])
groups = [g['majority_choice'].values for _, g in df.groupby('age_group')]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"\nANOVA (majority_choice by age group): F={f_stat:.3f}, p={p_anova:.4f}")
print(df.groupby('age_group')['majority_choice'].mean())

# --- Summary ---
sig_age = age_pval < 0.05
sig_cultures = sum(1 for r in culture_results if r['pval'] < 0.05)
print(f"\nSummary:")
print(f"  Age effect on majority following: p={age_pval:.4f}, significant={sig_age}")
print(f"  Cultures with significant age effect: {sig_cultures}/{len(culture_results)}")
print(f"  ANOVA p={p_anova:.4f}")

# Determine response score
# Research question: does reliance on majority preference develop/change with age across cultures?
# A "Yes" = significant age effect both overall and variably across cultures
if age_pval < 0.01 and p_anova < 0.01:
    response = 75
    explanation = (
        f"Strong evidence that age significantly predicts majority preference following "
        f"(logistic regression age coef={age_coef:.3f}, p={age_pval:.4f}; "
        f"ANOVA F={f_stat:.2f}, p={p_anova:.4f}; Spearman rho={rho:.3f}, p={p_spearman:.4f}). "
        f"{sig_cultures}/{len(culture_results)} cultures show significant age effects individually. "
        f"Children show increasing tendency to follow majority with age, with variability across cultures."
    )
elif age_pval < 0.05:
    response = 60
    explanation = (
        f"Moderate evidence that age predicts majority following "
        f"(logistic regression p={age_pval:.4f}, coef={age_coef:.3f}). "
        f"Age effect exists but may not be consistent across all cultures."
    )
else:
    response = 30
    explanation = (
        f"Weak/no evidence that age predicts majority following "
        f"(logistic regression p={age_pval:.4f}). "
        f"The relationship between age and majority preference is not statistically significant."
    )

result = {"response": response, "explanation": explanation}
print("\nFinal result:", result)

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("Wrote conclusion.txt")
