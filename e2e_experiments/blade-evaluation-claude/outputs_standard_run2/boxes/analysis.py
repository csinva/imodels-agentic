import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('boxes.csv')
print(df.describe())
print(df['y'].value_counts())

# Create binary: did child follow majority? (y==2)
df['majority_choice'] = (df['y'] == 2).astype(int)

print("\nMajority choice rate by age:")
age_group = df.groupby('age')['majority_choice'].mean()
print(age_group)

print("\nMajority choice rate by culture:")
cult_group = df.groupby('culture')['majority_choice'].mean()
print(cult_group)

# Logistic regression: majority_choice ~ age + culture + age*culture
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# OLS on majority_choice with age, culture, interaction
df['age_z'] = (df['age'] - df['age'].mean()) / df['age'].std()
df['culture_z'] = (df['culture'] - df['culture'].mean()) / df['culture'].std()
df['age_culture'] = df['age_z'] * df['culture_z']

X = sm.add_constant(df[['age_z', 'culture_z', 'age_culture']])
model = sm.OLS(df['majority_choice'], X).fit()
print("\nOLS summary:")
print(model.summary())

# Spearman correlation: age vs majority_choice
rho, pval = stats.spearmanr(df['age'], df['majority_choice'])
print(f"\nSpearman age vs majority_choice: rho={rho:.3f}, p={pval:.4f}")

# ANOVA: does majority rate differ by culture?
culture_groups = [df[df['culture'] == c]['majority_choice'].values for c in df['culture'].unique()]
f_stat, f_pval = stats.f_oneway(*culture_groups)
print(f"ANOVA culture effect: F={f_stat:.3f}, p={f_pval:.4f}")

# Interaction: correlation of age vs majority within each culture
print("\nAge-majority correlation by culture:")
for c in sorted(df['culture'].unique()):
    sub = df[df['culture'] == c]
    r, p = stats.spearmanr(sub['age'], sub['majority_choice'])
    print(f"  culture {c}: rho={r:.3f}, p={p:.4f}, n={len(sub)}")

# Age trend test: does majority_choice increase with age?
age_corr_p = pval
age_coef = model.params['age_z']
age_p = model.pvalues['age_z']
culture_p = model.pvalues['culture_z']
interaction_p = model.pvalues['age_culture']

print(f"\nKey results:")
print(f"Age effect coef={age_coef:.3f}, p={age_p:.4f}")
print(f"Culture effect p={culture_p:.4f}")
print(f"Age x Culture interaction p={interaction_p:.4f}")

# Determine response score
# Research question: How do children's reliance on majority preference develop over age across cultural contexts?
# Key: is there an age effect? Is there a cultural moderation?

# Age effect significant?
age_sig = age_p < 0.05
culture_sig = culture_p < 0.05
interaction_sig = interaction_p < 0.05

# Overall majority rate
maj_rate = df['majority_choice'].mean()
print(f"\nOverall majority choice rate: {maj_rate:.3f}")

# The question asks if reliance on majority DEVELOPS (increases) with age across cultures
# Score reflects strength of evidence for age-related development of majority preference
if age_sig and rho > 0:
    base_score = 75
elif age_sig and rho <= 0:
    base_score = 30
elif age_p < 0.10:
    base_score = 55
else:
    base_score = 25

# Adjust for cultural moderation
if interaction_sig:
    explanation_suffix = (
        f"Age significantly predicts majority preference (p={age_p:.3f}), "
        f"and the effect varies significantly across cultures (interaction p={interaction_p:.3f}). "
        f"Overall majority choice rate is {maj_rate:.2f}. "
        f"Spearman rho(age, majority)={rho:.3f} (p={pval:.4f})."
    )
    score = base_score
elif culture_sig:
    explanation_suffix = (
        f"Age effect p={age_p:.3f} (rho={rho:.3f}), culture effect p={culture_p:.3f}. "
        f"No significant age-culture interaction (p={interaction_p:.3f}). "
        f"Overall majority rate={maj_rate:.2f}."
    )
    score = base_score
else:
    explanation_suffix = (
        f"Age effect p={age_p:.3f} (rho={rho:.3f}), culture p={culture_p:.3f}, "
        f"interaction p={interaction_p:.3f}. Overall majority rate={maj_rate:.2f}."
    )
    score = base_score

print(f"\nFinal score: {score}")

conclusion = {
    "response": score,
    "explanation": (
        f"Children's majority preference reliance shows developmental change with age. "
        + explanation_suffix +
        f" Age coefficient in OLS: {age_coef:.3f}. "
        f"ANOVA culture effect: F={f_stat:.3f}, p={f_pval:.4f}."
    )
}

with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print("conclusion.txt written.")
print(json.dumps(conclusion, indent=2))
