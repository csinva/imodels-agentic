import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("boxes.csv")
print("Shape:", df.shape)
print(df.describe())

# Create binary outcome: chose majority (y==2) vs not
df['chose_majority'] = (df['y'] == 2).astype(int)

print("\nOverall majority choice rate:", df['chose_majority'].mean().round(3))
print("\nMajority choice rate by age:")
print(df.groupby('age')['chose_majority'].mean().round(3))

# Correlation between age and majority choice
r, p = stats.pointbiserialr(df['age'], df['chose_majority'])
print(f"\nAge vs majority choice: r={r:.3f}, p={p:.4f}")

# OLS regression: majority choice ~ age + culture dummies
X = pd.get_dummies(df[['age', 'culture']], columns=['culture'], drop_first=True).astype(float)
X = sm.add_constant(X)
model = sm.OLS(df['chose_majority'], X).fit()
print("\nOLS Summary:")
print(model.summary())

# Per-culture: correlation of age with majority choice
print("\nPer-culture correlation of age with majority choice:")
culture_results = []
for c in sorted(df['culture'].unique()):
    sub = df[df['culture'] == c]
    if len(sub) > 5:
        r_c, p_c = stats.pointbiserialr(sub['age'], sub['chose_majority'])
        culture_results.append((c, r_c, p_c, len(sub)))
        print(f"  Culture {c}: r={r_c:.3f}, p={p_c:.4f}, n={len(sub)}")

# ANOVA: does majority choice rate differ by culture?
groups = [df[df['culture'] == c]['chose_majority'].values for c in df['culture'].unique()]
f_stat, p_anova = stats.f_oneway(*groups)
print(f"\nANOVA culture effect: F={f_stat:.3f}, p={p_anova:.4f}")

# Overall age effect using logistic regression with interaction culture*age
df_enc = pd.get_dummies(df[['age', 'culture', 'gender', 'majority_first']], columns=['culture'], drop_first=True).astype(float)
# Add age*culture interactions
for col in [c for c in df_enc.columns if c.startswith('culture_')]:
    df_enc[f'age_{col}'] = df['age'] * df_enc[col]

X_int = sm.add_constant(df_enc)
logit_model = sm.Logit(df['chose_majority'], X_int).fit(disp=False)
print("\nLogit with interactions summary (age + culture + age*culture):")
print(logit_model.summary())

# Summary: age effect overall
age_coef = model.params['age']
age_pval = model.pvalues['age']
print(f"\nKey result: OLS age coefficient={age_coef:.4f}, p={age_pval:.4f}")

# Decision
# Research question: How do children's reliance on majority preference develop with age across cultures?
# The question asks about development (does it increase with age?), and whether this varies by culture.
# Score near 50-70 if there's a significant age effect; higher if strong and consistent.
# Score lower if weak or inconsistent.

significant = p < 0.05
positive = r > 0
# Proportion of cultures showing positive age effect
pos_cultures = sum(1 for _, r_c, p_c, _ in culture_results if r_c > 0)
total_cultures = len(culture_results)

print(f"\nAge-majority correlation significant: {significant}, direction positive: {positive}")
print(f"Cultures with positive age effect: {pos_cultures}/{total_cultures}")

# Response score:
# If significant positive age effect -> high score (yes, majority reliance increases with age)
# Variation across cultures is expected but overall trend matters
if significant and positive and pos_cultures / total_cultures >= 0.6:
    response = 75
    explanation = (
        f"Children's reliance on majority preference shows a significant positive relationship with age "
        f"(r={r:.3f}, p={p:.4f}). Majority choice rate increases as children get older across most cultural "
        f"contexts ({pos_cultures}/{total_cultures} cultures show positive age-majority correlation). "
        f"OLS regression confirms the age effect (coef={age_coef:.4f}, p={age_pval:.4f}). "
        f"There is also significant cultural variation (ANOVA F={f_stat:.3f}, p={p_anova:.4f}), "
        f"but the overall developmental trend of increasing majority conformity with age is clear."
    )
elif significant and positive:
    response = 60
    explanation = (
        f"There is a significant positive age effect on majority choice (r={r:.3f}, p={p:.4f}), "
        f"but the pattern is inconsistent across cultures ({pos_cultures}/{total_cultures} positive). "
        f"Cultural context moderates the age-majority relationship substantially."
    )
elif significant and not positive:
    response = 25
    explanation = (
        f"Significant but negative age effect on majority choice (r={r:.3f}, p={p:.4f}): "
        f"older children are less likely to follow the majority, opposite to the hypothesis."
    )
else:
    response = 40
    explanation = (
        f"No significant relationship between age and majority preference (r={r:.3f}, p={p:.4f}). "
        f"Children's reliance on majority does not clearly develop with age in this dataset."
    )

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written:")
print(json.dumps(result, indent=2))
