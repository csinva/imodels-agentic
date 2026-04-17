import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
import json

# Load data
df = pd.read_csv("panda_nuts.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())
print(df.dtypes)

# Compute efficiency: nuts per second
df['efficiency'] = df['nuts_opened'] / df['seconds']

print("\nEfficiency stats:")
print(df['efficiency'].describe())

# Encode categoricals
df['sex_enc'] = (df['sex'] == 'm').astype(int)
df['help_enc'] = (df['help'] == 'y').astype(int)

# --- Age vs efficiency ---
corr_age, p_age = stats.pearsonr(df['age'], df['efficiency'])
print(f"\nAge-efficiency Pearson r={corr_age:.3f}, p={p_age:.4f}")

# --- Sex vs efficiency ---
male = df[df['sex'] == 'm']['efficiency']
female = df[df['sex'] == 'f']['efficiency']
t_sex, p_sex = stats.ttest_ind(male, female)
print(f"Sex t-test: t={t_sex:.3f}, p={p_sex:.4f}")
print(f"  Male mean={male.mean():.3f}, Female mean={female.mean():.3f}")

# --- Help vs efficiency ---
helped = df[df['help'] == 'y']['efficiency']
not_helped = df[df['help'] == 'N']['efficiency']
t_help, p_help = stats.ttest_ind(helped, not_helped)
print(f"Help t-test: t={t_help:.3f}, p={p_help:.4f}")
print(f"  Helped mean={helped.mean():.3f}, Not helped mean={not_helped.mean():.3f}")

# --- OLS regression ---
X = df[['age', 'sex_enc', 'help_enc']].copy()
X = sm.add_constant(X)
model = sm.OLS(df['efficiency'], X).fit()
print("\nOLS summary:")
print(model.summary())

# --- Ridge regression for feature importances ---
Xr = df[['age', 'sex_enc', 'help_enc']].values
yr = df['efficiency'].values
ridge = Ridge(alpha=1.0)
ridge.fit(Xr, yr)
print("\nRidge coefficients:")
for name, coef in zip(['age', 'sex_enc', 'help_enc'], ridge.coef_):
    print(f"  {name}: {coef:.4f}")

# Summarize significance
sig_age = p_age < 0.05
sig_sex = p_sex < 0.05
sig_help = p_help < 0.05

print(f"\nSignificant predictors: age={sig_age}, sex={sig_sex}, help={sig_help}")

# Build conclusion score
# Research question: How do age, sex, and receiving help influence nut-cracking efficiency?
# We rate whether there is a meaningful influence (any of the three variables significant)
n_sig = sum([sig_age, sig_sex, sig_help])

# Compute effect size for age (Pearson r)
# Compute Cohen's d for sex and help
def cohens_d(a, b):
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    return abs(a.mean() - b.mean()) / pooled_std if pooled_std > 0 else 0

d_sex = cohens_d(male, female)
d_help = cohens_d(helped, not_helped)
print(f"\nEffect sizes: age r={corr_age:.3f}, sex d={d_sex:.3f}, help d={d_help:.3f}")

# Score: if age and help are significant with meaningful effects => high score
# Age significant + help significant => strong yes (~80+)
# Only one significant => moderate (~50-65)
# None significant => low (~15-25)

if sig_age and sig_help:
    score = 82
    explanation = (
        f"Age significantly correlates with nut-cracking efficiency (r={corr_age:.3f}, p={p_age:.4f}), "
        f"and receiving help also significantly affects efficiency (t={t_help:.3f}, p={p_help:.4f}, d={d_help:.3f}). "
        f"Sex was not significant (p={p_sex:.4f}). OLS confirms age (p={model.pvalues['age']:.4f}) and help "
        f"(p={model.pvalues['help_enc']:.4f}) as predictors. Overall, age and help clearly influence efficiency."
    )
elif sig_age or sig_help:
    score = 65
    explanation = (
        f"Partial influence detected. Age r={corr_age:.3f} (p={p_age:.4f}), sex p={p_sex:.4f}, help p={p_help:.4f}. "
        f"Only some predictors reach significance."
    )
elif sig_sex:
    score = 50
    explanation = f"Only sex is significant (p={p_sex:.4f}). Age p={p_age:.4f}, help p={p_help:.4f}."
else:
    score = 20
    explanation = (
        f"No predictor is statistically significant at 0.05 level. "
        f"Age p={p_age:.4f}, sex p={p_sex:.4f}, help p={p_help:.4f}."
    )

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print(f"\nConclusion written: score={score}")
print(explanation)
