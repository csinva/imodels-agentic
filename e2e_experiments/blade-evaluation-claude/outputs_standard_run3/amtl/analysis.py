import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('amtl.csv')
print("Shape:", df.shape)
print(df.head())
print(df.dtypes)
print(df['genus'].value_counts())

# Compute AMTL rate
df['amtl_rate'] = df['num_amtl'] / df['sockets']
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

print("\nAMTL rate by genus:")
print(df.groupby('genus')['amtl_rate'].describe())

# Encode tooth class
tooth_dummies = pd.get_dummies(df['tooth_class'], drop_first=True, prefix='tooth')
df = pd.concat([df, tooth_dummies], axis=1)

# --- Logistic regression (binomial) using statsmodels ---
# Model: AMTL ~ is_human + age + prob_male + tooth_class
# Use binomial with n_trials = sockets
print("\n--- Binomial regression ---")
formula = 'amtl_rate ~ is_human + age + prob_male + C(tooth_class)'
try:
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df['sockets']
    )
    result = model.fit()
    print(result.summary())
    human_coef = result.params['is_human']
    human_pval = result.pvalues['is_human']
    human_ci = result.conf_int().loc['is_human']
    print(f"\nis_human coef: {human_coef:.4f}, p-value: {human_pval:.4e}")
    print(f"95% CI: [{human_ci[0]:.4f}, {human_ci[1]:.4f}]")
except Exception as e:
    print("GLM error:", e)
    human_coef = None
    human_pval = None

# --- Also test with genus dummies ---
print("\n--- Binomial regression with genus dummies ---")
formula2 = 'amtl_rate ~ C(genus) + age + prob_male + C(tooth_class)'
try:
    model2 = smf.glm(
        formula=formula2,
        data=df,
        family=sm.families.Binomial(),
        freq_weights=df['sockets']
    )
    result2 = model2.fit()
    print(result2.summary())
except Exception as e:
    print("GLM2 error:", e)

# --- Simple comparison: Homo vs non-human primates ---
human_rates = df[df['is_human'] == 1]['amtl_rate']
nonhuman_rates = df[df['is_human'] == 0]['amtl_rate']
tstat, pval_t = stats.ttest_ind(human_rates, nonhuman_rates)
print(f"\nSimple t-test: t={tstat:.4f}, p={pval_t:.4e}")
print(f"Human mean AMTL rate: {human_rates.mean():.4f}")
print(f"Non-human mean AMTL rate: {nonhuman_rates.mean():.4f}")

# --- Ridge regression for feature importances ---
print("\n--- Ridge regression for feature importances ---")
features = ['is_human', 'age', 'prob_male'] + list(tooth_dummies.columns)
X = df[features].fillna(0)
y = df['amtl_rate']
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
for feat, coef in zip(features, ridge.coef_):
    print(f"  {feat}: {coef:.4f}")

# Determine conclusion
# Use binomial regression result for is_human
if human_pval is not None:
    significant = human_pval < 0.05
    positive_effect = human_coef > 0
    if significant and positive_effect:
        score = 90
        explanation = (
            f"Yes. Binomial regression shows humans have significantly higher AMTL rates "
            f"after controlling for age, sex, and tooth class (coef={human_coef:.3f}, "
            f"p={human_pval:.2e}). Human mean AMTL rate ({human_rates.mean():.3f}) is "
            f"substantially higher than non-human primates ({nonhuman_rates.mean():.3f})."
        )
    elif significant and not positive_effect:
        score = 10
        explanation = (
            f"No. Humans have significantly LOWER AMTL rates after controlling for covariates "
            f"(coef={human_coef:.3f}, p={human_pval:.2e})."
        )
    else:
        score = 30
        explanation = (
            f"No significant difference in AMTL rates between humans and non-human primates "
            f"after controlling for age, sex, and tooth class (coef={human_coef:.3f}, "
            f"p={human_pval:.2e})."
        )
else:
    # Fallback to t-test
    if pval_t < 0.05 and human_rates.mean() > nonhuman_rates.mean():
        score = 80
        explanation = (
            f"Yes. T-test shows humans have significantly higher AMTL rates "
            f"(human={human_rates.mean():.3f}, non-human={nonhuman_rates.mean():.3f}, p={pval_t:.2e})."
        )
    else:
        score = 30
        explanation = "Insufficient evidence for higher AMTL in humans."

import json
conclusion = {"response": score, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(conclusion, f)

print(f"\nConclusion: {conclusion}")
