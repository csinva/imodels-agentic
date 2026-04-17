import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import json
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('amtl.csv')
print("Shape:", df.shape)
print(df.head())
print(df['genus'].value_counts())

# Compute AMTL rate per row
df['amtl_rate'] = df['num_amtl'] / df['sockets']

# Binary: Homo sapiens vs non-human primates
df['is_human'] = (df['genus'] == 'Homo sapiens').astype(int)

print("\nMean AMTL rate by genus:")
print(df.groupby('genus')['amtl_rate'].mean())

print("\nMean AMTL rate by is_human:")
print(df.groupby('is_human')['amtl_rate'].mean())

# T-test: humans vs non-human primates
human_rates = df[df['is_human'] == 1]['amtl_rate']
nonhuman_rates = df[df['is_human'] == 0]['amtl_rate']
t_stat, p_val = stats.ttest_ind(human_rates, nonhuman_rates)
print(f"\nT-test: t={t_stat:.4f}, p={p_val:.6f}")

# Logistic/linear regression controlling for age, sex, tooth_class
df_model = df.copy()
df_model['tooth_anterior'] = (df_model['tooth_class'] == 'Anterior').astype(int)
df_model['tooth_posterior'] = (df_model['tooth_class'] == 'Posterior').astype(int)

X = df_model[['is_human', 'age', 'prob_male', 'tooth_anterior', 'tooth_posterior']]
X = sm.add_constant(X)
y = df_model['amtl_rate']

ols = sm.OLS(y, X).fit()
print("\nOLS Results:")
print(ols.summary())

is_human_coef = ols.params['is_human']
is_human_pval = ols.pvalues['is_human']
print(f"\nis_human coef: {is_human_coef:.4f}, p={is_human_pval:.6f}")

# Binomial GLM (logistic) using binomial counts
# Use genus dummies
df_model['genus_pan'] = (df_model['genus'] == 'Pan').astype(int)
df_model['genus_papio'] = (df_model['genus'] == 'Papio').astype(int)
df_model['genus_pongo'] = (df_model['genus'] == 'Pongo').astype(int)

try:
    glm_binom = smf.glm(
        'amtl_rate ~ is_human + age + prob_male + tooth_anterior + tooth_posterior',
        data=df_model,
        family=sm.families.Binomial()
    ).fit()
    print("\nBinomial GLM Results:")
    print(glm_binom.summary())
    glm_coef = glm_binom.params['is_human']
    glm_pval = glm_binom.pvalues['is_human']
    print(f"\nBinomial GLM is_human coef: {glm_coef:.4f}, p={glm_pval:.6f}")
except Exception as e:
    print(f"GLM error: {e}")
    glm_coef = is_human_coef
    glm_pval = is_human_pval

# Decision tree for interpretability
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder

df_tree = df_model[['is_human', 'age', 'prob_male', 'tooth_anterior', 'tooth_posterior', 'amtl_rate']].dropna()
X_tree = df_tree[['is_human', 'age', 'prob_male', 'tooth_anterior', 'tooth_posterior']]
y_tree = df_tree['amtl_rate']

tree = DecisionTreeRegressor(max_depth=4, random_state=42)
tree.fit(X_tree, y_tree)
feat_names = ['is_human', 'age', 'prob_male', 'tooth_anterior', 'tooth_posterior']
print("\nDecision Tree Feature Importances:")
for name, imp in sorted(zip(feat_names, tree.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.4f}")

# Summary
human_mean = human_rates.mean()
nonhuman_mean = nonhuman_rates.mean()
print(f"\nHuman AMTL rate: {human_mean:.4f}")
print(f"Non-human AMTL rate: {nonhuman_mean:.4f}")
print(f"Humans higher: {human_mean > nonhuman_mean}")
print(f"OLS p-value for is_human: {is_human_pval:.6f}")
print(f"T-test p-value: {p_val:.6f}")

# Determine response score using Binomial GLM (appropriate for proportion/count data)
# Fall back to OLS if GLM failed
primary_coef = glm_coef
primary_pval = glm_pval
print(f"\nPrimary model (Binomial GLM): coef={primary_coef:.4f}, p={primary_pval:.6f}")

significant = primary_pval < 0.05
higher = primary_coef > 0

if significant and higher:
    response = 85
    explanation = (
        f"Yes. Modern humans (Homo sapiens) have significantly higher AMTL rates than non-human primates "
        f"(Pan, Pongo, Papio) even after controlling for age, sex, and tooth class. "
        f"Binomial GLM (most appropriate model for count/proportion data): is_human coefficient={primary_coef:.4f} (log-odds), p={primary_pval:.6e}. "
        f"Mean AMTL rate: humans={human_mean:.4f}, non-humans={nonhuman_mean:.4f}. "
        f"Raw t-test p={p_val:.6e}. OLS p={is_human_pval:.4f} (less appropriate for bounded proportion data). "
        f"The Binomial GLM, the correct model for this type of data, confirms significantly higher AMTL in humans."
    )
elif significant and not higher:
    response = 15
    explanation = (
        f"No. After controlling for age, sex, and tooth class, humans actually show significantly LOWER AMTL rates. "
        f"Binomial GLM is_human coef={primary_coef:.4f}, p={primary_pval:.6e}."
    )
else:
    response = 35
    explanation = (
        f"No clear significant difference. Binomial GLM is_human coef={primary_coef:.4f}, p={primary_pval:.6e} (not significant at 0.05). "
        f"Mean AMTL: humans={human_mean:.4f}, non-humans={nonhuman_mean:.4f}."
    )

result = {"response": response, "explanation": explanation}
with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\nConclusion written to conclusion.txt")
print(json.dumps(result, indent=2))
