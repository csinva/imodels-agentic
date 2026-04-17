import json
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv("teachingratings.csv")
print("Shape:", df.shape)
print(df.describe())

# Encode categorical variables
df_enc = df.copy()
for col in ["minority", "gender", "credits", "division", "native", "tenure"]:
    df_enc[col] = (df_enc[col] == df_enc[col].unique()[0]).astype(int)

# 1. Simple correlation between beauty and eval
r, p_corr = stats.pearsonr(df["beauty"], df["eval"])
print(f"\nPearson r(beauty, eval) = {r:.4f}, p = {p_corr:.4e}")

# 2. OLS regression: eval ~ beauty (simple)
X_simple = sm.add_constant(df["beauty"])
ols_simple = sm.OLS(df["eval"], X_simple).fit()
print("\n--- Simple OLS: eval ~ beauty ---")
print(ols_simple.summary())

# 3. Multiple OLS with controls
feature_cols = ["beauty", "age", "minority", "gender", "credits", "division", "native", "tenure", "students"]
X_full = sm.add_constant(df_enc[feature_cols])
ols_full = sm.OLS(df_enc["eval"], X_full).fit()
print("\n--- Multiple OLS: eval ~ beauty + controls ---")
print(ols_full.summary())

beauty_coef = ols_full.params["beauty"]
beauty_pval = ols_full.pvalues["beauty"]
print(f"\nBeauty coefficient (controlled): {beauty_coef:.4f}, p = {beauty_pval:.4e}")

# 4. Decision tree for feature importance
X_dt = df_enc[feature_cols].fillna(0)
y = df_enc["eval"]
dt = DecisionTreeRegressor(max_depth=4, random_state=42)
dt.fit(X_dt, y)
importances = pd.Series(dt.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nDecision Tree Feature Importances:")
print(importances)

# 5. Ridge regression coefficients
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_dt)
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)
ridge_coefs = pd.Series(ridge.coef_, index=feature_cols).sort_values(key=abs, ascending=False)
print("\nRidge Coefficients (standardized):")
print(ridge_coefs)

# Determine response score
# Beauty is significant (p < 0.05) with positive coefficient -> Yes (high score)
significant = beauty_pval < 0.05
positive_effect = beauty_coef > 0

if significant and positive_effect:
    response = 80
    explanation = (
        f"Beauty has a statistically significant positive impact on teaching evaluations. "
        f"Simple correlation r={r:.3f} (p={p_corr:.4f}). "
        f"In multiple regression controlling for age, gender, minority, credits, division, native, tenure, and class size, "
        f"beauty coefficient={beauty_coef:.4f} (p={beauty_pval:.4e}). "
        f"Ridge standardized coefficient for beauty={ridge_coefs['beauty']:.4f}, ranked #{list(ridge_coefs.index).index('beauty')+1} in importance. "
        f"The evidence consistently shows that more attractive instructors receive higher teaching evaluations."
    )
elif significant and not positive_effect:
    response = 20
    explanation = (
        f"Beauty has a statistically significant but negative impact on teaching evaluations. "
        f"Coefficient={beauty_coef:.4f} (p={beauty_pval:.4e})."
    )
else:
    response = 20
    explanation = (
        f"Beauty does not have a statistically significant impact on teaching evaluations. "
        f"Coefficient={beauty_coef:.4f} (p={beauty_pval:.4e})."
    )

print(f"\nResponse: {response}")
print(f"Explanation: {explanation}")

result = {"response": response, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nWrote conclusion.txt")
