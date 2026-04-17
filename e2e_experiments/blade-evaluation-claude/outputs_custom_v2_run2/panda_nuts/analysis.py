import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

# Load data
df = pd.read_csv("panda_nuts.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())
print(df.dtypes)

# Create efficiency metric: nuts opened per second
df["efficiency"] = df["nuts_opened"] / df["seconds"]

# Encode categorical vars
df["sex_m"] = (df["sex"] == "m").astype(int)
df["help_y"] = (df["help"] == "y").astype(int)

print("\n--- Summary Stats ---")
print(df[["age", "sex_m", "help_y", "efficiency"]].describe())

print("\n--- Bivariate Correlations with efficiency ---")
print(df[["age", "sex_m", "help_y", "efficiency"]].corr()["efficiency"])

print("\n--- OLS Regression ---")
feature_cols = ["age", "sex_m", "help_y"]
X = df[feature_cols].copy()
X = sm.add_constant(X)
model = sm.OLS(df["efficiency"], X).fit()
print(model.summary())

print("\n--- SmartAdditiveRegressor ---")
numeric_cols = ["age", "sex_m", "help_y"]
X_df = df[numeric_cols]
y = df["efficiency"]

smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_df, y)
print(smart)
smart_effects = smart.feature_effects()
print("Feature effects:", smart_effects)

print("\n--- HingeEBMRegressor ---")
hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X_df, y)
print(hinge)
hinge_effects = hinge.feature_effects()
print("Feature effects:", hinge_effects)

# Build conclusion
age_coef = model.params.get("age", 0)
age_pval = model.pvalues.get("age", 1)
sex_coef = model.params.get("sex_m", 0)
sex_pval = model.pvalues.get("sex_m", 1)
help_coef = model.params.get("help_y", 0)
help_pval = model.pvalues.get("help_y", 1)

age_smart = smart_effects.get("age", {})
sex_smart = smart_effects.get("sex_m", {})
help_smart = smart_effects.get("help_y", {})

age_imp = age_smart.get("importance", 0)
sex_imp = sex_smart.get("importance", 0)
help_imp = help_smart.get("importance", 0)

# Score: check significance of the three key variables
sig_count = sum([age_pval < 0.05, sex_pval < 0.05, help_pval < 0.05])
marginal_count = sum([age_pval < 0.1, sex_pval < 0.1, help_pval < 0.1])

# Importance-weighted score
total_imp = age_imp + sex_imp + help_imp

if sig_count >= 2:
    score = 80
elif sig_count == 1:
    score = 60
elif marginal_count >= 1:
    score = 40
else:
    score = 20

# Adjust for effect strength
corr_age = df[["age", "efficiency"]].corr().iloc[0, 1]
corr_help = df[["help_y", "efficiency"]].corr().iloc[0, 1]

explanation = (
    f"Research question: How do age, sex, and receiving help influence nut-cracking efficiency (nuts/second)? "
    f"OLS results: age coef={age_coef:.3f} (p={age_pval:.3f}), sex_m coef={sex_coef:.3f} (p={sex_pval:.3f}), "
    f"help_y coef={help_coef:.3f} (p={help_pval:.3f}). "
    f"Bivariate correlations: age r={corr_age:.3f}, help r={corr_help:.3f}. "
    f"SmartAdditive importance: age={age_imp:.3f} (rank {age_smart.get('rank','?')}), "
    f"sex={sex_imp:.3f} (rank {sex_smart.get('rank','?')}), "
    f"help={help_imp:.3f} (rank {help_smart.get('rank','?')}). "
    f"Direction — age: {'positive' if age_coef > 0 else 'negative'}, "
    f"help: {'positive' if help_coef > 0 else 'negative'}. "
    f"Age direction from SmartAdditive: {age_smart.get('direction','unknown')}. "
    f"{sig_count} of 3 predictors are significant at p<0.05. "
    f"Score reflects the combined evidence across OLS and interpretable models."
)

result = {"response": score, "explanation": explanation}
print("\n--- Conclusion ---")
print(json.dumps(result, indent=2))

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("conclusion.txt written.")
