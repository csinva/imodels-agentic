import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

df = pd.read_csv("affairs.csv")

# Encode categorical variables
df["gender_num"] = (df["gender"] == "male").astype(int)
df["children_num"] = (df["children"] == "yes").astype(int)

print("=== Summary Statistics ===")
print(df[["affairs", "children_num", "age", "yearsmarried", "religiousness", "education", "occupation", "rating"]].describe())

print("\n=== Affairs by children ===")
print(df.groupby("children")["affairs"].agg(["mean", "std", "count"]))

print("\n=== Bivariate correlation: children_num vs affairs ===")
print(df[["children_num", "affairs"]].corr())

# OLS with controls
feature_cols = ["children_num", "age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender_num"]
X = df[feature_cols].copy()
X = sm.add_constant(X)
model = sm.OLS(df["affairs"], X).fit()
print("\n=== OLS Regression ===")
print(model.summary())

# Interpretable models
numeric_cols = ["children_num", "age", "yearsmarried", "religiousness", "education", "occupation", "rating", "gender_num"]
X_df = df[numeric_cols]
y = df["affairs"]

print("\n=== SmartAdditiveRegressor ===")
smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_df, y)
print(smart)
effects_smart = smart.feature_effects()
print(effects_smart)

print("\n=== HingeEBMRegressor ===")
hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X_df, y)
print(hinge)
effects_hinge = hinge.feature_effects()
print(effects_hinge)

# Gather findings
children_ols_coef = model.params["children_num"]
children_ols_pval = model.pvalues["children_num"]

children_smart = effects_smart.get("children_num", {})
children_hinge = effects_hinge.get("children_num", {})

smart_dir = children_smart.get("direction", "unknown")
smart_imp = children_smart.get("importance", 0)
smart_rank = children_smart.get("rank", 0)
hinge_dir = children_hinge.get("direction", "unknown")
hinge_imp = children_hinge.get("importance", 0)

# Score determination
# If children has negative OLS coef (decreasing affairs) and is significant, that supports "yes"
# Question: Does having children DECREASE extramarital affairs?
# Direction: children_num positive means having children -> more affairs (No to question)
# Direction: children_num negative means having children -> fewer affairs (Yes to question)

mean_with = df[df["children"] == "yes"]["affairs"].mean()
mean_without = df[df["children"] == "no"]["affairs"].mean()

print(f"\nMean affairs WITH children: {mean_with:.3f}")
print(f"Mean affairs WITHOUT children: {mean_without:.3f}")
print(f"OLS children_num coef: {children_ols_coef:.4f}, p={children_ols_pval:.4f}")
print(f"SmartAdditive: direction={smart_dir}, importance={smart_imp:.3f}, rank={smart_rank}")
print(f"HingeEBM: direction={hinge_dir}, importance={hinge_imp:.3f}")

# Bivariate: with children have MORE affairs (higher mean)
# But OLS may flip after controls
# Score based on controlled analysis
if children_ols_coef < 0 and children_ols_pval < 0.05:
    response = 65  # controlled negative effect, significant
elif children_ols_coef < 0 and children_ols_pval < 0.1:
    response = 45
elif children_ols_coef > 0 and children_ols_pval < 0.05:
    response = 15  # effect is opposite (children -> more affairs)
else:
    response = 25  # no clear controlled effect

# Adjust based on bivariate direction
bivariate_diff = mean_without - mean_with  # positive = without > with (children decrease affairs)

explanation = (
    f"Research question: Does having children decrease extramarital affairs? "
    f"Bivariate analysis shows mean affairs WITH children={mean_with:.2f} vs WITHOUT={mean_without:.2f} "
    f"(diff={mean_with - mean_without:.2f}; having children is associated with {'more' if mean_with > mean_without else 'fewer'} affairs before controls). "
    f"OLS controlled regression: children_num coefficient={children_ols_coef:.4f} (p={children_ols_pval:.4f}), "
    f"{'significant' if children_ols_pval < 0.05 else 'not significant'} after controlling for age, yearsmarried, religiousness, education, occupation, rating, gender. "
    f"SmartAdditiveRegressor: children_num direction='{smart_dir}', importance={smart_imp:.3f}, rank={smart_rank}. "
    f"HingeEBMRegressor: children_num direction='{hinge_dir}', importance={hinge_imp:.3f}. "
    f"The strongest predictors in the OLS model are marriage rating (coef={model.params['rating']:.3f}, p={model.pvalues['rating']:.4f}) "
    f"and religiousness (coef={model.params['religiousness']:.3f}, p={model.pvalues['religiousness']:.4f}). "
    f"Overall: the evidence {'supports' if response >= 50 else 'does not strongly support'} the hypothesis that having children decreases affairs."
)

result = {"response": response, "explanation": explanation}
print(f"\nFinal result: {result}")

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("conclusion.txt written.")
