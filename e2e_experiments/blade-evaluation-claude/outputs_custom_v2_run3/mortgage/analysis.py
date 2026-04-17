import pandas as pd
import numpy as np
import statsmodels.api as sm
import json
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

df = pd.read_csv("mortgage.csv")
print("Shape:", df.shape)
print(df.describe())

# DV: accept (or deny), IV: female (gender)
print("\nAccept rate by gender:")
print(df.groupby("female")["accept"].mean())

print("\nCorrelation of female with accept:", df["female"].corr(df["accept"]))

# OLS with controls
control_cols = ["black", "housing_expense_ratio", "self_employed", "married",
                "mortgage_credit", "consumer_credit", "bad_history", "PI_ratio",
                "loan_to_value", "denied_PMI"]
feature_cols = ["female"] + control_cols
df_model = df[feature_cols + ["accept"]].dropna()

X = sm.add_constant(df_model[feature_cols])
model = sm.OLS(df_model["accept"], X).fit()
print(model.summary())

female_coef = model.params["female"]
female_pval = model.pvalues["female"]

# SmartAdditiveRegressor
numeric_cols = feature_cols
X_df = df_model[numeric_cols]
y = df_model["accept"]

smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_df, y)
print("\nSmartAdditiveRegressor:")
print(smart)
smart_effects = smart.feature_effects()
print(smart_effects)

# HingeEBMRegressor
hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X_df, y)
print("\nHingeEBMRegressor:")
print(hinge)
hinge_effects = hinge.feature_effects()
print(hinge_effects)

# Summarize
female_smart = smart_effects.get("female", {})
female_hinge = hinge_effects.get("female", {})

smart_dir = female_smart.get("direction", "unknown")
smart_imp = female_smart.get("importance", 0)
smart_rank = female_smart.get("rank", None)
hinge_dir = female_hinge.get("direction", "unknown")
hinge_imp = female_hinge.get("importance", 0)
hinge_rank = female_hinge.get("rank", None)

explanation = (
    f"OLS: female coef={female_coef:.4f}, p={female_pval:.4f}. "
    f"SmartAdditive: direction={smart_dir}, importance={smart_imp:.4f}, rank={smart_rank}. "
    f"HingeEBM: direction={hinge_dir}, importance={hinge_imp:.4f}, rank={hinge_rank}. "
)

# Determine response score
if female_pval < 0.05:
    if abs(female_coef) > 0.03 and smart_imp > 0.03:
        score = 65
    else:
        score = 50
else:
    if smart_imp < 0.01 and hinge_imp < 0.01:
        score = 10
    else:
        score = 25

# Build top predictors context
top_features = sorted(smart_effects.items(), key=lambda x: -x[1].get("importance", 0))[:3]
top_str = ", ".join([f"{f}(imp={v.get('importance',0):.3f})" for f, v in top_features])
explanation += f"Top predictors by SmartAdditive: {top_str}. "

if female_pval >= 0.05:
    explanation += "Gender (female) does NOT have a statistically significant effect on mortgage approval after controlling for creditworthiness and other variables. "
else:
    direction_word = "positive" if female_coef > 0 else "negative"
    explanation += f"Gender has a {direction_word} significant effect on approval. "

explanation += "Overall, credit history, debt ratios, and financial variables drive approval decisions; gender plays a minimal role."

result = {"response": score, "explanation": explanation}
print("\nResult:", json.dumps(result, indent=2))

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("conclusion.txt written.")
