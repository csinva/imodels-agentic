import json
import pandas as pd
import numpy as np
import statsmodels.api as sm
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

# Load data
df = pd.read_csv("affairs.csv")
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# Encode categoricals
df["gender_num"] = (df["gender"] == "male").astype(int)
df["children_num"] = (df["children"] == "yes").astype(int)

print("\nAffairs by children:")
print(df.groupby("children")["affairs"].describe())
print("\nMean affairs by children:", df.groupby("children")["affairs"].mean())

# Bivariate correlation
from scipy import stats
no_kids = df[df["children"] == "no"]["affairs"]
yes_kids = df[df["children"] == "yes"]["affairs"]
t, p = stats.ttest_ind(no_kids, yes_kids)
print(f"\nt-test: t={t:.3f}, p={p:.4f}")
print(f"No children mean: {no_kids.mean():.3f}, With children mean: {yes_kids.mean():.3f}")

# OLS with controls
feature_cols = ["children_num", "gender_num", "age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
X = df[feature_cols]
X = sm.add_constant(X)
model = sm.OLS(df["affairs"], X).fit()
print("\nOLS Summary:")
print(model.summary())

# SmartAdditiveRegressor
numeric_cols = ["children_num", "gender_num", "age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
X_df = df[numeric_cols].copy()
X_df.columns = ["children", "gender", "age", "yearsmarried", "religiousness", "education", "occupation", "rating"]
y = df["affairs"]

smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_df, y)
print("\nSmartAdditiveRegressor:")
print(smart)
smart_effects = smart.feature_effects()
print("Feature effects:", smart_effects)

# HingeEBMRegressor
hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X_df, y)
print("\nHingeEBMRegressor:")
print(hinge)
hinge_effects = hinge.feature_effects()
print("Feature effects:", hinge_effects)

# Build conclusion
children_ols_coef = model.params["children_num"]
children_ols_p = model.pvalues["children_num"]
children_smart = smart_effects.get("children", {})
children_hinge = hinge_effects.get("children", {})

# The question asks: does having children DECREASE affairs?
# "decrease" = negative effect of children on affairs
# Score: if children has negative significant effect -> high score (Yes)
# children_num=1 means "yes children", so negative coef = having children -> fewer affairs

bivariate_diff = yes_kids.mean() - no_kids.mean()
direction_bivariate = "negative" if bivariate_diff < 0 else "positive"

explanation = (
    f"Research question: Does having children decrease extramarital affairs? "
    f"Bivariate analysis: mean affairs for those without children = {no_kids.mean():.3f}, "
    f"with children = {yes_kids.mean():.3f} (diff={bivariate_diff:.3f}, {direction_bivariate} effect of children). "
    f"t-test: t={t:.3f}, p={p:.4f}. "
    f"OLS with controls: children coefficient = {children_ols_coef:.3f} (p={children_ols_p:.4f}). "
    f"SmartAdditive: children direction={children_smart.get('direction','unknown')}, importance={children_smart.get('importance',0):.3f}, rank={children_smart.get('rank','?')}. "
    f"HingeEBM: children direction={children_hinge.get('direction','unknown')}, importance={children_hinge.get('importance',0):.3f}, rank={children_hinge.get('rank','?')}. "
)

# Determine score
# If OLS coef for children is negative and p < 0.05 -> strong yes (75-100)
# If negative but p > 0.05 -> weak (15-40)
# Note: the question is about DECREASE, so negative coef = yes
if children_ols_coef < 0 and children_ols_p < 0.05:
    score = 70
    explanation += "The effect is statistically significant and negative in OLS: having children is associated with fewer affairs. However, bivariate analysis shows those WITH children have HIGHER mean affairs, suggesting confounding (e.g., longer marriages). After controlling for years married and other factors, the children effect may be attenuated or reversed. Rating (marital happiness) and religiousness are stronger predictors."
elif children_ols_coef < 0 and children_ols_p < 0.10:
    score = 45
    explanation += "Marginal negative effect in OLS, not statistically significant at 0.05 level."
elif children_ols_coef > 0 and children_ols_p < 0.05:
    score = 15
    explanation += "Counterintuitively, having children is associated with MORE affairs in controlled analysis."
else:
    score = 25
    explanation += "Weak or inconsistent effect of children on affairs across models."

print(f"\nFinal score: {score}")
print(f"Explanation: {explanation}")

conclusion = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(conclusion, f)

print("\nWrote conclusion.txt")
