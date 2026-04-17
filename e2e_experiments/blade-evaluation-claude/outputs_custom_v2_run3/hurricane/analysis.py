import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from interp_models import SmartAdditiveRegressor, HingeEBMRegressor

# Load data
df = pd.read_csv("hurricane.csv")
print("Shape:", df.shape)
print(df.describe())

# Research question: Do more feminine hurricane names lead to fewer precautionary measures (more deaths)?
# DV: alldeaths (proxy for precautionary behavior - more deaths = fewer precautions)
# IV: masfem (femininity index, higher = more feminine)

# Step 1: Explore
print("\n--- Bivariate correlations with alldeaths ---")
numeric_cols = ["masfem", "gender_mf", "masfem_mturk", "category", "min", "wind", "ndam", "ndam15", "elapsedyrs", "year"]
for col in numeric_cols:
    r, p = stats.pearsonr(df[col].dropna(), df.loc[df[col].notna(), "alldeaths"])
    print(f"  {col}: r={r:.3f}, p={p:.3f}")

# Step 2: OLS with controls - log-transform deaths (skewed)
df["log_deaths"] = np.log1p(df["alldeaths"])
df["log_ndam"] = np.log1p(df["ndam"])

feature_columns = ["masfem", "category", "min", "wind", "log_ndam", "elapsedyrs"]
df_clean = df[feature_columns + ["log_deaths"]].dropna()
df_clean = df_clean.replace([np.inf, -np.inf], np.nan).dropna()
X = df_clean[feature_columns].copy()
X = sm.add_constant(X)
model = sm.OLS(df_clean["log_deaths"], X).fit()
print("\n--- OLS with controls (log_deaths) ---")
print(model.summary())

# Also try binary gender
feature_columns2 = ["gender_mf", "category", "min", "wind", "log_ndam", "elapsedyrs"]
df_clean2 = df[feature_columns2 + ["log_deaths"]].dropna().replace([np.inf, -np.inf], np.nan).dropna()
X2 = df_clean2[feature_columns2].copy()
X2 = sm.add_constant(X2)
model2 = sm.OLS(df_clean2["log_deaths"], X2).fit()
print("\n--- OLS with binary gender ---")
print(model2.summary()[["gender_mf"]] if hasattr(model2.summary(), '__getitem__') else model2.summary())

# Step 3: Interpretable models
numeric_columns = ["masfem", "gender_mf", "masfem_mturk", "category", "min", "wind", "log_ndam", "elapsedyrs", "year"]
df_interp = df[numeric_columns + ["log_deaths"]].replace([np.inf, -np.inf], np.nan).dropna()
X_interp = df_interp[numeric_columns].copy()
y = df_interp["log_deaths"]

print("\n--- SmartAdditiveRegressor ---")
smart = SmartAdditiveRegressor(n_rounds=200)
smart.fit(X_interp, y)
print(smart)
effects_smart = smart.feature_effects()
print("Feature effects:", effects_smart)

print("\n--- HingeEBMRegressor ---")
hinge = HingeEBMRegressor(n_knots=3)
hinge.fit(X_interp, y)
print(hinge)
effects_hinge = hinge.feature_effects()
print("Feature effects:", effects_hinge)

# Step 4: Synthesize
masfem_ols_coef = model.params.get("masfem", None)
masfem_ols_p = model.pvalues.get("masfem", None)
masfem_smart = effects_smart.get("masfem", {})
masfem_hinge = effects_hinge.get("masfem", {})

print(f"\n--- Summary for masfem ---")
print(f"OLS coef={masfem_ols_coef:.4f}, p={masfem_ols_p:.4f}")
print(f"SmartAdditive: {masfem_smart}")
print(f"HingeEBM: {masfem_hinge}")

# Determine score
# The question asks about feminine names -> fewer precautions -> more deaths
# We look at whether masfem predicts more deaths
masfem_significant = masfem_ols_p is not None and masfem_ols_p < 0.05
masfem_positive = masfem_ols_coef is not None and masfem_ols_coef > 0
masfem_importance = masfem_smart.get("importance", 0)

# Bivariate correlation
r_biv, p_biv = stats.pearsonr(df["masfem"], df["log_deaths"])
print(f"\nBivariate: r={r_biv:.3f}, p={p_biv:.3f}")

# Score based on evidence
if masfem_significant and masfem_positive and masfem_importance > 0.1:
    score = 75
elif masfem_significant and masfem_positive:
    score = 60
elif not masfem_significant and masfem_positive and p_biv < 0.1:
    score = 35
elif not masfem_significant:
    score = 20
else:
    score = 15

# Adjust based on importance rank
if masfem_importance > 0.2:
    score = min(score + 10, 100)
elif masfem_importance < 0.05:
    score = max(score - 10, 0)

explanation = (
    f"The research question asks whether more feminine hurricane names lead to fewer precautionary measures, "
    f"proxied here by higher death tolls. "
    f"Bivariate correlation: masfem vs log(deaths+1): r={r_biv:.3f}, p={p_biv:.3f}. "
    f"OLS with controls (category, min pressure, wind, log damage, elapsed years): "
    f"masfem coef={masfem_ols_coef:.4f}, p={masfem_ols_p:.4f}. "
    f"{'Effect is statistically significant' if masfem_significant else 'Effect is not statistically significant'} at p<0.05 with controls. "
    f"SmartAdditiveRegressor ranks masfem with importance={masfem_importance:.3f}, direction={masfem_smart.get('direction','unknown')}. "
    f"HingeEBM: importance={masfem_hinge.get('importance', 0):.3f}, direction={masfem_hinge.get('direction','unknown')}. "
    f"The dominant predictors are storm severity measures (category, wind, pressure, damage). "
    f"The evidence for the feminine-name effect on precautionary behavior is "
    f"{'moderate to weak' if not masfem_significant else 'present but modest'}, "
    f"consistent with the broader replication literature showing the original Jung et al. (2014) result is fragile."
)

result = {"response": score, "explanation": explanation}
print("\n--- CONCLUSION ---")
print(json.dumps(result, indent=2))

with open("conclusion.txt", "w") as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
