import json
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("hurricane.csv")
print("Shape:", df.shape)
print(df[["masfem", "gender_mf", "alldeaths", "category", "ndam", "min", "wind"]].describe())

# Correlation: masfem vs alldeaths
r_masfem, p_masfem = stats.pearsonr(df["masfem"], df["alldeaths"])
print(f"\nPearson r(masfem, alldeaths) = {r_masfem:.4f}, p = {p_masfem:.4f}")

r_log, p_log = stats.pearsonr(df["masfem"], np.log1p(df["alldeaths"]))
print(f"Pearson r(masfem, log(alldeaths)) = {r_log:.4f}, p = {p_log:.4f}")

# t-test: female vs male hurricanes
female = df[df["gender_mf"] == 1]["alldeaths"]
male = df[df["gender_mf"] == 0]["alldeaths"]
t, p_t = stats.ttest_ind(female, male)
print(f"\nt-test female vs male deaths: t={t:.4f}, p={p_t:.4f}")
print(f"Female mean deaths: {female.mean():.2f}, Male mean deaths: {male.mean():.2f}")

# OLS controlling for storm severity
cols = ["masfem", "min", "wind", "ndam", "category", "alldeaths"]
df_clean = df[cols].dropna()
X = df_clean[["masfem", "min", "wind", "ndam", "category"]].copy()
X = sm.add_constant(X)
y = np.log1p(df_clean["alldeaths"])
model = sm.OLS(y, X).fit()
print("\nOLS (log deaths ~ masfem + controls):")
print(model.summary().tables[1])

# OLS without controls
X2 = sm.add_constant(df_clean[["masfem"]])
model2 = sm.OLS(y, X2).fit()
print("\nOLS (log deaths ~ masfem only):")
print(model2.summary().tables[1])

# Ridge regression
scaler = StandardScaler()
features = ["masfem", "min", "wind", "ndam", "category"]
X_scaled = scaler.fit_transform(df_clean[features])
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, np.log1p(df_clean["alldeaths"]))
print("\nRidge coefficients (standardized):")
for f, c in zip(features, ridge.coef_):
    print(f"  {f}: {c:.4f}")

# Spearman correlation
rho, p_sp = stats.spearmanr(df["masfem"], df["alldeaths"])
print(f"\nSpearman rho(masfem, alldeaths) = {rho:.4f}, p = {p_sp:.4f}")

# Summary
masfem_coef = model.params["masfem"]
masfem_pval = model.pvalues["masfem"]
masfem_coef_raw = model2.params["masfem"]
masfem_pval_raw = model2.pvalues["masfem"]

print(f"\nControlled model: masfem coef={masfem_coef:.4f}, p={masfem_pval:.4f}")
print(f"Raw model: masfem coef={masfem_coef_raw:.4f}, p={masfem_pval_raw:.4f}")

# Decision: weak positive correlation, but not significant after controlling for severity
# The raw correlation is weakly positive but not significant
# After controlling for storm severity, masfem effect becomes even weaker
significant_raw = p_masfem < 0.05
significant_controlled = masfem_pval < 0.05

if significant_raw and masfem_coef_raw > 0:
    score = 65
    explanation = (
        f"There is a statistically significant positive correlation between hurricane name femininity "
        f"(masfem) and deaths in the raw analysis (r={r_masfem:.3f}, p={p_masfem:.4f}), "
        f"but after controlling for storm severity (wind, pressure, damage, category), "
        f"the effect is {'significant' if significant_controlled else 'not significant'} "
        f"(coef={masfem_coef:.4f}, p={masfem_pval:.4f}). "
        f"Female-named hurricanes averaged {female.mean():.1f} deaths vs {male.mean():.1f} for male-named. "
        f"The evidence partially supports the hypothesis but is sensitive to controls."
    )
elif not significant_raw:
    score = 30
    explanation = (
        f"The correlation between hurricane name femininity (masfem) and deaths is not statistically "
        f"significant in the raw analysis (r={r_masfem:.3f}, p={p_masfem:.4f}). "
        f"After controlling for storm severity, masfem coef={masfem_coef:.4f}, p={masfem_pval:.4f}. "
        f"Female-named hurricanes averaged {female.mean():.1f} deaths vs {male.mean():.1f} for male-named. "
        f"The data does not strongly support the hypothesis that more feminine hurricane names lead to more deaths."
    )
else:
    score = 40
    explanation = (
        f"Mixed evidence: raw correlation r={r_masfem:.3f} p={p_masfem:.4f}, "
        f"controlled coef={masfem_coef:.4f} p={masfem_pval:.4f}."
    )

result = {"response": score, "explanation": explanation}
with open("conclusion.txt", "w") as f:
    json.dump(result, f)
print(f"\nConclusion written: score={score}")
