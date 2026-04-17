import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import json
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('fertility.csv')
print("Shape:", df.shape)
print(df.head())
print(df.describe())

# Parse dates
for col in ['DateTesting', 'StartDateofLastPeriod', 'StartDateofPeriodBeforeLast']:
    df[col] = pd.to_datetime(df[col], format='%m/%d/%y', errors='coerce')

# Compute days since last period (proxy for cycle phase)
df['DaysSinceLast'] = (df['DateTesting'] - df['StartDateofLastPeriod']).dt.days

# Compute cycle length from the two period dates
df['ComputedCycleLength'] = (df['StartDateofLastPeriod'] - df['StartDateofPeriodBeforeLast']).dt.days

# Use reported cycle length where available, else computed
df['CycleLength'] = df['ReportedCycleLength'].fillna(df['ComputedCycleLength'])

# Estimated day in cycle (1 = first day of period)
df['CycleDay'] = df['DaysSinceLast']

# Ovulation typically around day 14 for 28-day cycle; scale by actual cycle length
df['FertilityWindow'] = df['CycleDay'] / df['CycleLength']  # 0=period start, ~0.5=ovulation

# High fertility: days 10-17 (follicular/ovulation phase) in a normalized sense
# Use absolute day estimate for ovulation
df['DaysToOvulation'] = df['CycleLength'] - 14 - (df['CycleLength'] - df['CycleDay'])
df['EstOvulationDay'] = df['CycleLength'] - 14
df['DaysFromOvulation'] = (df['CycleDay'] - df['EstOvulationDay']).abs()

# High fertility = within 5 days of estimated ovulation
df['HighFertility'] = (df['DaysFromOvulation'] <= 5).astype(int)

# Composite religiosity score
df['Religiosity'] = df[['Rel1', 'Rel2', 'Rel3']].mean(axis=1)

# Drop rows with missing key variables
df_clean = df.dropna(subset=['CycleDay', 'CycleLength', 'Religiosity'])
print(f"\nClean N={len(df_clean)}, HighFertility N={df_clean['HighFertility'].sum()}")

# --- Statistical Tests ---

high = df_clean[df_clean['HighFertility'] == 1]['Religiosity']
low = df_clean[df_clean['HighFertility'] == 0]['Religiosity']
print(f"\nHigh fertility religiosity: mean={high.mean():.3f}, n={len(high)}")
print(f"Low fertility religiosity:  mean={low.mean():.3f}, n={len(low)}")

t_stat, p_val = stats.ttest_ind(high, low)
print(f"t-test: t={t_stat:.3f}, p={p_val:.4f}")

# Correlation between cycle day / fertility window and religiosity
r_day, p_day = stats.pearsonr(df_clean['CycleDay'].dropna(), df_clean.loc[df_clean['CycleDay'].notna(), 'Religiosity'])
print(f"\nCorrelation CycleDay vs Religiosity: r={r_day:.3f}, p={p_day:.4f}")

r_fw, p_fw = stats.pearsonr(df_clean['FertilityWindow'].dropna(), df_clean.loc[df_clean['FertilityWindow'].notna(), 'Religiosity'])
print(f"Correlation FertilityWindow vs Religiosity: r={r_fw:.3f}, p={p_fw:.4f}")

# OLS regression controlling for relationship status
X = df_clean[['HighFertility', 'Relationship']].dropna()
y = df_clean.loc[X.index, 'Religiosity']
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
print("\nOLS summary:")
print(model.summary())

hf_coef = model.params['HighFertility']
hf_pval = model.pvalues['HighFertility']
print(f"\nHighFertility coef={hf_coef:.3f}, p={hf_pval:.4f}")

# Decision: is there a significant effect?
alpha = 0.05
significant = (p_val < alpha) or (hf_pval < alpha)

# Effect size (Cohen's d)
pooled_std = np.sqrt((high.std()**2 + low.std()**2) / 2)
cohens_d = (high.mean() - low.mean()) / pooled_std if pooled_std > 0 else 0
print(f"\nCohen's d = {cohens_d:.3f}")

# Construct response score: base on p-values and effect direction
# Low p-value and effect in either direction → higher score
if significant and abs(cohens_d) > 0.1:
    response = 65
elif p_val < 0.1 or hf_pval < 0.1:
    response = 40
else:
    response = 20

explanation = (
    f"Research question: Does fertility (hormonal fluctuations) affect women's religiosity? "
    f"N={len(df_clean)} women after cleaning. "
    f"High-fertility group (within 5 days of estimated ovulation, n={len(high)}) had mean religiosity={high.mean():.2f} "
    f"vs low-fertility group (n={len(low)}) mean={low.mean():.2f}. "
    f"Independent t-test: t={t_stat:.3f}, p={p_val:.4f}. "
    f"OLS regression (controlling for relationship): HighFertility coef={hf_coef:.3f}, p={hf_pval:.4f}. "
    f"Cohen's d={cohens_d:.3f}. "
    f"Correlation of cycle day with religiosity: r={r_day:.3f}, p={p_day:.4f}. "
    f"{'The effect is statistically significant, suggesting fertility fluctuations do affect religiosity.' if significant else 'No statistically significant effect found; results do not support a relationship between hormonal fluctuations and religiosity.'}"
)

result = {"response": response, "explanation": explanation}
print("\n" + json.dumps(result, indent=2))

with open('conclusion.txt', 'w') as f:
    json.dump(result, f)

print("\nconclusion.txt written.")
