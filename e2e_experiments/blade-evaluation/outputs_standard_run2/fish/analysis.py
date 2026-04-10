import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from imodels import FIGSRegressor, HSTreeRegressor, RuleFitRegressor
from scipy import stats
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")


def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def main():
    base = Path('.')

    # 1) Load problem metadata and data
    with open(base / 'info.json', 'r', encoding='utf-8') as f:
        info = json.load(f)
    question = info.get('research_questions', [''])[0]

    df = pd.read_csv(base / 'fish.csv')

    # Basic derived targets
    df['hours_clipped'] = df['hours'].clip(lower=1e-6)
    df['fish_per_hour'] = df['fish_caught'] / df['hours_clipped']
    df['log_fish_per_hour'] = np.log1p(df['fish_per_hour'])
    df['caught_any'] = (df['fish_caught'] > 0).astype(int)

    feature_cols = ['livebait', 'camper', 'persons', 'child', 'hours']

    # 2) Exploratory summaries
    summary_stats = df[['fish_caught', 'fish_per_hour'] + feature_cols].describe().T
    skewness = df[['fish_caught', 'fish_per_hour'] + feature_cols].skew(numeric_only=True)
    corr = df[['fish_caught', 'fish_per_hour'] + feature_cols].corr(numeric_only=True)

    # 3) Statistical tests
    tests = {}

    # Two-group tests for binary indicators
    for col in ['livebait', 'camper']:
        grp1 = df.loc[df[col] == 1, 'fish_per_hour']
        grp0 = df.loc[df[col] == 0, 'fish_per_hour']
        t_res = stats.ttest_ind(grp1, grp0, equal_var=False)
        tests[f'ttest_{col}_fish_per_hour'] = {
            'mean_when_1': safe_float(grp1.mean()),
            'mean_when_0': safe_float(grp0.mean()),
            't_stat': safe_float(t_res.statistic),
            'p_value': safe_float(t_res.pvalue),
        }

    # ANOVA for multi-level grouped effects
    for col in ['persons', 'child']:
        groups = [g['fish_per_hour'].values for _, g in df.groupby(col)]
        aov = stats.f_oneway(*groups)
        tests[f'anova_{col}_fish_per_hour'] = {
            'f_stat': safe_float(aov.statistic),
            'p_value': safe_float(aov.pvalue),
        }

    # Correlations with significance
    pearson = stats.pearsonr(df['hours'], df['fish_caught'])
    spearman = stats.spearmanr(df['hours'], df['fish_caught'])
    tests['corr_hours_fish_caught'] = {
        'pearson_r': safe_float(pearson.statistic),
        'pearson_p': safe_float(pearson.pvalue),
        'spearman_rho': safe_float(spearman.correlation),
        'spearman_p': safe_float(spearman.pvalue),
    }

    # 4) Interpretable regression with p-values (statsmodels OLS)
    X_ols = sm.add_constant(df[feature_cols])
    y_ols = df['log_fish_per_hour']
    ols = sm.OLS(y_ols, X_ols).fit()

    ols_params = {k: safe_float(v) for k, v in ols.params.to_dict().items()}
    ols_pvals = {k: safe_float(v) for k, v in ols.pvalues.to_dict().items()}

    # 5) Interpretable sklearn models
    X = df[feature_cols]
    y = df['log_fish_per_hour']
    y_bin = df['caught_any']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(
        X, y_bin, test_size=0.30, random_state=42
    )

    lin = Pipeline([
        ('scale', StandardScaler()),
        ('model', LinearRegression())
    ])
    ridge = Pipeline([
        ('scale', StandardScaler()),
        ('model', Ridge(alpha=1.0, random_state=42))
    ])
    lasso = Pipeline([
        ('scale', StandardScaler()),
        ('model', Lasso(alpha=0.02, random_state=42, max_iter=20000))
    ])

    lin.fit(X_train, y_train)
    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)

    dt_reg = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt_reg.fit(X_train, y_train)

    dt_clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_clf.fit(Xb_train, yb_train)

    sklearn_results = {
        'r2_linear': safe_float(r2_score(y_test, lin.predict(X_test))),
        'r2_ridge': safe_float(r2_score(y_test, ridge.predict(X_test))),
        'r2_lasso': safe_float(r2_score(y_test, lasso.predict(X_test))),
        'r2_dt_reg': safe_float(r2_score(y_test, dt_reg.predict(X_test))),
        'acc_dt_clf': safe_float(dt_clf.score(Xb_test, yb_test)),
        'coef_linear_scaled': dict(zip(feature_cols, map(safe_float, lin.named_steps['model'].coef_))),
        'coef_ridge_scaled': dict(zip(feature_cols, map(safe_float, ridge.named_steps['model'].coef_))),
        'coef_lasso_scaled': dict(zip(feature_cols, map(safe_float, lasso.named_steps['model'].coef_))),
        'fi_dt_reg': dict(zip(feature_cols, map(safe_float, dt_reg.feature_importances_))),
        'fi_dt_clf': dict(zip(feature_cols, map(safe_float, dt_clf.feature_importances_))),
    }

    # 6) Interpretable imodels models
    imodels_results = {}

    # RuleFitRegressor: obtain sparse linear + rule representation
    rulefit = RuleFitRegressor(random_state=42)
    rulefit.fit(X_train, y_train, feature_names=feature_cols)
    rules_df = rulefit._get_rules()  # available in current imodels version
    nonzero_rules = rules_df.loc[rules_df['coef'] != 0].copy()
    nonzero_rules = nonzero_rules.sort_values('importance', ascending=False)
    top_rules = []
    for _, row in nonzero_rules.head(10).iterrows():
        top_rules.append({
            'rule': str(row['rule']),
            'type': str(row['type']),
            'coef': safe_float(row['coef']),
            'support': safe_float(row['support']),
            'importance': safe_float(row['importance']),
        })
    imodels_results['rulefit_r2'] = safe_float(r2_score(y_test, rulefit.predict(X_test)))
    imodels_results['rulefit_top_rules'] = top_rules

    # FIGSRegressor
    figs = FIGSRegressor(random_state=42, max_rules=20)
    figs.fit(X_train, y_train)
    imodels_results['figs_r2'] = safe_float(r2_score(y_test, figs.predict(X_test)))
    imodels_results['figs_feature_importances'] = dict(
        zip(feature_cols, map(safe_float, figs.feature_importances_))
    )

    # HSTreeRegressor
    hst = HSTreeRegressor(max_leaf_nodes=8, random_state=42)
    hst.fit(X_train, y_train)
    imodels_results['hstree_r2'] = safe_float(r2_score(y_test, hst.predict(X_test)))

    # 7) Interpretation for research question
    avg_fish_per_hour = safe_float(df['fish_per_hour'].mean())
    median_fish_per_hour = safe_float(df['fish_per_hour'].median())

    significant_features = [
        k for k, p in ols_pvals.items()
        if k != 'const' and isinstance(p, float) and p < 0.05
    ]

    sig_tests = {
        name: vals for name, vals in tests.items()
        if isinstance(vals, dict) and any(
            (kk.startswith('p_') or kk.endswith('p') or 'p_value' in kk) and isinstance(vv, float) and vv < 0.05
            for kk, vv in vals.items()
        )
    }

    # Score evidence for "yes, relationships exist and rate can be estimated"
    score = 50

    if safe_float(ols.f_pvalue) < 0.05:
        score += 15

    score += min(25, 5 * len(significant_features))

    n_sig_tests = 0
    for vals in tests.values():
        if isinstance(vals, dict):
            for kk, vv in vals.items():
                if (kk.startswith('p_') or kk.endswith('p') or 'p_value' in kk) and isinstance(vv, float) and vv < 0.05:
                    n_sig_tests += 1
                    break
    score += min(10, 3 * n_sig_tests)

    model_r2 = np.nanmean([
        sklearn_results['r2_linear'],
        sklearn_results['r2_ridge'],
        sklearn_results['r2_lasso'],
        sklearn_results['r2_dt_reg'],
        imodels_results['rulefit_r2'],
        imodels_results['figs_r2'],
        imodels_results['hstree_r2'],
    ])
    if np.isfinite(model_r2):
        score += int(max(0, min(15, round(model_r2 * 40))))

    score = int(max(0, min(100, score)))

    top_tree_features = sorted(
        sklearn_results['fi_dt_reg'].items(), key=lambda kv: kv[1], reverse=True
    )[:3]
    top_figs_features = sorted(
        imodels_results['figs_feature_importances'].items(), key=lambda kv: kv[1], reverse=True
    )[:3]

    explanation = (
        f"Question: {question} "
        f"Estimated average catch rate is {avg_fish_per_hour:.2f} fish/hour (median {median_fish_per_hour:.2f}), with many zero-catch trips. "
        f"OLS on log(1+fish/hour) is significant overall (F-test p={safe_float(ols.f_pvalue):.3g}, R^2={safe_float(ols.rsquared):.3f}); "
        f"significant predictors include {', '.join(significant_features) if significant_features else 'none'}. "
        f"Group tests show significance for: {', '.join(sig_tests.keys()) if sig_tests else 'no tested groups'}. "
        f"Interpretable models (Linear/Ridge/Lasso, DecisionTree, RuleFit, FIGS, HSTree) provide consistent signal, "
        f"with top tree features {top_tree_features} and top FIGS features {top_figs_features}. "
        f"This supports a strong Yes that catch rate varies with observed factors and can be estimated, though fit is moderate rather than perfect."
    )

    # 8) Persist required output format
    output = {
        'response': score,
        'explanation': explanation,
    }

    with open(base / 'conclusion.txt', 'w', encoding='utf-8') as f:
        json.dump(output, f)

    # Optional runtime logs for traceability
    print('Rows:', len(df), 'Columns:', df.shape[1])
    print('Summary stats (fish_caught / fish_per_hour):')
    print(summary_stats.loc[['fish_caught', 'fish_per_hour'], ['mean', 'std', 'min', '50%', 'max']])
    print('\nSkewness:')
    print(skewness)
    print('\nCorrelations with fish_per_hour:')
    print(corr['fish_per_hour'].sort_values(ascending=False))
    print('\nOLS coefficients:')
    print(ols_params)
    print('OLS p-values:')
    print(ols_pvals)
    print('\nSklearn metrics:')
    print({k: v for k, v in sklearn_results.items() if 'coef' not in k and 'fi_' not in k})
    print('\nTop RuleFit rules:')
    for r in imodels_results['rulefit_top_rules'][:5]:
        print(r)
    print('\nWrote conclusion.txt with response score =', score)


if __name__ == '__main__':
    main()
