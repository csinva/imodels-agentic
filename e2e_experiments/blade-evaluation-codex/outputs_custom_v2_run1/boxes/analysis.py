import json
import warnings
from pathlib import Path
import sys
import glob

# Ensure this script can find the project virtualenv packages when invoked with system python3.
for site_dir in sorted(glob.glob('/home/chansingh/imodels-evolve/.venv/lib/python*/site-packages')):
    if site_dir not in sys.path:
        sys.path.append(site_dir)

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr, chi2
import statsmodels.formula.api as smf
from sklearn.metrics import r2_score

from agentic_imodels import (
    SmartAdditiveRegressor,
    HingeEBMRegressor,
    WinsorizedSparseOLSRegressor,
)


warnings.filterwarnings('ignore')


def print_section(title: str) -> None:
    print('\n' + '=' * 80)
    print(title)
    print('=' * 80)


def logistic_fit(formula: str, data: pd.DataFrame):
    return smf.logit(formula, data=data).fit(disp=False, maxiter=200)


def safe_mean_by_group(df: pd.DataFrame, group_col: str, target_col: str) -> pd.DataFrame:
    out = df.groupby(group_col)[target_col].agg(['mean', 'count']).reset_index()
    return out.sort_values(group_col)


def extract_age_signal_hinge_ebm(model: HingeEBMRegressor) -> float:
    """Total absolute hinge-lasso signal involving age (feature x0 in this script)."""
    coefs = np.asarray(model.lasso_.coef_, dtype=float)
    n_sel = len(model.selected_)
    selected = list(model.selected_)

    if 0 not in selected:
        return 0.0

    pos = selected.index(0)
    total = abs(float(coefs[pos]))

    for k, (feat_idx_in_selected, _knot, _direction) in enumerate(model.hinge_info_):
        if feat_idx_in_selected == pos:
            total += abs(float(coefs[n_sel + k]))

    return float(total)


def main() -> None:
    base = Path('.')

    info_path = base / 'info.json'
    data_path = base / 'boxes.csv'

    info = json.loads(info_path.read_text())
    question = info['research_questions'][0]

    df = pd.read_csv(data_path)
    df['majority_choice'] = (df['y'] == 2).astype(int)
    df['gender_boy'] = (df['gender'] == 2).astype(int)

    print_section('Research Question')
    print(question)

    print_section('Data Overview')
    print(f'Shape: {df.shape}')
    print('Columns:', df.columns.tolist())
    print('\nSummary statistics:')
    print(df.describe(include='all').T)

    print('\nOutcome distribution (y):')
    print(df['y'].value_counts().sort_index())

    print('\nBinary outcome majority_choice distribution:')
    print(df['majority_choice'].value_counts().sort_index())
    print(f"Majority-choice rate: {df['majority_choice'].mean():.3f}")

    print('\nMajority-choice rate by age:')
    print(safe_mean_by_group(df, 'age', 'majority_choice'))

    print('\nMajority-choice rate by culture:')
    print(safe_mean_by_group(df, 'culture', 'majority_choice'))

    corr_cols = ['majority_choice', 'age', 'gender_boy', 'majority_first', 'culture']
    print('\nPearson correlation matrix (key variables):')
    print(df[corr_cols].corr())

    r_pb, p_pb = pointbiserialr(df['majority_choice'], df['age'])
    print('\nPoint-biserial test (age vs majority_choice):')
    print(f'r = {r_pb:.4f}, p = {p_pb:.4g}')

    print_section('Classical Statistical Tests (Logistic Regression)')

    m_biv = logistic_fit('majority_choice ~ age', df)
    print('\nBivariate model: majority_choice ~ age')
    print(m_biv.summary())

    m_ctrl = logistic_fit('majority_choice ~ age + gender_boy + majority_first + C(culture)', df)
    print('\nControlled model: majority_choice ~ age + gender_boy + majority_first + C(culture)')
    print(m_ctrl.summary())

    m_int = logistic_fit('majority_choice ~ age * C(culture) + gender_boy + majority_first', df)
    print('\nInteraction model: majority_choice ~ age * C(culture) + gender_boy + majority_first')
    print(m_int.summary())

    lr_stat = float(2 * (m_int.llf - m_ctrl.llf))
    df_diff = int(m_int.df_model - m_ctrl.df_model)
    p_lr = float(chi2.sf(lr_stat, df_diff))
    print('\nLikelihood-ratio test for adding age-by-culture interactions:')
    print(f'LR stat = {lr_stat:.4f}, df = {df_diff}, p = {p_lr:.4g}')

    base_age = float(m_int.params.get('age', np.nan))
    print('\nEstimated age log-odds slope by culture from interaction model:')
    for c in sorted(df['culture'].unique()):
        term = f'age:C(culture)[T.{c}]'
        slope = base_age + float(m_int.params.get(term, 0.0))
        print(f'culture={c}: age_slope={slope:.4f}')

    print_section('Interpretable Models (agentic_imodels)')

    X = pd.DataFrame(
        {
            'age': df['age'],
            'gender_boy': df['gender_boy'],
            'majority_first': df['majority_first'],
        }
    )

    culture_dummies = pd.get_dummies(
        df['culture'].astype('category'),
        prefix='culture',
        drop_first=True,
        dtype=int,
    )
    X = pd.concat([X, culture_dummies], axis=1)

    for col in culture_dummies.columns:
        X[f'age_x_{col}'] = df['age'] * culture_dummies[col]

    y = df['majority_choice'].to_numpy(dtype=float)

    feature_names = list(X.columns)
    print('Feature index map used by printed models:')
    for i, name in enumerate(feature_names):
        print(f'x{i} -> {name}')

    model_specs = [
        ('SmartAdditiveRegressor', SmartAdditiveRegressor()),
        ('HingeEBMRegressor', HingeEBMRegressor()),
        ('WinsorizedSparseOLSRegressor', WinsorizedSparseOLSRegressor()),
    ]

    fitted = {}
    for name, model in model_specs:
        model.fit(X, y)
        pred = model.predict(X)
        r2 = float(r2_score(y, pred))

        print(f'\n{name}: in-sample R^2 = {r2:.4f}')
        print(model)

        fitted[name] = {
            'model': model,
            'r2': r2,
        }

        if hasattr(model, 'feature_importances_'):
            fi = np.asarray(model.feature_importances_, dtype=float)
            order = np.argsort(-fi)
            print('\nTop feature importances:')
            shown = 0
            for idx in order:
                if fi[idx] <= 0:
                    continue
                print(f'  {feature_names[idx]}: {fi[idx]:.4f}')
                shown += 1
                if shown >= 10:
                    break

        if hasattr(model, 'support_'):
            print('\nSelected sparse features:')
            for idx in model.support_:
                print(f'  x{idx} -> {feature_names[int(idx)]}')

    smart = fitted['SmartAdditiveRegressor']['model']
    hinge = fitted['HingeEBMRegressor']['model']
    sparse = fitted['WinsorizedSparseOLSRegressor']['model']

    age_p_biv = float(m_biv.pvalues.get('age', np.nan))
    age_p_ctrl = float(m_ctrl.pvalues.get('age', np.nan))

    # Age evidence from interpretable models.
    smart_age_importance = float(np.asarray(smart.feature_importances_)[0]) if hasattr(smart, 'feature_importances_') else 0.0
    smart_sorted = np.argsort(-np.asarray(smart.feature_importances_)) if hasattr(smart, 'feature_importances_') else np.array([], dtype=int)
    smart_age_rank = int(np.where(smart_sorted == 0)[0][0] + 1) if smart_sorted.size else 999

    hinge_age_signal = extract_age_signal_hinge_ebm(hinge)
    sparse_age_selected = 0 in set(int(i) for i in getattr(sparse, 'support_', []))

    # Likert scoring calibrated by SKILL rubric.
    score = 50.0

    if age_p_ctrl < 0.01:
        score += 25
    elif age_p_ctrl < 0.05:
        score += 15
    elif age_p_ctrl < 0.10:
        score += 5
    else:
        score -= 15

    if age_p_biv < 0.05:
        score += 5
    else:
        score -= 5

    if p_lr < 0.05:
        score += 10
    else:
        score -= 5

    if hinge_age_signal > 1e-3:
        score += 8
    else:
        score -= 8

    if sparse_age_selected:
        score += 8
    else:
        score -= 8

    if smart_age_rank <= 3:
        score += 8
    elif smart_age_rank <= 6:
        score += 3
    else:
        score -= 3

    score = int(np.clip(round(score), 0, 100))

    explanation = (
        f"Question: {question} "
        f"Using majority-choice (y=2) as outcome, age showed no significant bivariate association "
        f"(logit p={age_p_biv:.3g}) and remained non-significant after controls for gender, majority-first order, "
        f"and culture fixed effects (p={age_p_ctrl:.3g}). Age-by-culture interactions were jointly non-significant "
        f"(LR p={p_lr:.3g}), so there is no robust evidence that age trends differ systematically across cultures in this sample. "
        f"Interpretable models were mixed: SmartAdditive assigned age modest importance (rank {smart_age_rank}, importance {smart_age_importance:.3f}), "
        f"but both hinge/Lasso-style sparse models gave null-style evidence for a direct age effect "
        f"(HingeEBM age signal={hinge_age_signal:.4f}; WinsorizedSparseOLS selected_age={sparse_age_selected}). "
        f"Stronger and more consistent predictors were demonstration order (majority_first) and, secondarily, gender/culture terms. "
        f"Overall evidence for a robust developmental increase in reliance on majority preference across cultural contexts is weak."
    )

    out = {
        'response': score,
        'explanation': explanation,
    }

    (base / 'conclusion.txt').write_text(json.dumps(out))

    print_section('Final Calibrated Conclusion')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
