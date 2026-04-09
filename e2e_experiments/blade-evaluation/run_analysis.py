"""Generic analysis script that runs on any Blade dataset.

Reads info.json for the research question and dataset metadata,
loads the CSV, performs appropriate statistical analysis, and
writes conclusion.txt with a Likert score and explanation.

Usage:
    python run_analysis.py outputs/hurricane
    python run_analysis.py --all          # run on all datasets missing conclusions
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")

DATASETS = [
    "affairs", "amtl", "boxes", "caschools", "crofoot", "fertility",
    "fish", "hurricane", "mortgage", "panda_nuts", "reading", "soccer",
    "teachingratings",
]


def analyze_dataset(dataset_dir: str) -> dict:
    """Analyze a single dataset and return conclusion dict."""
    # Load metadata
    with open(os.path.join(dataset_dir, "info.json")) as f:
        info = json.load(f)
    question = info["research_questions"][0]
    fields = info["data_desc"]["fields"]
    field_names = [f["column"] for f in fields]

    # Find CSV file
    csv_files = [f for f in os.listdir(dataset_dir) if f.endswith(".csv")]
    if not csv_files:
        return {"response": 0, "explanation": "No CSV data file found."}
    df = pd.read_csv(os.path.join(dataset_dir, csv_files[0]))

    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Build field descriptions for context
    field_descs = {}
    for f in fields:
        col = f["column"]
        desc = f.get("properties", {}).get("description", "")
        field_descs[col] = desc

    # Strategy: try multiple approaches and pick the most informative
    results = []

    # 1. Pairwise correlations among numeric variables
    if len(numeric_cols) >= 2:
        corr_results = []
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i+1:]:
                valid = df[[c1, c2]].dropna()
                if len(valid) < 10:
                    continue
                r, p = stats.pearsonr(valid[c1], valid[c2])
                corr_results.append((c1, c2, r, p))

        # Find the most relevant correlation based on field descriptions matching question
        question_lower = question.lower()
        best_corr = None
        best_score = -1
        for c1, c2, r, p in corr_results:
            # Score relevance by how many words from question appear in field descriptions
            desc1 = field_descs.get(c1, "").lower()
            desc2 = field_descs.get(c2, "").lower()
            combined = desc1 + " " + desc2 + " " + c1.lower() + " " + c2.lower()
            score = sum(1 for word in question_lower.split()
                       if len(word) > 3 and word in combined)
            if score > best_score:
                best_score = score
                best_corr = (c1, c2, r, p)

        if best_corr:
            c1, c2, r, p = best_corr
            results.append({
                "test": "Pearson correlation",
                "vars": f"{c1} vs {c2}",
                "statistic": r,
                "p_value": p,
                "desc": f"Pearson r={r:.3f}, p={p:.3e} between {c1} ({field_descs.get(c1, '')[:50]}) and {c2} ({field_descs.get(c2, '')[:50]})"
            })

    # 2. Try OLS regression with the most relevant DV and IVs
    if len(numeric_cols) >= 2:
        try:
            import statsmodels.api as sm

            # Pick DV: the numeric column whose description best matches the question
            question_lower = question.lower()
            dv_scores = []
            for col in numeric_cols:
                desc = field_descs.get(col, "").lower() + " " + col.lower()
                score = sum(1 for word in question_lower.split()
                           if len(word) > 3 and word in desc)
                dv_scores.append((col, score))
            dv_scores.sort(key=lambda x: -x[1])
            dv = dv_scores[0][0]

            # IVs: next most relevant numeric columns
            ivs = [col for col, _ in dv_scores[1:4] if col != dv]
            if ivs:
                data = df[[dv] + ivs].dropna()
                if len(data) >= 20:
                    X = sm.add_constant(data[ivs])
                    y = data[dv]
                    model = sm.OLS(y, X).fit()

                    # Check if any IV is significant
                    sig_vars = [v for v in ivs if model.pvalues.get(v, 1) < 0.05]
                    overall_p = model.f_pvalue if hasattr(model, 'f_pvalue') else 1.0

                    results.append({
                        "test": "OLS regression",
                        "vars": f"DV={dv}, IVs={ivs}",
                        "statistic": model.rsquared,
                        "p_value": overall_p,
                        "sig_vars": sig_vars,
                        "desc": f"OLS: R²={model.rsquared:.3f}, F p={overall_p:.3e}, significant predictors: {sig_vars}"
                    })
        except ImportError:
            pass
        except Exception:
            pass

    # 3. For binary/categorical outcomes, try chi-square or t-test
    for cat_col in cat_cols[:3]:
        unique_vals = df[cat_col].dropna().unique()
        if 2 <= len(unique_vals) <= 5:
            for num_col in numeric_cols[:3]:
                groups = [df.loc[df[cat_col] == v, num_col].dropna() for v in unique_vals]
                groups = [g for g in groups if len(g) >= 5]
                if len(groups) >= 2:
                    if len(groups) == 2:
                        t, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                        results.append({
                            "test": "Welch's t-test",
                            "vars": f"{num_col} by {cat_col}",
                            "statistic": t,
                            "p_value": p,
                            "desc": f"Welch's t-test: {num_col} by {cat_col}, t={t:.3f}, p={p:.3e}"
                        })
                    else:
                        f, p = stats.f_oneway(*groups)
                        results.append({
                            "test": "ANOVA",
                            "vars": f"{num_col} by {cat_col}",
                            "statistic": f,
                            "p_value": p,
                            "desc": f"ANOVA: {num_col} by {cat_col}, F={f:.3f}, p={p:.3e}"
                        })

    # Synthesize conclusion
    if not results:
        return {
            "response": 50,
            "explanation": f"Unable to identify clear variables to test. Research question: {question}"
        }

    # Pick the most relevant result (lowest p-value among relevant tests)
    # Prefer OLS if available
    ols = [r for r in results if r["test"] == "OLS regression"]
    if ols:
        best = ols[0]
    else:
        best = min(results, key=lambda r: r["p_value"])

    p = best["p_value"]
    significant = p < 0.05

    if significant:
        # Scale response by effect size / strength
        if "statistic" in best:
            strength = min(abs(best["statistic"]), 1.0)
            response = int(50 + strength * 50)  # 50-100 range
        else:
            response = 75
    else:
        response = int(max(0, 25 * (1 - p / 0.05)))  # 0-25 range

    explanation_parts = [
        f"Research question: {question[:200]}",
        f"Primary test: {best['desc']}",
        f"Conclusion: {'Significant' if significant else 'Not significant'} at alpha=0.05.",
    ]
    if len(results) > 1:
        other_tests = [r["desc"] for r in results if r is not best][:2]
        explanation_parts.append(f"Additional tests: {'; '.join(other_tests)}")

    return {
        "response": response,
        "explanation": " | ".join(explanation_parts)
    }


def run_dataset(dataset_name: str):
    """Run analysis for a single dataset."""
    dataset_dir = os.path.join(OUTPUT_DIR, dataset_name)
    if not os.path.isdir(dataset_dir):
        print(f"  SKIP: {dataset_name} (directory not found)")
        return False

    print(f"  Analyzing: {dataset_name}...", end=" ", flush=True)
    try:
        conclusion = analyze_dataset(dataset_dir)
        with open(os.path.join(dataset_dir, "conclusion.txt"), "w") as f:
            json.dump(conclusion, f)
        print(f"response={conclusion['response']}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run generic analysis on Blade datasets")
    parser.add_argument("dataset_dir", nargs="?", help="Path to a specific dataset directory")
    parser.add_argument("--all", action="store_true", help="Run on all datasets missing conclusions")
    parser.add_argument("--force", action="store_true", help="Overwrite existing conclusions")
    args = parser.parse_args()

    if args.dataset_dir:
        dataset_name = os.path.basename(args.dataset_dir.rstrip("/"))
        run_dataset(dataset_name)
    elif args.all:
        success = 0
        for ds in DATASETS:
            conclusion_path = os.path.join(OUTPUT_DIR, ds, "conclusion.txt")
            if os.path.exists(conclusion_path) and not args.force:
                print(f"  SKIP: {ds} (conclusion exists, use --force to overwrite)")
                continue
            if run_dataset(ds):
                success += 1
        print(f"\nDone: {success} datasets analyzed")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
