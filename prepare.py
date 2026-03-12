"""
Data preparation for interpretable classifier autoresearch.
Downloads and caches classification datasets from TabArena.

Usage:
    python prepare.py              # download and cache all datasets
    python prepare.py --list       # list available datasets

Data is cached in ~/.cache/imodels-evolve/.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300        # experiment time budget in seconds (5 minutes)
RANDOM_STATE = 42
TEST_SIZE = 0.2

# TabArena classification dataset names (OpenML-backed)
# Drawn from the TabArena paper benchmark suite
DATASET_NAMES = [
    "adult",
    "blood-transfusion-service-center",
    "breast-cancer",
    "california",
    "credit-g",
    "diabetes",
    "higgs",
    "jannis",
    "kr-vs-kp",
    "mfeat-factors",
    "numerai28.6",
    "phoneme",
    "sylvine",
    "volkert",
]

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "imodels-evolve")

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _get_dataset_path(name):
    return os.path.join(CACHE_DIR, f"{name}.parquet")


def _download_dataset(name):
    """Download a single dataset from OpenML and save to cache."""
    import openml
    os.makedirs(CACHE_DIR, exist_ok=True)
    openml.config.cache_directory = os.path.join(CACHE_DIR, "openml")
    # Search by name
    datasets_list = openml.datasets.list_datasets(output_format="dataframe")
    matches = datasets_list[datasets_list["name"] == name]
    if len(matches) == 0:
        raise ValueError(f"Dataset '{name}' not found on OpenML")
    # Pick most recent active dataset with this name
    matches = matches[matches["status"] == "active"]
    dataset_id = int(matches.sort_values("did").iloc[-1]["did"])
    dataset = openml.datasets.get_dataset(dataset_id, download_data=True)
    X, y, categorical_indicator, attribute_names = dataset.get_data(
        target=dataset.default_target_attribute
    )
    df = pd.DataFrame(X, columns=attribute_names)
    df["__target__"] = y
    df.to_parquet(_get_dataset_path(name), index=False)
    print(f"  Downloaded '{name}' (id={dataset_id}, rows={len(df)}, cols={len(attribute_names)})")
    return df


def _load_cached_dataset(name):
    path = _get_dataset_path(name)
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def load_dataset(name):
    """
    Load a dataset by name. Returns (X_train, X_test, y_train, y_test).

    - Categorical features are ordinal-encoded (unknown categories -> -1).
    - Target is label-encoded to 0/1/... integers.
    - Train/test split is fixed (TEST_SIZE, RANDOM_STATE).
    """
    df = _load_cached_dataset(name)
    if df is None:
        df = _download_dataset(name)

    # Split target
    y_raw = df["__target__"].values
    X_raw = df.drop(columns=["__target__"])

    # Encode target
    le = LabelEncoder()
    y = le.fit_transform(y_raw.astype(str))

    # Identify and encode categorical columns
    cat_cols = X_raw.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X_raw.columns if c not in cat_cols]

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Numeric: fill NaN with median
    X_train = X_train_raw.copy()
    X_test = X_test_raw.copy()
    for col in num_cols:
        median = X_train[col].median()
        X_train[col] = pd.to_numeric(X_train[col], errors="coerce").fillna(median)
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce").fillna(median)

    # Categorical: ordinal encode
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.float32)
        X_train[cat_cols] = enc.fit_transform(X_train[cat_cols].astype(str))
        X_test[cat_cols] = enc.transform(X_test[cat_cols].astype(str))

    X_train = X_train.astype(np.float32).values
    X_test = X_test.astype(np.float32).values

    return X_train, X_test, y_train, y_test


def get_all_datasets():
    """
    Iterate over all datasets. Yields (name, X_train, X_test, y_train, y_test).
    Skips datasets that fail to load with a warning.
    """
    for name in DATASET_NAMES:
        try:
            X_train, X_test, y_train, y_test = load_dataset(name)
            yield name, X_train, X_test, y_train, y_test
        except Exception as e:
            print(f"  WARNING: skipping '{name}': {e}")


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

def evaluate_auc(model_factory):
    """
    Evaluate a model across all TabArena classification datasets.

    Args:
        model_factory: callable with no arguments that returns a fresh
                       sklearn-compatible classifier (fit/predict_proba).

    Returns:
        mean_auc (float): mean one-vs-rest AUC across all datasets.
    """
    aucs = []
    for name, X_train, X_test, y_train, y_test in get_all_datasets():
        model = model_factory()
        model.fit(X_train, y_train)
        n_classes = len(np.unique(y_train))
        if n_classes == 2:
            proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
        else:
            proba = model.predict_proba(X_test)
            auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        aucs.append(auc)
        print(f"  {name:40s} auc={auc:.4f}")
    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    return mean_auc


# ---------------------------------------------------------------------------
# Main: download and cache all datasets
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare TabArena datasets for autoresearch")
    parser.add_argument("--list", action="store_true", help="List available datasets and exit")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Subset of datasets to download (default: all)")
    args = parser.parse_args()

    if args.list:
        print("TabArena classification datasets:")
        for name in DATASET_NAMES:
            cached = "cached" if os.path.exists(_get_dataset_path(name)) else "not cached"
            print(f"  {name:40s} [{cached}]")
        sys.exit(0)

    os.makedirs(CACHE_DIR, exist_ok=True)
    names = args.datasets if args.datasets else DATASET_NAMES

    print(f"Cache directory: {CACHE_DIR}")
    print(f"Downloading {len(names)} datasets...")
    for name in names:
        path = _get_dataset_path(name)
        if os.path.exists(path):
            print(f"  '{name}' already cached")
        else:
            try:
                _download_dataset(name)
            except Exception as e:
                print(f"  ERROR: failed to download '{name}': {e}")

    print()
    print("Done! Verifying datasets...")
    ok = 0
    for name in names:
        try:
            X_train, X_test, y_train, y_test = load_dataset(name)
            print(f"  {name:40s} train={X_train.shape}, test={X_test.shape}, classes={len(np.unique(y_train))}")
            ok += 1
        except Exception as e:
            print(f"  ERROR: {name}: {e}")
    print(f"\n{ok}/{len(names)} datasets ready.")
