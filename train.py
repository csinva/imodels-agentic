"""
Interpretable classifier autoresearch script.
Defines a scikit-learn compatible interpretable classifier and evaluates it
on TabArena classification datasets using AUC.

Usage: uv run train.py
"""

import time
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.utils.validation import check_is_fitted

from prepare import TIME_BUDGET, DATASET_NAMES, evaluate_auc

# ---------------------------------------------------------------------------
# Interpretable Classifier (edit this — everything below is fair game)
# ---------------------------------------------------------------------------

# Hyperparameters
MAX_DEPTH = 4          # maximum depth of decision tree (controls complexity)
MIN_SAMPLES_LEAF = 10  # minimum samples per leaf (controls overfitting)
CRITERION = "gini"     # split criterion: "gini" or "entropy"


class InterpretableClassifier(BaseEstimator, ClassifierMixin):
    """
    Interpretable scikit-learn compatible classifier.

    This is the baseline: a shallow decision tree.
    The agent may modify this class freely — architecture, algorithm, features, etc.
    Must implement: fit(X, y), predict(X), predict_proba(X).
    """

    def __init__(self, max_depth=MAX_DEPTH, min_samples_leaf=MIN_SAMPLES_LEAF,
                 criterion=CRITERION):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion

    def fit(self, X, y):
        self.tree_ = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            criterion=self.criterion,
            random_state=42,
        )
        self.tree_.fit(X, y)
        self.classes_ = self.tree_.classes_
        return self

    def predict(self, X):
        check_is_fitted(self, "tree_")
        return self.tree_.predict(X)

    def predict_proba(self, X):
        check_is_fitted(self, "tree_")
        return self.tree_.predict_proba(X)

    def __str__(self):
        """Human-readable representation of the classifier."""
        check_is_fitted(self, "tree_")
        n_leaves = self.tree_.get_n_leaves()
        n_nodes = self.tree_.tree_.node_count
        return (
            f"InterpretableClassifier(max_depth={self.max_depth}, "
            f"min_samples_leaf={self.min_samples_leaf}, "
            f"criterion={self.criterion!r})\n"
            f"  nodes={n_nodes}, leaves={n_leaves}"
        )


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

t_start = time.time()

print(f"Datasets: {DATASET_NAMES}")
print(f"Time budget: {TIME_BUDGET}s")
print()

# Evaluate the classifier across all TabArena datasets.
# model_factory is called once per dataset to get a fresh model.
def model_factory():
    return InterpretableClassifier()


print("Evaluating classifier on TabArena classification datasets...")
mean_auc = evaluate_auc(model_factory)

t_end = time.time()
total_seconds = t_end - t_start

# Print one trained example for interpretability inspection
print()
print("Example classifier (trained on first available dataset):")
from prepare import get_all_datasets
for name, X_train, X_test, y_train, y_test in get_all_datasets():
    model = model_factory()
    model.fit(X_train, y_train)
    print(f"  Dataset: {name}")
    print(f"  {model}")
    print("  Tree structure (first 3 levels):")
    tree_text = export_text(model.tree_, max_depth=3)
    for line in tree_text.split("\n")[:20]:
        print(f"    {line}")
    break

# Final summary
print()
print("---")
print(f"mean_auc:         {mean_auc:.6f}")
print(f"total_seconds:    {total_seconds:.1f}")
print(f"max_depth:        {MAX_DEPTH}")
print(f"min_samples_leaf: {MIN_SAMPLES_LEAF}")
print(f"criterion:        {CRITERION}")
print(f"num_datasets:     {len(DATASET_NAMES)}")
