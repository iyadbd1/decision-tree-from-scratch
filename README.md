# Decision Trees from Scratch

An educational binary decision-tree classifier implemented with NumPy. The
project makes split search, stopping conditions, tree-node structure, and
prediction traversal explicit instead of hiding them behind a machine-learning
framework.

The repository includes a reproducible comparison with
`sklearn.tree.DecisionTreeClassifier` on the Breast Cancer Wisconsin dataset. scikit-learn is
used for the dataset, train/test split, metrics, and reference model; it is
not used by the implementation in [`src/decision_tree/`](./src/decision_tree/).

## What is implemented

- `DecisionTreeClassifier.fit(X_train, y_train)` and `.predict(X_test)`
- `fit_predict(X_train, y_train, X_test)`
- `predict_proba(X_test)`, returning a NumPy object array of per-leaf
  class-probability dictionaries
- Gini-impurity split search, with information gain available through
  `DecisionTreeNode.find_split(split_method="ig")`
- Breadth-first tree growth with `min_leaf_size`, `max_depth`, and `max_nodes`
  stopping controls
- Deterministic majority-class predictions (ties select the first class
  encountered)
- Tree inspection helpers: `depth()`, `count_nodes()`, and stored split
  criteria

This is a learning project, not a drop-in replacement for scikit-learn. It
currently supports dense numeric NumPy arrays and classification labels. It
does not implement categorical features, missing-value handling, pruning,
feature importances, or random forests.

## Project layout

```text
src/decision_tree/       Classifier and tree-node implementation
src/utils.py             Majority-vote and leaf-probability helpers
notebooks/               Reproducible Breast Cancer comparison
doc/                     Project specification and lessons learned
requirements.txt         Runtime dependencies for the notebook
pyproject.toml            Package metadata and editable-install configuration
LICENSE                  MIT License
```

## Installation

Python 3.9 or newer is required.

```bash
git clone https://github.com/iyadbd1/tree-based-models-from-scratch.git
cd tree-based-models-from-scratch
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The editable install makes the `decision_tree` package importable from scripts
and notebooks. The same runtime dependencies are declared in `pyproject.toml`
for package installation.

## Quick start

```python
import numpy as np
from decision_tree import DecisionTreeClassifier

X = np.array([[0.0], [0.2], [1.0], [1.2]])
y = np.array([0, 0, 1, 1])

model = DecisionTreeClassifier()
model.fit(X, y, max_depth=2)
print(model.predict(np.array([[0.1], [1.1]])))
```

The constructor and `fit` expose the same training controls. Pass them to
`fit` when training a model:

```python
model.fit(
    X,
    y,
    min_leaf_size=1,
    max_depth=float("inf"),
    max_nodes=float("inf"),
)
```

`min_leaf_size` is the minimum number of samples required for a node to be
considered for splitting. It is not the same as scikit-learn's
`min_samples_leaf`, which constrains both children after a split.

## Reproducing the comparison

Open [`notebooks/decision_tree_classifier.ipynb`](./notebooks/decision_tree_classifier.ipynb)
and run all cells. The notebook:

1. Loads the Breast Cancer Wisconsin dataset and creates one fixed, stratified
   80/20 train/test split.
2. Fits both models with Gini impurity and `max_depth=5`.
3. Reports test accuracy, exact prediction agreement, depth, node count, and
   confusion matrices.
4. Explains what the comparison does and does not establish.

The comparison is intentionally honest. Both models optimize the same
high-level criterion, but their tie-breaking, split-search details, stopping
semantics, tree-growth strategy, and probability-output conventions differ.
Matching accuracy alone is therefore not proof that the implementations are
structurally identical.

## Development checks

Install the optional development dependency and run any tests present in the
checkout with:

```bash
python -m pip install -e ".[dev]"
python -m pytest
```

The notebook is the primary executable demonstration and should be run from
the repository with the editable package installed.

## Design notes

The original implementation bugs and their fixes are documented in
[`doc/lessons_learned/dt_classifier_bug_report.md`](./doc/lessons_learned/dt_classifier_bug_report.md).
The broader project specification and completion checklist are in
[`doc/PROBLEM_STATEMENT.md`](./doc/PROBLEM_STATEMENT.md) and
[`doc/CHECKLIST.md`](./doc/CHECKLIST.md).

## Publishing checklist

Before changing the GitHub repository visibility to public:

1. Run the installation instructions in a clean virtual environment.
2. Run the notebook from top to bottom and confirm its outputs are reproducible.
3. Run `python -m pytest` and review any failures.
4. Confirm that `.gitignore` excludes virtual environments, caches, secrets, and
   notebook checkpoints.
5. Review the staged diff to ensure no local data, credentials, or generated
   artifacts are included.
6. Push the intended branch to GitHub, then set the repository description to
   **“An educational binary decision-tree classifier implemented from scratch with NumPy.”**
7. Suggested topics: `machine-learning`, `decision-tree`, `numpy`, `scikit-learn`,
   `python`, `from-scratch`.
