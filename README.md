# Tree-Based Models from Scratch

A small educational project for understanding decision trees and random-forest-style ensemble thinking without relying on a library implementation for the core algorithm.

## Overview

This repository contains a custom `DecisionTreeClassifier` built from first principles, along with supporting notes and a scikit-learn comparison workflow. The emphasis is on learning the mechanics of tree construction, splitting criteria, stopping rules, and prediction logic rather than production optimization.

This project is intended to be readable, portable, and easy to evaluate in a fresh environment. It is best suited for study, portfolio use, and demonstration of core ML ideas.

## What is included

- `DecisionTreeClassifier` built with a sklearn-style API: `fit`, `predict`, `predict_proba`
- node-based tree structure with recursive splitting logic
- simple impurity-based split selection
- a comparison notebook using `scikit-learn` as a reference baseline
- project documentation covering the problem statement, checklist, and lessons learned

## Why this project exists

This project is a learning-focused implementation of a tree-based classifier: the goal is to understand the underlying mechanics of model construction, not to compete with optimized production libraries.

By the end of the workflow, you should be able to explain:

- how a decision tree chooses a split
- how impurity and stopping rules affect the final model
- why a tree is easy to interpret but sensitive to variance
- how a reference implementation differs from a custom one

## Quick validation

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
pytest -q
```

## Repository structure

```text
.
├── data/                          # dataset storage or experiment artifacts
├── doc/                          # project documentation
│   ├── CHECKLIST.md
│   ├── DECISION_TREE_COMPARISON_README.md
│   ├── Learned_Lessons.md
│   └── PROBLEM_STATEMENT.md
├── notebooks/
│   └── decision_tree_comparison.ipynb
├── src/
│   ├── decision_tree/
│   │   ├── __init__.py
│   │   ├── classifier.py
│   │   └── node.py
│   └── utils.py
├── tests/
│   └── test_smoke.py
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
├── requirements.txt
└── .venv/
```

## Installation

Clone the repository and set up a virtual environment:

```bash
git clone <your-repo-url>
cd <repo-folder>
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If you want to install the package in editable mode as well:

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from decision_tree import DecisionTreeClassifier

X_train = np.array([
    [0.0, 0.0],
    [1.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0],
])
y_train = np.array([0, 0, 1, 1])

X_test = np.array([[0.2, 0.8], [0.9, 0.2]])

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
print(model.predict(X_test))
print(model.predict_proba(X_test))
```

## Run the smoke test

```bash
pytest
```

The smoke test checks that the package imports correctly and that a tiny trained tree can fit and predict on a simple dataset.

## Notes on the implementation

The tree follows a classic recursive partitioning pattern:

- evaluate candidate splits across features and thresholds
- rank them by impurity reduction
- stop when depth, purity, or minimum-size constraints are reached
- use majority voting at leaf nodes for class prediction

This is intentionally educational rather than production-optimized.

## Comparison and documentation

The notebook in `notebooks/decision_tree_comparison.ipynb` compares the custom model with scikit-learn's reference implementation on a common classification workflow. The documentation in `doc/` explains the project goals, phase plan, and lessons learned.

## Project status

This repository is structured as a clean, shareable learning project and is ready for public GitHub use. It is suitable for demonstrating both the decision-tree idea and the discipline of building a small ML project from scratch.

## Public release checklist

Before pushing the repo publicly, confirm:

- the repository is pushed to GitHub
- the license file is included and visible
- the README works from a fresh clone
- the smoke test passes in CI or local setup
- no local-only secrets, usernames, or machine-specific config are included

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.


