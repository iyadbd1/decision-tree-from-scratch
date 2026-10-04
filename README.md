# Tree-Based Models from Scratch

A hands-on, from-scratch machine learning project focused on decision trees and tree-based model fundamentals. The repository implements a custom decision tree classifier, explores the design choices behind tree construction, and compares the implementation against scikit-learn on a small benchmark workflow.

## Overview

This project was built as a learning exercise in interpretable machine learning and software engineering. The goal is not to replace production libraries, but to understand the underlying mechanics of how a decision tree grows, splits, and predicts.

The repository currently includes:

- a custom `DecisionTreeClassifier` built from first principles
- a node-based tree implementation with recursive splitting logic
- probability estimation at leaves
- a comparison notebook using scikit-learn as a reference baseline
- project documentation describing the problem statement, design choices, and learning notes

## Project goals

- Implement decision tree logic without relying on a library model implementation
- Use a familiar sklearn-style API (`fit`, `predict`, `predict_proba`)
- Compare custom behavior against an established reference model
- Keep the project understandable, portable, and easy to run from a fresh clone

## Repository structure

```text
.
├── data/                          # dataset storage (empty or project-specific data files)
├── doc/                          # project notes and specifications
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
├── pyproject.toml
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Create and activate a virtual environment, then install the project dependencies:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

If you want to work in editable package mode from the project root:

```bash
pip install -e .
```

## Quick start

```python
import numpy as np
from decision_tree import DecisionTreeClassifier

X_train = np.array([
    [1.0, 2.0],
    [2.0, 1.5],
    [3.0, 3.0],
    [4.0, 2.5],
])
y_train = np.array([0, 0, 1, 1])

X_test = np.array([[2.5, 2.0]])

model = DecisionTreeClassifier(min_leaf_size=1, max_depth=5)
model.fit(X_train, y_train)
preds = model.predict(X_test)
probs = model.predict_proba(X_test)

print(preds)
print(probs)
```

## Model details

The custom tree follows a classic recursive partitioning approach:

- feature thresholds are evaluated greedily
- splits are chosen by minimizing impurity
- the implementation uses Gini impurity and entropy-based information gain logic internally
- leaf nodes return class predictions by majority vote
- stopping criteria are based on node size, depth, and class purity

This makes the implementation a compact, educational baseline rather than a production-optimized tree engine.

## Comparison workflow

The notebook in `notebooks/decision_tree_comparison.ipynb` compares the custom implementation with scikit-learn's `DecisionTreeClassifier` using the same input data and matching hyperparameters where possible. The comparison is intentionally framed as an educational benchmark: it highlights both where the custom implementation behaves well and where a library implementation is more robust and optimized.

## Documentation

Relevant project documents are located in `doc/`:

- `doc/PROBLEM_STATEMENT.md` — specification and project requirements
- `doc/CHECKLIST.md` — delivery checklist and phase tracking
- `doc/DECISION_TREE_COMPARISON_README.md` — documentation for the comparison notebook
- `doc/Learned_Lessons.md` — retrospective notes and implementation insights

## Notes and limitations

This project is designed for learning and demonstration. The custom tree implementation is intentionally simpler than mature production libraries and is best suited for understanding the core mechanics of tree construction rather than for large-scale or highly optimized ML workloads.


