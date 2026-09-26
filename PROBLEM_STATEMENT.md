# Project Problem Statement
## Tree-Based Models from Scratch: Decision Tree & Random Forest

---

## 1. Purpose of This Document

This document is your specification, not your tutorial. It tells you **what** to build, **what constraints** you must respect, and **what standard of quality** to hit — the same way a real project brief, a research advisor, or a job task would. It deliberately does **not** tell you *how* the algorithms work internally, what formulas to use, or how to structure your code internally. Figuring that out — through your own notes, math, and iteration — is the actual learning objective.

If you get stuck on the algorithm itself, that is expected and correct. Go back to your notes, a textbook, or a primary source (not an existing implementation's source code) and work it out. Do not read scikit-learn's or any other library's source code for this project until after your v1.0 is complete and tagged.

---

## 2. Learning Objectives

By the end of this project, you should be able to:

1. Explain, from memory and without notes, how a decision tree decides where to split, when to stop, and how it predicts.
2. Explain how a random forest differs from a single tree and *why* that difference reduces variance.
3. Design a small object-oriented Python library with a clean public API, the way real ML libraries are structured.
4. Validate your own implementation's correctness against a trusted reference, and articulate *why* results match, are close, or diverge.
5. Produce a GitHub repository that a stranger (e.g. a recruiter, a collaborator, or future-you in a year) could clone, understand, and evaluate in under 10 minutes.
6. Practice the full methodology of a real ML/software project: version control discipline, incremental development, testing, documentation, and reproducibility — not just "writing code that works."

---

## 3. Scope

### 3.1 In Scope (build from scratch, i.e. only using Python + NumPy)
- `DecisionTreeClassifier` — supports classification only (you may extend to regression afterward as a stretch goal, see §9).
- `RandomForestClassifier` — an ensemble built on your own `DecisionTreeClassifier`.
- Any supporting internal data structures (e.g. node representations) needed to make the above work.
- A small internal module for computing whatever splitting criteria you choose to support (you decide which — at minimum support one, and justify your choice in the README).

### 3.2 Explicitly Allowed From `sklearn` (or other established libraries)
These are things you have *already* learned and are not the target of this project. Using library implementations for these is not "cheating" — reimplementing them would just waste time you should spend on the tree algorithms.
- Dataset loading (`sklearn.datasets`) or any public dataset (CSV, Kaggle, UCI, etc.)
- Preprocessing: scaling, encoding, imputation, `train_test_split`
- Evaluation metrics: accuracy, precision, recall, F1, confusion matrix, ROC/AUC
- Cross-validation utilities, if you choose to use them for evaluation
- Baseline/reference models: `sklearn.tree.DecisionTreeClassifier` and `sklearn.ensemble.RandomForestClassifier` — **but only as a comparison benchmark, never inside your own class, never called during your model's `fit`/`predict`.**

### 3.3 Explicitly Out of Scope / Forbidden
- Importing or calling any `sklearn` model-fitting/prediction logic from inside your own classes.
- Copying implementation code (not even "for reference") from scikit-learn, XGBoost, any blog, or any AI-generated code for the core algorithm logic. Public API *shape* may resemble sklearn (see §5); internal logic must be entirely yours.
- Using `pandas`/`numpy` built-ins that would trivialize the actual algorithmic problem (e.g., don't go looking for a "build decision tree" function — there isn't one, but the spirit of the rule is: if you find yourself searching for a shortcut around the hard part, stop).
- Deep learning frameworks, `xgboost`, `lightgbm`, or any other tree-model library, anywhere in the core implementation.

If you are ever unsure whether something is in scope, write down your reasoning in the README's "Design Decisions" section (§6.4) rather than silently deciding — this is what real engineers do when a spec is ambiguous.

---

## 4. Project Phases

Work through these phases **in order**. Each phase has a "Gate" — a condition that must be true before you move to the next phase. Do not skip gates; they exist to catch the exact mistakes that make from-scratch ML projects collapse halfway through (usually: starting to code before the design is clear, or building the ensemble before the single tree is verified correct).

### Phase 0 — Repository & Environment Setup
Set up the project the way you would start any serious repository, before writing a single line of ML code.

**Tasks:**
- Create a new GitHub repository with a clear, professional name (not "ml-project" or "test-repo").
- Initialize it locally with `git init` (or clone-first if created on GitHub), add a `.gitignore` appropriate for Python (venv, `__pycache__`, `.ipynb_checkpoints`, OS files, etc.).
- Set up a virtual environment and a dependency file (`requirements.txt` or `pyproject.toml`).
- Create the initial folder skeleton (you decide the exact layout, but it must separate: library source code, tests, notebooks/experiments, and documentation into distinct directories).
- Make your first commit: skeleton + `.gitignore` + empty `README.md` + license of your choice.
- Create a `LICENSE` file (pick one — MIT is a common default for portfolio projects).

**Gate:** Repository exists on GitHub, is cloneable, has a sensible structure, and your first meaningful commit is pushed. No algorithm code yet.

---

### Phase 1 — Problem Framing & Dataset Selection
Before writing any tree logic, decide precisely what you're solving.

**Tasks:**
- Choose at least **two datasets**: one simple/small (for debugging your algorithm against known behavior) and one moderately complex (for a meaningful benchmark). At least one should be a **classic, well-documented dataset** (so a reader can sanity-check your results against known baselines).
- For each dataset, briefly document (in a notebook or markdown notes file, not the README yet): number of samples, features, classes, class balance, and any preprocessing it will need.
- Decide your train/validation/test split strategy and write down *why* you chose it.
- Do any necessary preprocessing using `sklearn` utilities per §3.2, and save/document the resulting processed data or the preprocessing steps.

**Gate:** You have clean, split, ready-to-use data sitting in memory or on disk, and you can state in one sentence per dataset what problem you're solving and why it's a reasonable test of a tree model.

---

### Phase 2 — Decision Tree: Design Before Code
This is the most important phase to *not* rush. Real engineers design the interface and data structures before implementing logic.

**Tasks:**
- On paper or in a design notes file, specify:
  - The public API of your `DecisionTreeClassifier` class: what methods it exposes, what each method's inputs/outputs are, and what a user calling it would expect (think about how you've used `sklearn` models — your class should feel familiar to a user of `sklearn`, e.g. a `.fit(X, y)` / `.predict(X)` pattern, and constructor hyperparameters set at initialization).
  - What hyperparameters your tree will support (at minimum: something controlling tree depth, something controlling minimum samples to split, and your choice of splitting criterion). Justify each.
  - What internal object represents a "node" in your tree, and what information it must hold to support both growing the tree and predicting with it later.
  - How you will handle: ties in splitting decisions, features with no valid split, leaves with mixed classes, and a max-depth or stopping condition being reached.
- Write this design down in a `docs/` markdown file *before* coding. You will revise it — that's fine — but you must start with a plan, not a blank file and trial-and-error.

**Gate:** You have a written design for the class's public interface, its node structure, and its stopping/splitting behavior, independent of any code.

---

### Phase 3 — Decision Tree: Implementation
Now implement exactly the design from Phase 2 (revising the design doc if reality forces changes — and noting *why* in the doc).

**Requirements:**
- Follow clean OOP practice: meaningful class and method names, single-responsibility methods (a method that computes a split quality metric should not also be the method that recurses and builds nodes), docstrings on every public method, and type hints on public method signatures.
- Your class must work as a **drop-in-feel** classifier: instantiate with hyperparameters, `.fit(X, y)`, `.predict(X)`, and ideally `.predict_proba(X)`. `X`/`y` should accept standard NumPy arrays (and ideally pandas DataFrames/Series).
- No global state, no notebook-only "script" logic inside the class — the class must be importable and usable from a clean Python session.
- Commit incrementally. A single giant "implemented decision tree" commit is a process failure for this project — see §6.2 for commit expectations.

**Gate:** Your tree can `fit` on the simple dataset from Phase 1 and `predict` without crashing. It does not need to be *correct* yet — that's the next phase.

---

### Phase 4 — Decision Tree: Validation & Debugging
This phase teaches you the single most important skill in this project: how do you know your from-scratch algorithm is actually right, and not just "running without errors"?

**Tasks:**
- Design at least 3 small, hand-checkable test cases where you (a human) can work out or reason about the correct tree structure or prediction *before* running your code — e.g. a tiny dataset with an obvious split. Verify your implementation matches your manual reasoning.
- Compare your tree's predictions and accuracy against `sklearn.tree.DecisionTreeClassifier` on the same data with matched hyperparameters (same max depth, same criterion, etc.). They do not need to be identical (implementation details differ) but should be in a reasonable, explainable range.
- Investigate and document at least one case where your tree's behavior surprised you, and explain what you learned from it.
- Write automated tests (see §6.3) covering: basic fit/predict correctness on a tiny known dataset, a degenerate case (e.g., all samples same class), and a shape/type error case.

**Gate:** You can confidently explain, with evidence, why you believe your tree implementation is correct — not just that it "gets similar accuracy."

---

### Phase 5 — Random Forest: Design & Implementation
Only start this once Phase 4's gate is met. A random forest built on a broken tree is a waste of time.

**Tasks:**
- Design (again, written down first) your `RandomForestClassifier`: its hyperparameters (at minimum: number of trees, and whatever randomness/sampling controls make it a *forest* and not just "many identical trees"), its public API (should mirror your tree's API), and how it aggregates individual trees' predictions into a final prediction.
- Implement it using **your own** `DecisionTreeClassifier` as the base learner — no other tree implementation allowed.
- Ensure the ensemble is reproducible (support a random seed / `random_state` parameter).

**Gate:** Your forest fits and predicts on both datasets from Phase 1, and using more trees measurably changes behavior (you should be able to show this, e.g. variance in predictions decreasing, or accuracy stabilizing, as tree count grows).

---

### Phase 6 — Evaluation & Benchmarking
Now put on your "ML practitioner" hat (this is the "revision" part — using skills you already have).

**Tasks:**
- Evaluate both your tree and your forest on the held-out test set(s) from Phase 1 using `sklearn.metrics`.
- Benchmark against `sklearn`'s reference implementations (accuracy, and at least one other metric appropriate to your dataset — e.g. F1 for imbalanced classes).
- Produce at least 2 visualizations (e.g., accuracy vs. tree depth, accuracy vs. number of trees in the forest, a confusion matrix, or feature importance if you implement it as a stretch goal) using `matplotlib`.
- Write a short, honest analysis: where does your implementation match sklearn closely, where does it diverge, and what do you believe explains the divergence (e.g., different tie-breaking, different default criteria, no pruning, etc.)?

**Gate:** You have quantitative, visualized evidence of your models' behavior and a written interpretation of it — not just printed numbers.

---

### Phase 7 — Documentation, Packaging & Final Polish
This phase is what makes the difference between "code that works" and "a project that showcases skill."

**Tasks:**
- Write the final `README.md` per the structure in §6.1.
- Clean up the repository: remove dead code, stray print statements, unused files, and experimental scratch notebooks (or move them to a clearly-labeled `experiments/` or `notebooks/` folder).
- Make sure `requirements.txt`/`pyproject.toml` is accurate and the project installs/runs in a fresh environment (test this).
- Ensure every public class and method has a docstring, and the top of each module has a short description of its purpose.
- Tag a `v1.0` release on GitHub once everything above is done.

**Gate:** A stranger could clone your repo, follow the README, install dependencies, run your demo/example, and reproduce your headline results, without asking you a single question.

---

## 5. OOP & API Design Requirements

These are requirements on the *shape* of your code, not the algorithm inside it:

- Your `DecisionTreeClassifier` and `RandomForestClassifier` must each be a proper Python class with an `__init__` that stores hyperparameters, a `fit` method, and a `predict` method, at minimum.
- Favor composition over duplicated logic — e.g., whatever node/tree-building logic your single tree uses should be reused by the forest through its use of your tree class, not copy-pasted.
- Keep the "public API" (what a user of your library calls) clearly distinguished from "internal" methods/helpers (conventionally prefixed with `_` in Python).
- No method should be responsible for more than one clear task. If you find a method doing three unrelated things, that's a signal to split it.
- Your code should be organized into at least these logical units (you choose exact file/module names): a module for the tree model, a module for the forest model, and — if you factor out shared logic (e.g., splitting-criterion calculations) — a module for that.

---

## 6. Process & Methodology Requirements

This section is arguably the actual point of the project. Algorithms can be learned from a textbook; this teaches you how ML code is built and shared professionally.

### 6.1 README Structure
Your final `README.md` must include, in this order:
1. **Project title and one-paragraph summary** — what this is and why you built it.
2. **Demo/results snapshot** — headline numbers or a key plot, right up top (this is what a recruiter sees in the first 5 seconds).
3. **What's implemented** — bullet list of features/capabilities.
4. **What's explicitly *not* from scratch** — be upfront that preprocessing/evaluation used sklearn, and why (this is a sign of maturity, not weakness).
5. **Installation & usage** — exact commands to set up and run a working example.
6. **Project structure** — brief description of the folder layout.
7. **Design decisions** — notable choices and trade-offs you made (and anything you were unsure was in/out of scope, per §3.3).
8. **Results & benchmark comparison** — your evaluation from Phase 6.
9. **Limitations & future work** — what you'd add with more time (see §9 for ideas).
10. **License**.

### 6.2 Git & Commit Discipline
- Commit early and often — aim for commits that represent one coherent unit of work each (e.g., "Add Node class and tree skeleton", not "stuff" or one 2,000-line commit at the end).
- Write commit messages in imperative mood ("Add gini impurity calculation", not "added" or "adding").
- Use a `.gitignore` from the start; never commit virtual environments, `__pycache__`, or large data files.
- Consider using feature branches for each major phase (e.g., `feature/decision-tree`, `feature/random-forest`) and merging via pull request into `main`, even solo — this is standard practice and worth building the habit now.
- Tag your `v1.0` release when Phase 7 is complete.

### 6.3 Testing
- Use a real test framework (`pytest` is standard) rather than ad hoc print statements.
- Tests live in a dedicated `tests/` directory, separate from library code.
- At minimum, cover: correctness on a tiny hand-verifiable dataset, at least one edge/degenerate case, and basic API contract checks (e.g., predicting before fitting should raise a clear error, not crash obscurely).
- Your test suite should be runnable with a single command, and that command must be documented in the README.

### 6.4 Design Decisions Log
Keep a running log (in `docs/` or in the README's "Design decisions" section) of any point where you had to make a judgment call — about scope, about an ambiguous requirement in this document, or about a trade-off in your implementation. This is what distinguishes a project that shows *engineering judgment* from one that just "has working code."

---

## 7. Definition of Done

The project is complete when **all** of the following are true:

- [ ] Both classes are implemented from scratch per §3 and §5, with no forbidden dependencies.
- [ ] All Phase gates (§4) have been met, in order.
- [ ] The test suite passes and covers the cases in §6.3.
- [ ] The README meets the structure in §6.1 and is accurate (a fresh clone actually works as described).
- [ ] Benchmark comparison against sklearn is present, visualized, and honestly interpreted.
- [ ] The repository is clean: no dead code, no stray files, consistent style.
- [ ] A `v1.0` tag/release exists on GitHub.

---

## 8. How to Use This Document

Work phase by phase. Do not read ahead into "how do I implement X" tutorials — if you're stuck on the algorithm itself, that's the productive kind of stuck. Struggle with it, sketch it on paper, revisit your ML fundamentals notes, and only after you've formed your own hypothesis should you check it against a trusted textbook explanation (not example code). The goal is that by the end, you could explain and reimplement this project from memory in a different language.

---

## 9. Stretch Goals (Optional, Only After v1.0 Is Tagged)

Attempt these only after the Definition of Done (§7) is fully met — do not let them delay v1.0:

- Extend `DecisionTreeClassifier` to support regression (`DecisionTreeRegressor`), and build a `RandomForestRegressor` on top of it.
- Implement feature importance calculation.
- Implement a second splitting criterion and let hyperparameters switch between them; compare their effect on your benchmark datasets.
- Add support for handling missing values natively (rather than only via sklearn preprocessing).
- Add basic pruning (e.g., cost-complexity pruning) and evaluate its effect on overfitting.
- Package the library properly (installable via `pip install -e .`) with a `pyproject.toml`.
