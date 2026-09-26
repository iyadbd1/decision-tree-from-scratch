# Project Checklist
## Tree-Based Models from Scratch

Companion to `PROBLEM_STATEMENT.md`. Check items off in order — each phase should be fully checked before starting the next.

---

### Phase 0 — Repository & Environment Setup
- [x] GitHub repository created with a clear, professional name
- [x] Repo initialized locally / cloned
- [x] `.gitignore` added (Python-appropriate)
- [x] Virtual environment set up
- [x] `requirements.txt` or `pyproject.toml` created
- [x] Folder skeleton created (source / tests / notebooks / docs separated)
- [ ] `LICENSE` file added
- [x] Empty/placeholder `README.md` committed
- [x] First commit pushed to GitHub

### Phase 1 — Problem Framing & Dataset Selection
- [ ] Simple/small debug dataset chosen
- [ ] Moderately complex benchmark dataset chosen
- [ ] Dataset notes documented (size, features, classes, balance)
- [ ] Train/val/test split strategy decided and justified
- [ ] Preprocessing done via sklearn utilities (scope-compliant)
- [ ] One-sentence problem framing written per dataset

### Phase 2 — Decision Tree: Design Before Code
- [ ] Public API sketched (constructor params, `fit`, `predict`, etc.)
- [ ] Hyperparameters chosen and justified
- [ ] Node data structure defined
- [ ] Stopping conditions defined
- [ ] Tie-breaking / edge-case handling defined
- [ ] Design written in `docs/` **before** any implementation code

### Phase 3 — Decision Tree: Implementation
- [ ] Class implemented per the Phase 2 design
- [ ] Follows clean OOP practice (naming, single-responsibility, docstrings, type hints)
- [ ] Works with NumPy arrays as input
- [ ] `fit` / `predict` (/ `predict_proba`) implemented
- [ ] No notebook-only logic — class is cleanly importable
- [ ] Committed incrementally (not one giant commit)
- [ ] Runs end-to-end on the simple dataset without crashing

### Phase 4 — Decision Tree: Validation & Debugging
- [ ] 3+ hand-checkable test cases designed and verified
- [ ] Compared against `sklearn.tree.DecisionTreeClassifier` with matched hyperparameters
- [ ] At least one surprising behavior investigated and documented
- [ ] Automated tests written (basic correctness, degenerate case, error case)
- [ ] Can explain *why* the implementation is believed correct (not just "accuracy looks fine")

### Phase 5 — Random Forest: Design & Implementation
- [ ] Forest design written down first (hyperparameters, randomness mechanism, aggregation method)
- [ ] Implemented using your own `DecisionTreeClassifier` as base learner
- [ ] `random_state` / reproducibility supported
- [ ] Fits and predicts on both datasets
- [ ] Effect of tree count on behavior demonstrated

### Phase 6 — Evaluation & Benchmarking
- [ ] Evaluated on held-out test set(s) using `sklearn.metrics`
- [ ] Benchmarked against sklearn's reference tree and forest
- [ ] 2+ visualizations produced
- [ ] Written analysis of matches/divergences vs. sklearn

### Phase 7 — Documentation, Packaging & Final Polish
- [ ] Final `README.md` written per required structure
- [ ] Repository cleaned (no dead code, stray files, scratch notebooks moved/labeled)
- [ ] Fresh-environment install/run tested and confirmed working
- [ ] All public classes/methods have docstrings
- [ ] `v1.0` tag/release created on GitHub

---

### Final Definition of Done
- [ ] No forbidden dependencies used in core implementation
- [ ] All phase gates met in order
- [ ] Test suite passes, runnable via one documented command
- [ ] README accurate against a fresh clone
- [ ] Benchmark comparison present, visualized, honestly interpreted
- [ ] Repository clean and consistent
- [ ] `v1.0` tagged

---

### Stretch Goals (only after v1.0 is tagged)
- [ ] `DecisionTreeRegressor` + `RandomForestRegressor`
- [ ] Feature importance
- [ ] Second splitting criterion, compared empirically
- [ ] Native missing-value handling
- [ ] Cost-complexity pruning
- [ ] Packaged as `pip install -e .`
