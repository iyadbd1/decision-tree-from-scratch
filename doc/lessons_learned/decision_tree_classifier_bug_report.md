# Decision Tree Classifier: Comprehensive Bug Report

## 1. Global Sort Instead of Per-Feature Sort

* **Why it happened:** `np.sort(self.X_train)` sorted the entire 2D array globally inside the feature loop, losing row-column relationships. The result was stored in a local variable but never used; thresholds were computed from unsorted `self.X_train`.
* **What it broke:** Thresholds became midpoints between random adjacent rows instead of sorted feature values. The tree could not find meaningful decision boundaries (e.g., separating y=1.0 from y=2.0), producing meaningless splits.
* **Fix:** Use `np.argsort(self.X_train[:, j])` to sort only column j, keep `y_train` aligned via the same index array, and skip duplicate consecutive values when computing thresholds.

## 2. Double `split()` Call and Depth Not Propagated

* **Why it happened:** `current.split()` was called twice in the fit loop—first discarding its return value, then reassigning children. Additionally, `current_depth += 1` modified only the loop variable without setting child node depths.
* **What it broke:** Children were overwritten with fresh nodes (depth reset to 0), wasting 2× computation and corrupting the `depth()` method's recursive calculation. Tree structure became inconsistent.
* **Fix:** Call `split()` once, explicitly set `lc.d = rc.d = current_depth + 1`, and enqueue children with their correct depth values.

## 3. `predict_proba` List Index Assignment Crash

* **Why it happened:** `y_pred_probs` was initialized as an empty list `[]`, then assigned by index `[i]` inside the loop. Python lists do not support out-of-range index assignment.
* **What it broke:** Immediate `IndexError` on the first test sample, making probability prediction completely non-functional.
* **Fix:** Accumulate results in a list using `.append()`, then convert to `np.array(proba_list)` after the loop. This avoids needing to know `n_classes` at allocation time.

## 4. Information Gain Branch Using Unsorted Partitions

* **Why it happened:** The IG branch computed entropy on `y_left`/`y_right` derived from the same unsorted mask logic as the Gini branch. Even after fixing the sort bug, this branch remained disconnected from the corrected sorting pipeline.
* **What it broke:** Information gain values were incorrect, causing suboptimal or wrong feature/threshold selection when `split_method="ig"`.
* **Fix:** Move mask creation and `y_left`/`y_right` partitioning before the `if/elif` split-method check so both branches consume identically sorted partitions.

## 5. Stopping Conditions Failing for Default -1 Hyperparameters

* **Why it happened:** Comparisons like `current_depth < max_depth` evaluate to `False` when `max_depth=-1` (since `0 < -1` is false). The same applies to `node_count < max_nodes`.
* **What it broke:** With default parameters, the tree refused to split at all, remaining a single leaf that predicts the majority class for every sample.
* **Fix:** Map -1 to `float('inf')` for depth/node limits and to 1 for `min_leaf_size` before entering the splitting loop.

## 6. `best_gini` Initialized to -1 (Original Code)

* **Why it happened:** Gini impurity ranges from 0 to 1, but `best_gini` was initialized to -1. The condition `gini < best_gini` could never be true for valid splits.
* **What it broke:** `best_feat` and `best_thrs` remained at initial values (0, 0). The tree always split on Feature 0 at threshold 0 regardless of data distribution.
* **Fix:** Initialize `best_gini = float('inf')` so any valid Gini score can update it.

## 7. Missing Sorting Before Threshold Computation (Original Code)

* **Why it happened:** The original `find_split` iterated through raw row indices without sorting by feature j first. Thresholds were computed between arbitrary adjacent samples.
* **What it broke:** No valid decision boundary could be found between classes. Combined with Bug #6, the tree produced completely random splits.
* **Fix:** Sort data by feature j before iterating, compute midpoints only between consecutive unique sorted values.

## 8. `predict` Traversal Assigning `None` After Leaf Prediction

* **Why it happened:** When reaching a leaf (`lc` or `rc` is `None`), the code correctly computed the prediction but then immediately executed `current = current.lc` (setting `current = None`). While the loop exits correctly, this pattern is fragile and masks whether `split_feat`/`split_thrs` are properly set.
* **What it broke:** If Bugs #6/#7 left `split_feat=None`, the traversal would crash or navigate incorrectly. The pattern also makes debugging harder.
* **Fix:** Add explicit `if not f or not t: predict and break` check at the top of the while loop, ensuring leaves are detected before accessing split attributes.

## 9. IG Division by Zero on Empty Partitions (Original Code)

* **Why it happened:** When all samples went to one partition, `np.unique(y_left)` returned empty arrays, `counts.sum()` was 0, causing division by zero in entropy calculation.
* **What it broke:** Crash or NaN propagation during split evaluation when testing thresholds that produce empty partitions.
* **Fix:** Add `if len(y_left) == 0: e_left = 0` guard before computing entropy (already present in corrected code, but was missing originally).

## 10. `find_split` Called Unconditionally Before Stopping Check

* **Why it happened:** `current.find_split()` ran expensive O(m·n) computation even when `can_split` was `False` (e.g., `max_depth` reached). Nodes that shouldn't split still had `split_feat`/`split_thrs` set.
* **What it broke:** Wasted computation and made non-splitting nodes appear as internal nodes during prediction, leading to incorrect traversal.
* **Fix:** Only call `find_split()` inside the `if can_split:` block, or ensure nodes that shouldn't split have `split_feat = None`.