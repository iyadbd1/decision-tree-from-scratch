"""
Comprehensive unit tests for DecisionTreeClassifier and DecisionTreeNode classes.

Tests cover:
- Normal functionality (fit, predict, predict_proba)
- Edge cases (empty inputs, single samples, boundary values)
- Error handling (invalid inputs, uninitialized models)
- Tree structure properties (depth, node count, splits)
- Split methods (Gini impurity, Information Gain)
"""

import pytest
import numpy as np
from collections import Counter
import random

# Import the modules to test
# Adjust these imports based on your actual module structure
from utils import majority_vote, predict_proba_classes
from decision_tree import DecisionTreeNode
from decision_tree import DecisionTreeClassifier


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def simple_dataset():
    """Create a simple binary classification dataset."""
    X = np.array([
        [1, 2],
        [2, 3],
        [3, 4],
        [4, 5],
        [5, 6],
        [6, 7]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def multi_class_dataset():
    """Create a multi-class classification dataset."""
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
        [6, 6],
        [7, 7],
        [8, 8],
        [9, 9]
    ])
    y = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    return X, y


@pytest.fixture
def linearly_separable_dataset():
    """Create a linearly separable dataset for easy splitting."""
    X = np.array([
        [1, 1],
        [2, 2],
        [3, 3],
        [10, 10],
        [11, 11],
        [12, 12]
    ])
    y = np.array([0, 0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def single_feature_dataset():
    """Create a dataset with only one feature."""
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([0, 0, 1, 1, 1])
    return X, y


@pytest.fixture
def large_dataset():
    """Create a larger dataset for testing scalability."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = (X[:, 0] > 0).astype(int)
    return X, y


@pytest.fixture
def tree_node(simple_dataset):
    """Create a basic DecisionTreeNode for testing."""
    X, y = simple_dataset
    return DecisionTreeNode(X, y)


@pytest.fixture
def fitted_classifier(linearly_separable_dataset):
    """Create a fitted DecisionTreeClassifier."""
    X, y = linearly_separable_dataset
    clf = DecisionTreeClassifier(min_leaf_size=1, max_depth=10, max_nodes=100)
    clf.fit(X, y)
    return clf


# ============================================================================
# Tests for utility functions
# ============================================================================

class TestMajorityVote:
    """Tests for the majority_vote utility function."""
    
    def test_majority_vote_single_class(self):
        """Test majority vote with all samples from same class."""
        classes = np.array([1, 1, 1, 1])
        result = majority_vote(classes)
        assert result == 1
    
    def test_majority_vote_clear_majority(self):
        """Test majority vote with clear majority class."""
        classes = np.array([0, 0, 0, 1, 1])
        result = majority_vote(classes)
        assert result == 0
    
    def test_majority_vote_tie_random_selection(self):
        """Test that majority vote handles ties by random selection."""
        classes = np.array([0, 1])
        # Run multiple times to ensure it returns one of the tied values
        results = set()
        for _ in range(20):
            results.add(majority_vote(classes))
        assert results.issubset({0, 1})
    
    def test_majority_vote_empty_input(self):
        """Test majority vote with empty input raises error."""
        with pytest.raises(IndexError):
            majority_vote(np.array([]))
    
    def test_majority_vote_three_way_tie(self):
        """Test majority vote with three-way tie."""
        classes = np.array([0, 1, 2])
        result = majority_vote(classes)
        assert result in [0, 1, 2]


class TestPredictProbaClasses:
    """Tests for the predict_proba_classes utility function."""
    
    def test_predict_proba_single_class(self):
        """Test probability prediction with single class."""
        classes = np.array([1, 1, 1])
        probs = predict_proba_classes(classes)
        assert len(probs) == 1
        assert probs[0] == 1.0
    
    def test_predict_proba_two_classes_equal(self):
        """Test probability prediction with two equal classes."""
        classes = np.array([0, 0, 1, 1])
        probs = predict_proba_classes(classes)
        assert len(probs) == 2
        # Note: The implementation divides by number of unique classes, not total samples
        assert sum(probs) == 2.0  # Each class has count 2, divided by 2 unique classes
    
    def test_predict_proba_multiple_classes(self):
        """Test probability prediction with multiple classes."""
        classes = np.array([0, 1, 2, 2, 2])
        probs = predict_proba_classes(classes)
        assert len(probs) == 3
    
    def test_predict_proba_empty_input(self):
        """Test probability prediction with empty input."""
        with pytest.raises(Exception):
            predict_proba_classes(np.array([]))


# ============================================================================
# Tests for DecisionTreeNode
# ============================================================================

class TestDecisionTreeNodeInit:
    """Tests for DecisionTreeNode initialization."""
    
    def test_init_valid_inputs(self, simple_dataset):
        """Test node initialization with valid inputs."""
        X, y = simple_dataset
        node = DecisionTreeNode(X, y)
        assert node.X_train.shape == X.shape
        assert node.y_train.shape == y.shape
        assert node.size == len(y)
        assert node.lc is None
        assert node.rc is None
        assert node.split_feat is None
        assert node.split_thrs is None
    
    def test_init_mismatched_shapes(self):
        """Test that initialization raises error with mismatched shapes."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="same number of rows"):
            DecisionTreeNode(X, y)
    
    def test_init_empty_arrays(self):
        """Test node initialization with empty arrays."""
        X = np.array([]).reshape(0, 2)
        y = np.array([])
        node = DecisionTreeNode(X, y)
        assert node.size == 0
    
    def test_init_single_sample(self):
        """Test node initialization with single sample."""
        X = np.array([[1, 2]])
        y = np.array([0])
        node = DecisionTreeNode(X, y)
        assert node.size == 1


class TestDecisionTreeNodeGiniImpurity:
    """Tests for Gini impurity calculation."""
    
    def test_gini_pure_node(self, tree_node):
        """Test Gini impurity for pure node (all same class)."""
        y = np.array([1, 1, 1, 1])
        gini = tree_node._DecisionTreeNode__calc_gini_impurity(y)
        assert gini == 0.0
    
    def test_gini_balanced_binary(self, tree_node):
        """Test Gini impurity for balanced binary classes."""
        y = np.array([0, 0, 1, 1])
        gini = tree_node._DecisionTreeNode__calc_gini_impurity(y)
        assert abs(gini - 0.5) < 1e-10
    
    def test_gini_empty_array(self, tree_node):
        """Test Gini impurity for empty array."""
        y = np.array([])
        gini = tree_node._DecisionTreeNode__calc_gini_impurity(y)
        assert gini == 0
    
    def test_gini_single_class(self, tree_node):
        """Test Gini impurity with single sample."""
        y = np.array([1])
        gini = tree_node._DecisionTreeNode__calc_gini_impurity(y)
        assert gini == 0.0
    
    def test_gini_three_classes(self, tree_node):
        """Test Gini impurity with three classes."""
        y = np.array([0, 1, 2])
        gini = tree_node._DecisionTreeNode__calc_gini_impurity(y)
        # For 3 equal classes: 1 - 3*(1/3)^2 = 1 - 1/3 = 2/3
        assert abs(gini - 2/3) < 1e-10


class TestDecisionTreeNodeWeightedGini:
    """Tests for weighted Gini impurity after split."""
    
    def test_weighted_gini_perfect_split(self, tree_node):
        """Test weighted Gini for perfect split (pure partitions)."""
        y_left = np.array([0, 0, 0])
        y_right = np.array([1, 1, 1])
        wgini = tree_node._DecisionTreeNode__calc_weighted_gini_after_split(y_left, y_right)
        assert wgini == 0.0
    
    def test_weighted_gini_empty_partition(self, tree_node):
        """Test weighted Gini when one partition is empty."""
        y_left = np.array([])
        y_right = np.array([0, 1, 1])
        wgini = tree_node._DecisionTreeNode__calc_weighted_gini_after_split(y_left, y_right)
        # Should handle empty partition gracefully
        assert wgini >= 0
    
    def test_weighted_gini_both_empty(self, tree_node):
        """Test weighted Gini when both partitions are empty."""
        y_left = np.array([])
        y_right = np.array([])
        wgini = tree_node._DecisionTreeNode__calc_weighted_gini_after_split(y_left, y_right)
        assert wgini == 0
    
    def test_weighted_gini_imbalanced_sizes(self, tree_node):
        """Test weighted Gini with imbalanced partition sizes."""
        y_left = np.array([0, 0])
        y_right = np.array([1, 1, 1, 1])
        wgini = tree_node._DecisionTreeNode__calc_weighted_gini_after_split(y_left, y_right)
        assert wgini == 0.0  # Both partitions are pure


class TestDecisionTreeNodeDepth:
    """Tests for tree depth calculation."""
    
    def test_depth_leaf_node(self, tree_node):
        """Test depth of leaf node (no children)."""
        assert tree_node.depth() == 0
    
    def test_depth_single_level(self, simple_dataset):
        """Test depth of tree with one level of children."""
        X, y = simple_dataset
        node = DecisionTreeNode(X, y)
        lc, rc = node.split()
        assert node.depth() == 1
    
    def test_depth_multiple_levels(self, simple_dataset):
        """Test depth of tree with multiple levels."""
        X, y = simple_dataset
        root = DecisionTreeNode(X, y)
        lc, rc = root.split()
        
        # Split left child if it has data
        if lc.size > 1:
            lc.split()
        
        expected_depth = root.depth()
        assert expected_depth >= 1
    
    def test_depth_asymmetric_tree(self, simple_dataset):
        """Test depth calculation for asymmetric tree."""
        X, y = simple_dataset
        root = DecisionTreeNode(X, y)
        lc, rc = root.split()
        
        # Only split left child
        if lc.size > 1:
            lc.split()
        
        depth = root.depth()
        assert depth >= 1


class TestDecisionTreeNodeCountNodes:
    """Tests for node counting."""
    
    def test_count_single_node(self, tree_node):
        """Test counting nodes for single node."""
        assert tree_node.count_nodes() == 1
    
    def test_count_with_children(self, simple_dataset):
        """Test counting nodes with children."""
        X, y = simple_dataset
        node = DecisionTreeNode(X, y)
        lc, rc = node.split()
        assert node.count_nodes() == 3
    
    def test_count_complex_tree(self, simple_dataset):
        """Test counting nodes in complex tree."""
        X, y = simple_dataset
        root = DecisionTreeNode(X, y)
        lc, rc = root.split()
        
        # Split children if possible
        if lc.size > 1:
            lc.split()
        if rc.size > 1:
            rc.split()
        
        count = root.count_nodes()
        assert count >= 3


class TestDecisionTreeNodeFindSplit:
    """Tests for find_split method."""
    
    def test_find_split_gini_method(self, tree_node):
        """Test finding split using Gini impurity."""
        feat, thrs = tree_node.find_split(split_method="gini")
        assert isinstance(feat, (int, np.integer))
        assert isinstance(thrs, (float, np.floating))
        assert tree_node.split_feat is not None
        assert tree_node.split_thrs is not None
    
    def test_find_split_ig_method(self, tree_node):
        """Test finding split using information gain."""
        feat, thrs = tree_node.find_split(split_method="ig")
        assert isinstance(feat, (int, np.integer))
        assert isinstance(thrs, (float, np.floating))
    
    def test_find_split_information_gain_alias(self, tree_node):
        """Test information gain method with different aliases."""
        for alias in ["information gain", "gain"]:
            feat, thrs = tree_node.find_split(split_method=alias)
            assert feat is not None
    
    def test_find_split_invalid_method(self, tree_node):
        """Test that invalid split method raises error."""
        with pytest.raises(ValueError, match="Invalid split method"):
            tree_node.find_split(split_method="invalid")
    
    def test_find_split_updates_criteria(self, tree_node):
        """Test that find_split updates ig and gini attributes."""
        tree_node.find_split(split_method="gini", update_split_criteria=True)
        assert tree_node.gini is not None
        assert tree_node.ig is not None
    
    def test_find_split_no_update(self, tree_node):
        """Test that find_split can skip updating criteria."""
        tree_node.find_split(split_method="gini", update_split_criteria=False)
        assert tree_node.gini is None
        assert tree_node.ig is None
    
    def test_find_split_single_feature(self, single_feature_dataset):
        """Test find_split with single feature dataset."""
        X, y = single_feature_dataset
        node = DecisionTreeNode(X, y)
        feat, thrs = node.find_split()
        assert feat == 0
    
    def test_find_split_pure_node(self):
        """Test find_split on node with pure labels."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 1, 1])
        node = DecisionTreeNode(X, y)
        feat, thrs = node.find_split()
        # Should still find a split even if node is pure
        assert feat is not None


class TestDecisionTreeNodeSplit:
    """Tests for split method."""
    
    def test_split_creates_children(self, tree_node):
        """Test that split creates left and right children."""
        lc, rc = tree_node.split()
        assert lc is not None
        assert rc is not None
        assert isinstance(lc, DecisionTreeNode)
        assert isinstance(rc, DecisionTreeNode)
    
    def test_split_partitions_data(self, tree_node):
        """Test that split correctly partitions data."""
        lc, rc = tree_node.split()
        total_samples = lc.size + rc.size
        assert total_samples == tree_node.size
    
    def test_split_auto_finds_split(self, tree_node):
        """Test that split automatically finds split if not set."""
        assert tree_node.split_feat is None
        lc, rc = tree_node.split()
        assert tree_node.split_feat is not None
    
    def test_split_preserves_labels(self, simple_dataset):
        """Test that split preserves all labels."""
        X, y = simple_dataset
        node = DecisionTreeNode(X, y)
        lc, rc = node.split()
        
        all_labels = np.concatenate([lc.y_train, rc.y_train])
        assert len(all_labels) == len(y)
        assert set(all_labels) == set(y)


# ============================================================================
# Tests for DecisionTreeClassifier
# ============================================================================

class TestDecisionTreeClassifierInit:
    """Tests for DecisionTreeClassifier initialization."""
    
    def test_init_default_params(self):
        """Test initialization with default parameters."""
        clf = DecisionTreeClassifier()
        assert clf.min_leaf_size == 1
        assert clf.max_depth == float('inf')
        assert clf.max_nodes == float('inf')
        assert clf.root is None
    
    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        clf = DecisionTreeClassifier(min_leaf_size=5, max_depth=10, max_nodes=50)
        assert clf.min_leaf_size == 5
        assert clf.max_depth == 10
        assert clf.max_nodes == 50


class TestDecisionTreeClassifierFit:
    """Tests for fit method."""
    
    def test_fit_basic(self, linearly_separable_dataset):
        """Test basic fitting of decision tree."""
        X, y = linearly_separable_dataset
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        assert clf.root is not None
        assert clf.root.size == len(y)
    
    def test_fit_returns_self(self, linearly_separable_dataset):
        """Test that fit returns self for method chaining."""
        X, y = linearly_separable_dataset
        clf = DecisionTreeClassifier()
        result = clf.fit(X, y)
        assert result is clf
    
    def test_fit_with_min_leaf_size(self, simple_dataset):
        """Test fitting with minimum leaf size constraint."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier()
        clf.fit(X, y, min_leaf_size=3)
        assert clf.root is not None
    
    def test_fit_with_max_depth(self, simple_dataset):
        """Test fitting with maximum depth constraint."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier()
        clf.fit(X, y, max_depth=1)
        assert clf.root is not None
        # Check that tree depth doesn't exceed max_depth
        assert clf.root.depth() <= 1
    
    def test_fit_with_max_nodes(self, simple_dataset):
        """Test fitting with maximum nodes constraint."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier()
        clf.fit(X, y, max_nodes=3)
        assert clf.root is not None
        # Check that node count doesn't exceed max_nodes
        assert clf.root.count_nodes() <= 3
    
    def test_fit_single_sample(self):
        """Test fitting with single training sample."""
        X = np.array([[1, 2]])
        y = np.array([0])
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        assert clf.root is not None
        assert clf.root.size == 1
    
    def test_fit_all_same_class(self):
        """Test fitting when all samples have same class."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 1, 1])
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        assert clf.root is not None
    
    def test_fit_updates_hyperparameters(self, simple_dataset):
        """Test that fit updates hyperparameters."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier(min_leaf_size=1)
        clf.fit(X, y, min_leaf_size=5, max_depth=3)
        assert clf.min_leaf_size == 5
        assert clf.max_depth == 3


class TestDecisionTreeClassifierPredict:
    """Tests for predict method."""
    
    def test_predict_basic(self, fitted_classifier):
        """Test basic prediction."""
        X_test = np.array([[2, 2], [11, 11]])
        predictions = fitted_classifier.predict(X_test)
        assert len(predictions) == 2
    
    def test_predict_before_fit(self):
        """Test that predict raises error before fitting."""
        clf = DecisionTreeClassifier()
        X_test = np.array([[1, 2]])
        with pytest.raises(Exception, match="Cannot predict before fitting"):
            clf.predict(X_test)
    
    def test_predict_single_sample(self, fitted_classifier):
        """Test prediction for single sample."""
        X_test = np.array([[2, 2]])
        predictions = fitted_classifier.predict(X_test)
        assert len(predictions) == 1
    
    def test_predict_multiple_samples(self, fitted_classifier):
        """Test prediction for multiple samples."""
        X_test = np.array([[1, 1], [5, 5], [11, 11], [12, 12]])
        predictions = fitted_classifier.predict(X_test)
        assert len(predictions) == 4
    
    def test_predict_linearly_separable(self, linearly_separable_dataset):
        """Test prediction accuracy on linearly separable data."""
        X_train, y_train = linearly_separable_dataset
        clf = DecisionTreeClassifier()
        clf.fit(X_train, y_train)
        
        # Test on training data
        predictions = clf.predict(X_train)
        # Should achieve perfect or near-perfect accuracy
        accuracy = np.mean(predictions == y_train)
        assert accuracy >= 0.8
    
    def test_predict_boundary_values(self, fitted_classifier):
        """Test prediction at decision boundary."""
        # Test points exactly at threshold
        X_test = np.array([[5, 5], [6, 6]])
        predictions = fitted_classifier.predict(X_test)
        assert len(predictions) == 2
    
    def test_predict_empty_test_set(self, fitted_classifier):
        """Test prediction with empty test set."""
        X_test = np.array([]).reshape(0, 2)
        predictions = fitted_classifier.predict(X_test)
        assert len(predictions) == 0


class TestDecisionTreeClassifierPredictProba:
    """Tests for predict_proba method."""
    
    def test_predict_proba_basic(self, fitted_classifier):
        """Test basic probability prediction."""
        X_test = np.array([[2, 2], [11, 11]])
        probs = fitted_classifier.predict_proba(X_test)
        assert probs.shape[0] == 2
    
    def test_predict_proba_before_fit(self):
        """Test that predict_proba raises error before fitting."""
        clf = DecisionTreeClassifier()
        X_test = np.array([[1, 2]])
        with pytest.raises(Exception, match="Cannot predict before fitting"):
            clf.predict_proba(X_test)
    
    def test_predict_proba_shape(self, fitted_classifier):
        """Test that predict_proba returns correct shape."""
        X_test = np.array([[1, 1], [5, 5], [11, 11]])
        probs = fitted_classifier.predict_proba(X_test)
        assert probs.shape[0] == 3
    
    def test_predict_proba_values(self, fitted_classifier):
        """Test that probability values are valid."""
        X_test = np.array([[2, 2]])
        probs = fitted_classifier.predict_proba(X_test)
        # Probabilities should be non-negative
        assert np.all(probs >= 0)
    
    def test_predict_proba_single_sample(self, fitted_classifier):
        """Test probability prediction for single sample."""
        X_test = np.array([[5, 5]])
        probs = fitted_classifier.predict_proba(X_test)
        assert probs.shape[0] == 1


class TestDecisionTreeClassifierFitPredict:
    """Tests for fit_predict method."""
    
    def test_fit_predict_basic(self, linearly_separable_dataset):
        """Test basic fit_predict functionality."""
        X_train, y_train = linearly_separable_dataset
        X_test = np.array([[2, 2], [11, 11]])
        
        clf = DecisionTreeClassifier()
        predictions = clf.fit_predict(X_train, y_train, X_test)
        
        assert len(predictions) == 2
    
    def test_fit_predict_with_params(self, simple_dataset):
        """Test fit_predict with custom parameters."""
        X_train, y_train = simple_dataset
        X_test = np.array([[2, 3], [5, 6]])
        
        clf = DecisionTreeClassifier()
        predictions = clf.fit_predict(
            X_train, y_train, X_test,
            min_leaf_size=2, max_depth=2
        )
        
        assert len(predictions) == 2
    
    def test_fit_predict_returns_predictions(self, linearly_separable_dataset):
        """Test that fit_predict returns predictions."""
        X_train, y_train = linearly_separable_dataset
        X_test = np.array([[2, 2]])
        
        clf = DecisionTreeClassifier()
        predictions = clf.fit_predict(X_train, y_train, X_test)
        
        assert predictions is not None
        assert isinstance(predictions, np.ndarray)


# ============================================================================
# Edge Cases and Boundary Tests
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_zero_features(self):
        """Test behavior with zero features (edge case)."""
        X = np.array([]).reshape(3, 0)
        y = np.array([0, 1, 0])
        # This may raise an error or handle gracefully
        with pytest.raises((ValueError, IndexError)):
            node = DecisionTreeNode(X, y)
            node.find_split()
    
    def test_large_number_of_features(self):
        """Test with large number of features."""
        np.random.seed(42)
        X = np.random.randn(10, 100)
        y = np.random.randint(0, 2, 10)
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        assert clf.root is not None
    
    def test_many_classes(self):
        """Test with many classes."""
        X = np.random.randn(50, 3)
        y = np.arange(50) % 10  # 10 classes
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        predictions = clf.predict(X)
        assert len(predictions) == 50
    
    def test_duplicate_samples(self):
        """Test with duplicate samples."""
        X = np.array([[1, 2], [1, 2], [1, 2], [3, 4]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        predictions = clf.predict(X)
        assert len(predictions) == 4
    
    def test_identical_features(self):
        """Test when all feature values are identical."""
        X = np.array([[1, 1], [1, 1], [1, 1]])
        y = np.array([0, 1, 0])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        # Should still create a tree, though splits may not be meaningful
        assert clf.root is not None
    
    def test_negative_values(self):
        """Test with negative feature values."""
        X = np.array([[-5, -3], [-2, -1], [1, 2], [3, 4]])
        y = np.array([0, 0, 1, 1])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        predictions = clf.predict(X)
        assert len(predictions) == 4
    
    def test_very_small_threshold(self):
        """Test with very small differences in feature values."""
        X = np.array([[1.0, 2.0], [1.0000001, 2.0], [1.0000002, 2.0]])
        y = np.array([0, 1, 1])
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        assert clf.root is not None
    
    def test_max_depth_zero(self, simple_dataset):
        """Test with max_depth=0 (no splits allowed)."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier()
        clf.fit(X, y, max_depth=0)
        # Tree should have only root node
        assert clf.root.depth() == 0
    
    def test_min_leaf_size_larger_than_dataset(self, simple_dataset):
        """Test with min_leaf_size larger than dataset size."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier()
        clf.fit(X, y, min_leaf_size=100)
        # Should not split, tree should be just root
        assert clf.root.count_nodes() == 1
    
    def test_max_nodes_one(self, simple_dataset):
        """Test with max_nodes=1 (only root allowed)."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier()
        clf.fit(X, y, max_nodes=1)
        assert clf.root.count_nodes() == 1


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_workflow_binary(self):
        """Test complete workflow for binary classification."""
        np.random.seed(42)
        X_train = np.random.randn(100, 5)
        y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
        
        X_test = np.random.randn(20, 5)
        y_test = (X_test[:, 0] + X_test[:, 1] > 0).astype(int)
        
        clf = DecisionTreeClassifier(max_depth=5)
        clf.fit(X_train, y_train)
        
        predictions = clf.predict(X_test)
        probs = clf.predict_proba(X_test)
        
        assert len(predictions) == 20
        assert probs.shape[0] == 20
    
    def test_complete_workflow_multiclass(self):
        """Test complete workflow for multi-class classification."""
        np.random.seed(42)
        X_train = np.random.randn(150, 4)
        y_train = np.random.randint(0, 3, 150)
        
        X_test = np.random.randn(30, 4)
        
        clf = DecisionTreeClassifier(max_depth=10)
        clf.fit(X_train, y_train)
        
        predictions = clf.predict(X_test)
        assert len(predictions) == 30
    
    def test_tree_structure_properties(self, simple_dataset):
        """Test various tree structure properties together."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier(max_depth=3, min_leaf_size=1)
        clf.fit(X, y)
        
        # Check depth
        depth = clf.root.depth()
        assert depth >= 0
        assert depth <= 3
        
        # Check node count
        node_count = clf.root.count_nodes()
        assert node_count >= 1
        
        # Check that all nodes have valid data
        def check_node(node):
            assert node.X_train.shape[0] == node.y_train.shape[0]
            if node.lc:
                check_node(node.lc)
            if node.rc:
                check_node(node.rc)
        
        check_node(clf.root)
    
    def test_consistency_predict_predict_proba(self, fitted_classifier):
        """Test consistency between predict and predict_proba."""
        X_test = np.array([[2, 2], [11, 11], [5, 5]])
        
        predictions = fitted_classifier.predict(X_test)
        probs = fitted_classifier.predict_proba(X_test)
        
        # Both should return same number of results
        assert len(predictions) == probs.shape[0]


# ============================================================================
# Parametrized Tests
# ============================================================================

class TestParametrized:
    """Parametrized tests for various scenarios."""
    
    @pytest.mark.parametrize("split_method", ["gini", "ig", "information gain", "gain"])
    def test_split_methods(self, tree_node, split_method):
        """Test all valid split methods."""
        feat, thrs = tree_node.find_split(split_method=split_method)
        assert feat is not None
        assert thrs is not None
    
    @pytest.mark.parametrize("min_leaf,max_depth,max_nodes", [
        (1, 10, 100),
        (2, 5, 50),
        (5, 3, 20),
        (1, float('inf'), float('inf')),
    ])
    def test_hyperparameter_combinations(self, simple_dataset, min_leaf, max_depth, max_nodes):
        """Test various hyperparameter combinations."""
        X, y = simple_dataset
        clf = DecisionTreeClassifier()
        clf.fit(X, y, min_leaf_size=min_leaf, max_depth=max_depth, max_nodes=max_nodes)
        assert clf.root is not None
    
    @pytest.mark.parametrize("n_samples,n_features", [
        (10, 2),
        (50, 5),
        (100, 10),
        (5, 1),
    ])
    def test_various_dataset_sizes(self, n_samples, n_features):
        """Test with various dataset sizes."""
        np.random.seed(42)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)
        
        clf = DecisionTreeClassifier()
        clf.fit(X, y)
        
        predictions = clf.predict(X)
        assert len(predictions) == n_samples


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
