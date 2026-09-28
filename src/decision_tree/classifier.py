from .node import DecisionTreeNode
from utils import majority_vote, predict_proba_classes
import numpy as np

class DecisionTreeClassifier:
    """
    Represents a decision tree classification model with standard fit and predict API

    Attributes:
        root     (DecisionTreeNode): decision tree root node

        X_train  (numpy.ndarray):    training examples
        y_train  (numpy.ndarray):    training lables
        X_test   (numpy.ndarray):    test examples

        min_leaf_size (int):         minimum number of examples inside a node to split it
        max_depth (int):             maximum tree depth
        max_nodes (int):             maximum number of nodes
    """

    def __init__(self, min_leaf_size=-1, max_depth=-1, max_nodes=-1) -> None:
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.root = None

    def fit(
        self, X_train, y_train, min_leaf_size=-1, max_depth=-1, max_nodes=-1
    ) -> DecisionTreeClassifier:
        """
        Builds the decision tree based on training set and hyperparameters
        """

        # update hyperparameters
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.max_nodes = max_nodes

        self.root = DecisionTreeNode(X_train, y_train)

        # define a queue of nodes to split
        queue = [self.root]
        while queue:
            current = queue.pop(0)

            # check if current node can be split
            check_depth = current.depth() <= max_depth
            check_size = current.size >= min_leaf_size
            check_node_count = current.count_nodes() < max_nodes

            can_split = check_depth and check_size and check_node_count

            current.find_split()
            if can_split:
                current.split()
                queue.extend(current.split())
        return self

    def predict(self, X_test):
        """
        Predicts the class of each test point by navigating the decision tree
        """
        if not self.root:
            raise Exception("Cannot predict before fitting the DecisionTreeClassifier")

        m = X_test.shape[0]
        y_pred = np.empty(m)
        for i in range(m):
            # predict each test point's label
            current = self.root
            while current:
                f = current.split_feat
                t = current.split_thrs

                if X_test[i, f] <= t:
                    if not current.lc:
                        # leaf reached, perform a majority vote to find the prediction y_pred_i of X_test_i
                        y_pred[i] = majority_vote(current.y_train)
                    current = current.lc
                else:
                    if not current.rc:
                        # leaf reached, perform a majority vote to find the prediction y_pred_i of X_test_i
                        y_pred[i] = majority_vote(current.y_train)
                    current = current.rc
        return y_pred

    def predict_proba(self, X_test):
        """
        Predicts the probability of the most likely class of each test point by navigating the decision tree
        """
        if not self.root:
            raise Exception("Cannot predict before fitting the DecisionTreeClassifier")

        m = X_test.shape[0]
        y_pred_probs = []
        for i in range(m):
            # predict each test point's label
            current = self.root
            while current:
                f = current.split_feat
                t = current.split_thrs

                if X_test[i, f] <= t:
                    if not current.lc:
                        # leaf reached, perform a majority vote to find the prediction y_pred_i of X_test_i
                        y_pred_probs[i] = predict_proba_classes(current.y_train)
                    current = current.lc
                else:
                    if not current.rc:
                        # leaf reached, perform a majority vote to find the prediction y_pred_i of X_test_i
                        y_pred_probs[i] = predict_proba_classes(current.y_train)
                    current = current.rc
        return y_pred_probs

    def fit_predict(self, X_train, y_train, X_test, min_leaf_size=-1, max_depth=-1, max_nodes=-1):
        return self.fit(X_train, y_train, min_leaf_size, max_depth, max_nodes).predict(X_test)