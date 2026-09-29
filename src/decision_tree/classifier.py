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

    def __init__(self, min_leaf_size=1, max_depth=float('inf'), max_nodes=float('inf')) -> None:
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        self.root = None

    def fit(
        self, X_train, y_train, min_leaf_size=1, max_depth=float('inf'), max_nodes=float('inf')
    ) -> DecisionTreeClassifier:
        """
        Builds the decision tree based on training set and hyperparameters
        """

        # update hyperparameters
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        
        current_depth = 0
        node_count = 0
        self.root = DecisionTreeNode(X_train, y_train)
        node_count += 1
        

        # define a queue of nodes to split
        queue = [(self.root, 0)]
        while queue:
            current, current_depth = queue.pop(0)

            # check if current node can be split
            check_depth = current_depth < max_depth
            check_size = current.size >= min_leaf_size
            check_node_count = node_count < max_nodes
            check_purity = np.unique(current.y_train).size > 1
            
            can_split = check_depth and check_size and check_node_count and check_purity
            
            if can_split:
                lc, rc = current.split()
                lc.d = current_depth + 1
                rc.d = current_depth + 1
                queue.append((lc, lc.d))
                queue.append((rc, rc.d))
                node_count += 2
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
                
                if not f or not t: 
                    y_pred[i] = majority_vote(current.y_train)
                    break

                if X_test[i, f] <= t:
                    if not current.lc:
                        # leaf reached, perform a majority vote to find the prediction y_pred_i of X_test_i
                        y_pred[i] = majority_vote(current.y_train)
                        break
                    current = current.lc
                else:
                    if not current.rc:
                        # leaf reached, perform a majority vote to find the prediction y_pred_i of X_test_i
                        y_pred[i] = majority_vote(current.y_train)
                        break
                    current = current.rc
        return y_pred

    def predict_proba(self, X_test):
        """
        Predicts the probability of the most likely class of each test point by navigating the decision tree
        """
        if not self.root:
            raise Exception("Cannot predict before fitting the DecisionTreeClassifier")

        m = X_test.shape[0]
        probs_list = []
        for i in range(m):
            # predict each test point's label
            current = self.root
            while current:
                f = current.split_feat
                t = current.split_thrs
                
                if not f or not t: 
                    probs_list.append(predict_proba_classes(current.y_train))
                    break

                if X_test[i, f] <= t:
                    if not current.lc:
                        # leaf reached, perform a majority vote to find the prediction y_pred_i of X_test_i
                        probs_list.append(predict_proba_classes(current.y_train))
                    current = current.lc
                else:
                    if not current.rc:
                        # leaf reached, perform a majority vote to find the prediction y_pred_i of X_test_i
                        probs_list.append(predict_proba_classes(current.y_train))
                    current = current.rc
        return np.array(probs_list)

    def fit_predict(self, X_train, y_train, X_test, min_leaf_size=1, max_depth=float('inf'), max_nodes=float('inf')):
        return self.fit(X_train, y_train, min_leaf_size, max_depth, max_nodes).predict(X_test)
    
