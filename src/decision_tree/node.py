from scipy.stats import entropy
import numpy as np

class DecisionTreeNode:
    """
    A class to represent a decision tree node
    
    Attributes:
        lc   (DecisionTreeNode):        left child
        rc   (DecisionTreeNode):        right child
        
        X_train    (numpy.ndarray):     training examples at current node
        y_train    (numpy.ndarray):     training lables at current node
        size       (int):               datasetsize
        
        split_feat (int):               number of feature to split
        split_thrs (float):             threshold to split on
        
        ig   (float):                   information gain
        gini (float):                   gini impurity
    """
    
    def __init__(self, X_train, y_train):
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of rows")
        self.X_train = X_train
        self.y_train = y_train
        self.size = X_train.shape[0]
    
        self.lc = None
        self.rc = None
        self.d = 0
        
        self.gini = None
        self.ig = None
        
        self.split_feat = None
        self.split_thrs = None
    
    def __calc_gini_impurity(self, y):
        """Calculate Gini impurity for a single node"""
        if len(y) == 0:
            return 0
        
        # Get class probabilities
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        
        # Gini formula: 1 - sum(p_i^2)
        return 1 - np.sum(p ** 2)

    def __calc_weighted_gini_after_split(self, y_left, y_right):
        """
        Calculate weighted Gini impurity after a split.
        This is what we minimize to find the best split.
        """
        n_total = len(y_left) + len(y_right)
        
        # Handle edge case: empty split
        if n_total == 0:
            return 0
        
        # Calculate Gini for each partition
        gini_left = self.__calc_gini_impurity(y_left)
        gini_right = self.__calc_gini_impurity(y_right)
        
        # Weighted average (weighted by partition size)
        weighted_gini = (len(y_left) / n_total) * gini_left + \
                        (len(y_right) / n_total) * gini_right
        
        return weighted_gini
    
    def depth(self) -> int:
        """
        Calculates and returns the maximum depth of the tree rooted at this node.
        
        Returns:
            int: The maximum depth of the tree (0 for leaf nodes)
        """
        if self.lc is None and self.rc is None:
            # Leaf node - depth is 0
            return 0
        
        left_depth = self.lc.depth() if self.lc is not None else 0
        right_depth = self.rc.depth() if self.rc is not None else 0
        
        return 1 + max(left_depth, right_depth)

    def count_nodes(self) -> int:
        """
        Counts and returns the total number of nodes in the tree rooted at this node.
        
        Returns:
            int: Total number of nodes in the subtree
        """
        # Count this node
        count = 1
        
        # Add nodes from left subtree
        if self.lc is not None:
            count += self.lc.count_nodes()
        
        # Add nodes from right subtree
        if self.rc is not None:
            count += self.rc.count_nodes()
        
        return count
        
    def find_split(self, split_method: str="gini", update_split_criteria: bool=True):
        """
        Determines and sets the feature and threshold pair that results in the best split

        Returns:
            (int, int): feature and threshold to split    
        """
        
        # partitions the data to two subsets based on the best split
        m, n = self.X_train.shape
        
        _, counts = np.unique(self.y_train, return_counts=True)
        probabilities = counts / counts.sum()
        e = entropy(probabilities, base=2)
        
        best_gini = 1
        best_ig = -1
        best_feat = None
        best_thrs = None
        
        for j in range(n):
            # sort the data with respect to the jth feature
            sort_idx = np.argsort(self.X_train[:, j])
            Xj_sorted = self.X_train[sort_idx, j]  # Only sort column j
            y_sorted = self.y_train[sort_idx]     # Keep y aligned
            
            for i in range(m-1):
                if Xj_sorted[i] == Xj_sorted[i + 1]:
                    continue
                # determine the threshold to split data on
                t = (Xj_sorted[i] + Xj_sorted[i+1]) / 2
                
                # Create boolean mask based on threshold
                mask = Xj_sorted <= t

                y_left = y_sorted[mask]
                y_right = y_sorted[~mask]
                
                # calculate the split criteria
                if split_method.lower() == "gini":
                    gini = self.__calc_weighted_gini_after_split(y_left, y_right)
                    if gini < best_gini:
                        best_gini = gini
                        best_feat = j
                        best_thrs = t
                elif split_method.lower() in ["ig", "information gain", "gain"]:
                    _, counts = np.unique(y_left, return_counts=True)
                    if len(y_left) == 0:
                        e_left = 0
                    else:
                        probabilities = counts / counts.sum()
                        e_left = entropy(probabilities, base=2)
                    
                    _, counts = np.unique(y_right, return_counts=True)
                    if len(y_right) == 0:
                        e_right = 0
                    else:
                        probabilities = counts / counts.sum()
                        e_right = entropy(probabilities, base=2)
                    
                    ig = e - ((y_left.shape[0] / self.y_train.shape[0]) * e_left + (y_right.shape[0] / self.y_train.shape[0]) * e_right)
                    if ig > best_ig:
                        best_ig = ig
                        best_feat = j
                        best_thrs = t
                else:
                    raise ValueError("Invalid split method")
            
        if update_split_criteria:
            self.ig = best_ig
            self.gini = best_gini
            
        self.split_feat = best_feat
        self.split_thrs = best_thrs
        
        return (self.split_feat, self.split_thrs)
    
    def split(self) -> tuple[DecisionTreeNode, DecisionTreeNode] | None:
        """
        Partitions the node according to the best split recursively until a stopping condition is reached

        Returns:
            (DecisionTreeNode, DecisionTreeNode): left and right children after split
        """
        if self.split_feat is None or self.split_thrs is None:
            self.find_split()

        if self.split_feat is None or self.split_thrs is None:
            return None

        mask = self.X_train[:, self.split_feat] <= self.split_thrs
        X_left = self.X_train[mask]
        y_left = self.y_train[mask]
        X_right = self.X_train[~mask]
        y_right = self.y_train[~mask]

        if X_left.size == 0 or X_right.size == 0:
            return None

        self.lc = DecisionTreeNode(X_left, y_left)
        self.rc = DecisionTreeNode(X_right, y_right)

        return (self.lc, self.rc)
    
    