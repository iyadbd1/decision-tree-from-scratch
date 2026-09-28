from scipy.stats import entropy

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
        
        self.gini = -1
        self.ig = -1
    
    def __calc_gini(self, p):
        return 1 - np.sum(p ** 2)
    
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
        
    def find_split(self, split_method: str="ig", update_split_criteria: bool=True) -> tuple[int, float]:
        """
        Determines and sets the feature and threshold pair that results in the best split

        Returns:
            (int, int): feature and threshold to split    
        """
        if not X_train or not y_train:
            raise ValueError("Cannot split node with no data X_train, y_train")
        
        # partitions the data to two subsets based on the best split
        m, n = X_train.shape
        
        e = entropy(y_train)
        
        best_gini = -1
        best_ig = -1
        best_feat = 0
        best_thrs = 0
        
        for j in range(n):
            # sort the data with respect to the jth feature
            sorted_indices = np.argsort(X_train[:,j])
            sorted_labels = y_train[sorted_indices]
            for i in range(m-1):
                # determine the threshold to split data on
                t = np.mean(X_train[i, j], X_train[i+1, j])
                
                y_left = y_train[X_train[j] <= t]
                y_right = y_train[X_train[j] > t]
                
                # calculate the split criteria
                if split_method.lower() == "gini":
                    gini = self.__calc_gini(np.array([y_left, y_right]))
                    if gini > best_gini:
                        best_gini = gini
                        best_feat = j
                        best_thrs = t
                elif split_method.lower() in ["ig", "information gain", "gain"]:
                    ig = e - ((y_left.shape[0] / y_train.shape[0]) * entropy(y_left) + (y_right.shape[0] / y_train.shape[0]) * entropy(y_right))
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
    
    def split(self) -> tuple[DecisionTreeNode, DecisionTreeNode]:
        """
        Partitions the node according to the best split recursively until a stopping condition is reached
        
        Returns:
            (DecisionTreeNode, DecisionTreeNode): left and right children after split
        """
        
        
        if not self.split_feat or not self.split_thrs:
            self.find_split()     
        
        X_left = X_train[X_train[self.split_feat] <= self.split_thrs]
        y_left = y_train[X_train[self.split_feat] <= self.split_thrs]
        X_right = X_train[X_train[self.split_feat] > self.split_thrs]
        y_right = y_train[X_train[self.split_feat] > self.split_thrs]
        
        self.lc = DecisionTreeNode(X_left, y_left)
        self.rc = DecisionTreeNode(X_right, y_right)
        
        return (self.lc, self.rc)
    