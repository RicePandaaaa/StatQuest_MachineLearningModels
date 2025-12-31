import numpy as np
from xgboost.RegressionTreeNode import RegressionTreeNode

class RegressionTree:
    def __init__(self, data: np.ndarray, predictions: np.ndarray, learning_rate: float, gamma: float, _lambda: int=0, max_depth: int=6):
        """
        Initializes the regression tree

        Args:
            data          (np.ndarray): The data to train the tree on
            predictions   (np.ndarray): Initial predictions
            learning_rate (float):      The learning rate
            gamma         (float):      Minimum gain for branch pruning
            _lambda       (int):        Regularization parameter (default to 0)
            max_depth     (int):        Maximum depth of the tree (default to 6)
        """

        # Load parameters
        self.data = data
        self.predictions = predictions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self._lambda = _lambda
        self.max_depth = max_depth

        # Create the root node
        self.root = self.create_root()

        # Grow the tree
        self.grow_tree()

    def create_root(self) -> RegressionTreeNode:
        """
        Creates the root node of the tree
        """

        # Calculate the residuals
        residuals = self.predictions - self.data

        return RegressionTreeNode(residuals, self._lambda)

    def grow_tree(self) -> None:
        """
        Grows the tree by recursively creating children nodes until either the maximum depth is reached or there are no more residuals to split
        """

        nodes_to_check = [self.root]

        while nodes_to_check:
            current_node = nodes_to_check.pop(0)
            
            # Ensure there's enough depth to keep growing
            if current_node.get_depth() < self.max_depth:
                current_node.create_children()

                # Add children to the list of nodes to check, only if they exist
                if current_node.get_left():
                    nodes_to_check.append(current_node.get_left())
                if current_node.get_right():
                    nodes_to_check.append(current_node.get_right())
            
            # Forcibly update the output if the node has to be a leaf
            else:
                current_node_output = current_node.calculate_output(current_node.get_residuals())
                current_node.set_output(current_node_output)