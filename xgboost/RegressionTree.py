import numpy as np
from xgboost.RegressionTreeNode import RegressionTreeNode

class RegressionTree:
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, predictions: np.ndarray, learning_rate: float, gamma: float, _lambda: int=0, max_depth: int=6):
        """
        Initializes the regression tree

        Args:
            x_data        (np.ndarray): The x data
            y_data        (np.ndarray): The y data
            predictions   (np.ndarray): Initial predictions (of y values)
            learning_rate (float):      The learning rate
            gamma         (float):      Minimum gain for branch pruning
            _lambda       (int):        Regularization parameter (default to 0)
            max_depth     (int):        Maximum depth of the tree (default to 6)
        """

        # Load parameters
        self.x_data = x_data
        self.y_data = y_data
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

        return RegressionTreeNode(self.x_data, self.y_data, self.predictions, self._lambda)

    def grow_tree(self) -> None:
        """
        Grows the tree by recursively creating children nodes until 
        either the maximum depth is reached or there are no more residuals to split
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

    def prune_tree(self, node: RegressionTreeNode) -> None:
        """
        Recursively prunes a tree by checking if a node is a valid candidate for pruning and
        checking if the gain is less than the gamma parameter. If so, the branch is pruned by
        resetting it to a leaf node

        Args:
            node (RegressionTreeNode): The node to potentially prune the tree from
        """

        # Edge case checking for nodes that should NOT be pruned
        left_child, right_child = node.get_left(), node.get_right()

        # Edge case 1: The node is a leaf
        if not left_child and not right_child:
            return

        # Edge case 2: One of the two children has children (protected ancestor)
        if left_child and (left_child.get_left() or left_child.get_right()):
            return
        if right_child and (right_child.get_left() or right_child.get_right()):
            return

        # At this point, the node has two leaf children and is a valid candidate for pruning

        # If the gain is less than the gamma parameter, prune the branch by condensing the node into a leaf node
        if node.get_gain() < self.gamma:
            node.condense()
            

        