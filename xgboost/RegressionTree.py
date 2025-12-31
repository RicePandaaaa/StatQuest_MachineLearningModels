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

    def create_root(self) -> RegressionTreeNode:
        """
        Creates the root node of the tree
        """

        # Calculate the residuals
        residuals = self.predictions - self.data

        return RegressionTreeNode(residuals, self._lambda)
