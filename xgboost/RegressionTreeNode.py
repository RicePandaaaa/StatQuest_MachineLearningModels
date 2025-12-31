from __future__ import annotations
import numpy as np

class RegressionTreeNode:
    def __init__(self, residuals: np.ndarray, _lambda: int, depth: int=1):
        """
        Initializes the node along with the residuals, similarity, and gain

        Args:
            residuals (np.ndarray): The residuals of the data
            _lambda   (int):        Regularization parameter (default to 0)
            depth     (int):        The depth of the node within the tree (default to 1)
        """

        # From the data
        self._lambda = _lambda
        self.residuals = np.sort(residuals)
        self.depth = depth

        # Calculated values
        self.similarity = self.calculate_similarity(self.residuals)
        self.gain = 0
        self.threshold = 0
        self.output = float('inf')

        # Children
        self.left = None
        self.right = None

    def calculate_similarity(self, residuals: np.ndarray) -> float:
        """
        Calculate the similarity using the residuals and the lambda

        Defined as
        (sum of residuals)^2 / (number of residuals + lambda)

        Args:
            residuals (np.ndarray): The residuals to calculate the similarity with
        """

        # Ensure denominator is not zero
        if (len(residuals) + self._lambda) == 0:
            raise ZeroDivisionError(f"Sum of number of residuals ({len(residuals)}) and lambda ({self._lambda}) is zero.")

        return np.sum(residuals) ** 2 / (len(residuals) + self._lambda)

    def calculate_output(self, residuals: np.ndarray) -> float:
        """
        Calculate the output using the residuals and the lambda

        Defined as
        sum of residuals / (number of residuals + lambda)
        """

        return np.sum(residuals) / (len(residuals) + self._lambda)

    def create_children(self) -> None:
        """
        Determines data split and creates children
        """

        # Not enough residuals to split
        if len(self.residuals) < 2:
            # Update output as this is a leaf node
            self.output = self.calculate_output(self.residuals)

            return

        # Determine highest gain and corresponding split index
        best_gain = float('-inf')
        split_index = -1

        for index in range(1, len(self.residuals)):
            # Split into left and right
            left_residuals = self.residuals[:index]
            right_residuals = self.residuals[index:]

            # Calculate the gain
            left_similarity = self.calculate_similarity(left_residuals)
            right_similarity = self.calculate_similarity(right_residuals)
            gain = left_similarity + right_similarity - self.similarity

            # Update best gain and split index
            if gain > best_gain:
                best_gain = gain
                split_index = index

        # Update the threshold and gain
        self.gain = best_gain
        self.threshold = (self.residuals[split_index] + self.residuals[split_index - 1]) / 2

        # Create children using split index
        self.left = RegressionTreeNode(self.residuals[:split_index], self._lambda, self.depth + 1)
        self.right = RegressionTreeNode(self.residuals[split_index:], self._lambda, self.depth + 1)

    """
    GETTER METHODS
    """
    def get_residuals(self) -> np.ndarray:
        """
        Returns the residuals of the node
        """
        return self.residuals

    def get_lambda(self) -> int:
        """
        Returns the lambda of the node
        """
        return self._lambda

    def get_depth(self) -> int:
        """
        Returns the depth of the node
        """
        return self.depth

    def get_output(self) -> float:
        """
        Returns the output of the node
        """
        return self.output

    def get_left(self) -> RegressionTreeNode | None:
        """
        Returns the left child of the node
        """
        return self.left

    def get_right(self) -> RegressionTreeNode | None:
        """
        Returns the right child of the node
        """
        return self.right

    """
    SETTER METHODS
    """
    def set_output(self, output: float) -> None:
        """
        Sets the output of the node

        Args:
            output (float): The output to set
        """
        self.output = output

    """
    Overridden methods
    """
    def __str__(self) -> str:
        """
        Returns a string representation of the node
        """
        return f"""
Regression Tree Node
    Depth: {self.depth}
    Similarity: {self.similarity}
    Gain: {self.gain}
    Threshold: {self.threshold}
    Output: {self.output}

    Residuals: {self.residuals}
        """

    def __repr__(self) -> str:
        """
        Returns a string representation of the node
        """
        return self.__str__()