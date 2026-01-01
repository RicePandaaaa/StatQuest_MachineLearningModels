from xgboost.RegressionTree import RegressionTree
import numpy as np

class XGRegressionModel:
    def __init__(self, learning_rate: float, gamma: float, _lambda: int=0, max_depth: int=6, max_iterations: int=100):
        """
        Initializes the XGRegressionModel

        Args:
            learning_rate (float):      The learning rate
            gamma         (float):      Minimum gain for branch pruning
            _lambda       (int):        Regularization parameter (default to 0)
            max_depth     (int):        Maximum depth of the tree (default to 6)
            max_iterations (int):       Maximum number of iterations (default to 100)
        """

        # Load parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self._lambda = _lambda
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        
        # Store all trees
        self.trees: list[RegressionTree] = []

        # Store training data
        self.mse_history: list[float] = []
        self.mae_history: list[float] = []

    def fit(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        """
        Fits the model to the data using gradient boosting
        
        Args:
            x_data (np.ndarray): The x (feature) data
            y_data (np.ndarray): The y (target) data

        Returns:
            None
        """
        
        # Initialize predictions to 0.5
        predictions = np.full(len(x_data), 0.5)
        
        # Iterate through the maximum number of iterations
        for i in range(self.max_iterations):
            # Create a tree to fit the residuals
            tree = RegressionTree(
                x_data, 
                y_data, 
                predictions, 
                self.learning_rate, 
                self.gamma, 
                self._lambda, 
                self.max_depth
            )
            
            # Store the tree
            self.trees.append(tree)
            
            # Get predictions from this tree
            tree_outputs = np.array([tree.predict(x) for x in x_data])
            
            # Update predictions: predictions = predictions + learning_rate * tree_outputs
            new_predictions = predictions + self.learning_rate * tree_outputs

            
            # Update predictions for next iteration
            predictions = new_predictions

            # Output progress
            mse = np.mean((y_data - new_predictions) ** 2)
            mae = np.mean(np.abs(y_data - new_predictions))
            self.mse_history.append(mse)
            self.mae_history.append(mae)

            print(f"Iteration {i + 1}: MSE = {mse:.6f}, MAE = {mae:.6f}")

    def predict(self, x: float) -> float:
        """
        Predicts the y value for a given x value using all trees
        
        Args:
            x (float): The x value to predict for
            
        Returns:
            float: The predicted y value
        """
        if not self.trees:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        # Start with 0.5 as the initial prediction
        prediction = 0.5
        
        # Add contributions from all trees
        for tree in self.trees:
            tree_prediction = tree.predict(x)
            prediction += self.learning_rate * tree_prediction
        
        return prediction

    def get_mse_history(self) -> list[float]:
        """
        Returns the MSE history
        """
        return self.mse_history

    def get_mae_history(self) -> list[float]:
        """
        Returns the MAE history
        """
        return self.mae_history