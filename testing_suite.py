
import numpy as np
from xgboost.XGRegressionModel import XGRegressionModel
import matplotlib.pyplot as plt

def test_xgboost_model():
    """
    Tests the XGBoost Regression Model using a simple linear relationship: y = 2x + 1
    """
    
    # Simple linear relationship: y = 2x + 1
    x_data = np.linspace(0, 10, 101)
    y_data = 2 * x_data + 1
    
    # Create model
    model = XGRegressionModel(
        learning_rate=0.1,
        gamma=0.1,
        max_depth=3,
        max_iterations=100
    )
    
    # Fit
    print(f"\nFitting model...")
    model.fit(x_data, y_data)
    
    # Show the predictions versus the true values with a plot
    plt.plot(x_data, y_data, label='True Values')
    plt.plot(x_data, [model.predict(x) for x in x_data], label='Predictions')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('XGBoost Regression Model Predictions vs True Values')
    plt.grid(True)
    plt.legend()

    plt.savefig('testing_suite_results/xgboost_regression/xg_regression_model_predictions_vs_true_values.png')
    plt.close()

    # Show the MSE history
    plt.plot(model.get_mse_history(), label='MSE')

    plt.xlabel('Iteration')
    plt.ylabel('MSE')
    plt.title('XGBoost Regression Model MSE History')
    plt.grid(True)
    plt.legend()

    plt.savefig('testing_suite_results/xgboost_regression/xg_regression_model_mse_history.png')
    plt.close()

    # Show the MAE history
    plt.plot(model.get_mae_history(), label='MAE')

    plt.xlabel('Iteration')
    plt.ylabel('MAE')
    plt.title('XGBoost Regression Model MAE History')
    plt.grid(True)
    plt.legend()

    plt.savefig('testing_suite_results/xgboost_regression/xg_regression_model_mae_history.png')
    plt.close()
