import numpy as np
from xgboost.XGRegressionModel import XGRegressionModel
import matplotlib.pyplot as plt

from typing import Any

def test_xgboost_model():
    """
    Tests the XGBoost Regression Model on linear, quadratic, sinusoidal, and exponential relationships
    Creates 9 figures: 1 grid (2x2) for predictions, 4 MSE plots, 4 MAE plots
    """

    # Establish output directory
    output_dir = 'testing_suite_results/xgboost_regression'
    
    # Define test cases
    test_cases: list[dict[str, Any]] = [
        {
            'name': 'Linear',
            'x_data': np.linspace(0, 10, 101),
            'y_data': lambda x: 2 * x + 1,
            'title': 'Linear: y = 2x + 1'
        },
        {
            'name': 'Quadratic',
            'x_data': np.linspace(-5, 5, 101),
            'y_data': lambda x: x ** 2,
            'title': 'Quadratic: y = x²'
        },
        {
            'name': 'Sinusoidal',
            'x_data': np.linspace(0, 4 * np.pi, 101),
            'y_data': lambda x: np.sin(x),
            'title': 'Sinusoidal: y = sin(x)'
        },
        {
            'name': 'Exponential',
            'x_data': np.linspace(0, 3, 101),
            'y_data': lambda x: np.exp(x),
            'title': 'Exponential: y = e^x'
        }
    ]
    
    # Store results for each test case
    results: list[dict[str, Any]] = []
    
    # Test each function
    for test_case in test_cases:
        
        x_data = test_case['x_data']
        y_data = test_case['y_data'](x_data)
        
        # Create and fit model
        model = XGRegressionModel(
            learning_rate=0.1,
            gamma=0.1,
            max_depth=4,
            max_iterations=100
        )
        
        model.fit(x_data, y_data)
        
        # Get predictions
        predictions = np.array([model.predict(x) for x in x_data])
        
        # Store results
        results.append({
            'name': test_case['name'],
            'title': test_case['title'],
            'x_data': x_data,
            'y_data': y_data,
            'predictions': predictions,
            'model': model
        })
    
    # Create 2x2 grid for predictions vs true values
    _, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        ax = axes[i]
        ax.plot(result['x_data'], result['y_data'], 'b-', label='True Values', linewidth=2)
        ax.plot(result['x_data'], result['predictions'], 'r--', label='Predictions', linewidth=2)
        
        ax.set_xlabel('X', fontsize=11)
        ax.set_ylabel('Y', fontsize=11)
        
        ax.set_title(result['title'], fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/predictions_vs_true_values_grid.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create MSE plots for each function
    for result in results:
        plt.figure(figsize=(10, 6))
        iterations = list(range(1, len(result['model'].get_mse_history()) + 1))
        plt.plot(iterations, result['model'].get_mse_history(), 'b-', linewidth=2, marker='o', markersize=3)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Mean Squared Error (MSE)', fontsize=12)
        plt.title(f'{result["title"]} - MSE History', fontsize=14, fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.xlim(left=0)
        
        filename = f'{output_dir}/{result["name"].lower()}_mse_history.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create MAE plots for each function
    for result in results:
        plt.figure(figsize=(10, 6))
        iterations = list(range(1, len(result['model'].get_mae_history()) + 1))
        plt.plot(iterations, result['model'].get_mae_history(), 'r-', linewidth=2, marker='s', markersize=3)
        
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Mean Absolute Error (MAE)', fontsize=12)
        plt.title(f'{result["title"]} - MAE History', fontsize=14, fontweight='bold')
        
        plt.grid(True, alpha=0.3)
        plt.xlim(left=0)
        
        filename = f'{output_dir}/{result["name"].lower()}_mae_history.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
