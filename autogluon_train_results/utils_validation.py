# utils_validation.py
# This module handles model comparison, validation, and visualization.

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_model_comparison(lb_unscaled, lb_scaled):
    """Creates a bar chart to compare the performance of top models."""
    print("\n--- Visualizing Model Performance Comparison ---")
    
    # Prepare data for plotting
    top_unscaled = lb_unscaled[['model', 'score_test']].head(3).copy()
    top_unscaled['data_version'] = 'Unscaled'
    
    top_scaled = lb_scaled[['model', 'score_test']].head(3).copy()
    top_scaled['data_version'] = 'Scaled'
    
    plot_df = pd.concat([top_unscaled, top_scaled])
    plot_df['score_test'] = abs(plot_df['score_test'])
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    sns.barplot(data=plot_df, x='model', y='score_test', hue='data_version', palette='viridis')
    
    plt.title('Top 3 Model Performance: Scaled vs. Unscaled Data', fontsize=16)
    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('Root Mean Squared Error (RMSE)', fontsize=12)
    plt.xticks(rotation=15, ha='right')
    plt.legend(title='Data Version')
    plt.tight_layout()
    plt.show()

def compare_and_select_best(unscaled_predictor, scaled_predictor, lb_unscaled, lb_scaled):
    """Compares predictors and selects the best one based on leaderboard score."""
    
    print("\n--- Comparing Model Performance ---")
    
    score_unscaled = abs(lb_unscaled.iloc[0]['score_test'])
    score_scaled = abs(lb_scaled.iloc[0]['score_test'])
    
    print(f"Best Model RMSE (Unscaled Data): {score_unscaled:.4f}")
    print(f"Best Model RMSE (Scaled Data):   {score_scaled:.4f}")
    
    if score_scaled < score_unscaled:
        print("\n--> Conclusion: Scaled data produced the superior model.")
        return scaled_predictor, lb_scaled
    else:
        print("\n--> Conclusion: Unscaled data produced the superior model.")
        return unscaled_predictor, lb_unscaled

def simulate_and_visualize_drift(predictor, leaderboard, test_df):
    """
    Performs validation and simulates data drift by ADDING RANDOM NOISE.
    """
    print("\n--- Final Validation and Data Drift Simulation ---")
    
    best_model_name = leaderboard.iloc[0]['model']
    print(f"--> Using best model for validation: '{best_model_name}'")
    
    X_test = test_df.drop(columns=['Calories'])
    y_test = test_df['Calories']

    # 1. Validate with ORIGINAL test data
    predictions_original = predictor.predict(X_test)
    rmse_original = np.sqrt(mean_squared_error(y_test, predictions_original))
    print(f"1. Validation with ORIGINAL test data | RMSE: {rmse_original:.4f}")

    # 2. Simulate data drift by ADDING RANDOM NOISE
    X_test_modified = X_test.copy()
    cols_to_modify = ['Duration', 'Heart_Rate']
    
    # Generate Gaussian (normal) noise
    # loc=0 (mean), scale=3.0 (standard deviation)
    # The noise will have the same number of rows as the test set and 2 columns
    noise = np.random.normal(loc=0, scale=3.0, size=(len(X_test_modified), len(cols_to_modify)))
    
    # Add the noise to the selected columns
    X_test_modified[cols_to_modify] += noise
    print(f"2. Simulating drift: Added random Gaussian noise to '{cols_to_modify[0]}' and '{cols_to_modify[1]}' columns.")

    # 3. Validate with CHANGED test data
    predictions_modified = predictor.predict(X_test_modified)
    rmse_modified = np.sqrt(mean_squared_error(y_test, predictions_modified))
    print(f"3. Validation with CHANGED test data | RMSE: {rmse_modified:.4f}")

    # 4. Visualize the impact on prediction errors
    print("\n--- Visualizing Impact of Data Drift on Prediction Errors ---")
    errors_original = y_test - predictions_original
    errors_modified = y_test - predictions_modified
    
    plt.figure(figsize=(12, 7))
    sns.kdeplot(errors_original, fill=True, label='Original Data Errors', color='blue')
    sns.kdeplot(errors_modified, fill=True, label='Drifted Data Errors (with noise)', color='red', alpha=0.7)
    
    plt.title('Distribution of Prediction Errors: Original vs. Drifted Data', fontsize=16)
    plt.xlabel('Prediction Error (Actual - Predicted Calories)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()
    plt.axvline(0, color='black', linestyle='--')
    plt.show()
    
    print(f"\nMonitoring Observation: The model's RMSE changed from {rmse_original:.4f} to {rmse_modified:.4f} due to data drift.")