# utils_training.py
# This module encapsulates the AutoGluon training process.

from autogluon.tabular import TabularPredictor, TabularDataset

def train_automl_model(train_df, test_df, model_save_path):
    """
    Trains an AutoGluon model and returns the predictor and its leaderboard.
    """
    
    train_data = TabularDataset(train_df)
    test_data = TabularDataset(test_df)
    
    predictor = TabularPredictor(
        label='Calories',
        problem_type='regression',
        eval_metric='root_mean_squared_error',
        path=model_save_path
    ).fit(
        train_data,
        presets='best_quality',
        time_limit=300
    )
    
    print(f"\n--- AutoML Training Complete for models saved in '{model_save_path}' ---")
    
    leaderboard = predictor.leaderboard(test_data, silent=False)
    
    # Return both the predictor and the leaderboard for analysis
    return predictor, leaderboard