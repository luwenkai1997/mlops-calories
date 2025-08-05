# utils_data.py
# This module handles all basic data loading, cleaning, and preparation tasks.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# FIX: The function name is now 'load_data' to match the call in main.ipynb
def load_data(file_path='raw_data/calories.csv'):
    """Loads the initial dataset from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        # Best practice: ensure the label is a float for regression tasks
        df['Calories'] = df['Calories'].astype(float)
        print(f"Data loaded from {file_path}. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {file_path}. Please upload it.")
        raise

def prepare_data_versions(df):
    """Creates two versions of the data: unscaled and scaled."""
    if df is None:
        return None, None
        
    # 1. One-hot encode the 'Gender' column
    df_encoded = pd.get_dummies(df, columns=['Gender'], drop_first=True)
    
    # 2. Create the scaled version
    numeric_cols = [c for c in df_encoded.columns if c not in ['User_ID', 'Calories', 'Gender_male']]
    scaler = StandardScaler()
    df_scaled = df_encoded.copy()
    df_scaled[numeric_cols] = scaler.fit_transform(df_encoded[numeric_cols])
    
    print("Created both unscaled and scaled data versions.")
    return df_encoded, df_scaled