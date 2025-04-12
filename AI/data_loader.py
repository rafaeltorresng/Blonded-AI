import pandas as pd
import pickle
import os

def load_dataset(dataset_path):
    print(f"Loading dataset from {dataset_path}...")
    dataset = pd.read_csv(dataset_path)
    
    # Define feature columns
    feature_cols = [
        'popularity', 'danceability', 'energy', 'loudness', 
        'acousticness', 'instrumentalness', 'liveness', 
        'valence', 'tempo'
    ]
    scaled_feature_cols = [f"{col}_scaled" for col in feature_cols]
    
    # Check for missing values in scaled features
    missing_counts = dataset[scaled_feature_cols].isna().sum()
    if missing_counts.sum() > 0:
        print("Warning: Missing scaled feature values found.")
        print(missing_counts[missing_counts > 0])
        
        rows_with_missing = dataset[dataset[scaled_feature_cols].isna().any(axis=1)]
        print(f"Removing {len(rows_with_missing)} tracks with missing values")
        
        dataset = dataset.dropna(subset=scaled_feature_cols)
    
    return dataset, feature_cols, scaled_feature_cols

# Loading scaler model
def load_scaler(scaler_path):
    print(f"Loading scaler from {scaler_path}...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler