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
    
    # Check for missing values in scaled features
    missing_cols = [col for col in feature_cols if col not in dataset.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns in dataset: {missing_cols}")
        
    if dataset[feature_cols].isna().any(axis=1).sum() > 0:
        print("Warning: Removing tracks with missing feature values.")
        dataset = dataset.dropna(subset=feature_cols)
    
    return dataset, feature_cols

# Loading scaler model
def load_scaler(scaler_path):
    print(f"Loading scaler from {scaler_path}...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler