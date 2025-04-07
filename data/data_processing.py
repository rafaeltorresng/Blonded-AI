import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Path to the dataset
input_file = '/Users/rafatorres/Desktop/Shaco-AI/data/dataset.csv'
output_file = '/Users/rafatorres/Desktop/Shaco-AI/data/processed_dataset.csv'
model_file = '/Users/rafatorres/Desktop/Shaco-AI/data/scaler_model.pkl'

def process_csv():
    # Read the CSV file
    print("Reading the dataset...")
    
    # The file doesn't have header names, so let's analyze its structure first
    df = pd.read_csv(input_file, header=None)
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    
    # Based on the data preview, let's define column names
    columns = ['ID', 'track_id', 'artist', 'album', 'title', 'popularity', 
               'duration_ms', 'explicit', 'danceability', 'energy', 
               'key', 'loudness', 'mode', 'speechiness', 'acousticness', 
               'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'category']
    
    # Assign column names (if the count doesn't match exactly, adjust accordingly)
    if len(columns) == df.shape[1]:
        df.columns = columns
    else:
        print(f"Warning: Column count mismatch. Expected {len(columns)}, got {df.shape[1]}")
        # Use numeric columns as fallback
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    
    # Display information about the dataframe
    print("\nDataset info:")
    print(df.info())
    
    # Select relevant columns for recommendation system
    relevant_columns = [
        'track_id', 'artist', 'title', 'popularity',
        'danceability', 'energy', 'loudness', 
        'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'category'
    ]
    
    df_processed = df[relevant_columns]
    
    # Clean the data
    # 1. Remove any rows with missing values
    df_processed = df_processed.dropna()
    
    # 2. Ensure numerical columns are of the right type
    numeric_cols = [
        'popularity', 'danceability', 'energy', 'loudness',
        'acousticness', 'instrumentalness', 'liveness', 
        'valence', 'tempo'
    ]
    
    for col in numeric_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # 3. Remove any rows with invalid numeric values
    df_processed = df_processed.dropna()
    
    # 4. Normalize numeric features
    print("\nNormalizing numeric features...")
    # Create a copy of the dataframe with only numeric features for scaling
    features_to_scale = df_processed[numeric_cols].copy()
    
    # Use StandardScaler to normalize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_to_scale)
    
    # Create a new dataframe with the scaled features
    scaled_df = pd.DataFrame(features_scaled, columns=[f"{col}_scaled" for col in numeric_cols])
    
    # Add the scaled features to the processed dataframe
    df_processed = pd.concat([df_processed, scaled_df], axis=1)
    
    # Save the processed data
    print(f"\nSaving processed data to {output_file}")
    df_processed.to_csv(output_file, index=False)
    
    print(f"Processing complete. Kept {df_processed.shape[0]} rows and {df_processed.shape[1]} columns.")
    print(f"Sample of processed data (showing scaled features):")
    print(df_processed[['title', 'artist'] + [f"{col}_scaled" for col in numeric_cols[:3]]].head())

if __name__ == "__main__":
    process_csv()