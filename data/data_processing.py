import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle


input_file = '/Users/rafatorres/Desktop/Shaco-AI/data/dataset.csv'
output_file = '/Users/rafatorres/Desktop/Shaco-AI/data/processed_dataset.csv'

def process_csv():
    df = pd.read_csv(input_file, header=None, low_memory=False)
    
    columns = ['ID', 'track_id', 'artist', 'album', 'title', 'popularity', 
               'duration_ms', 'explicit', 'danceability', 'energy', 
               'key', 'loudness', 'mode', 'speechiness', 'acousticness', 
               'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'category']
    
    df.columns = columns
    
    print("\nDataset info:")
    print(df.info())
    
    # Relevant columns for recommendation system
    relevant_columns = [
        'track_id', 'artist', 'title', 'popularity',
        'danceability', 'energy', 'loudness', 
        'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'category'
    ]
    
    df_processed = df[relevant_columns]
    
    # Cleaning the data
    df_processed = df_processed.dropna()
 
    numeric_cols = [
        'popularity', 'danceability', 'energy', 'loudness',
        'acousticness', 'instrumentalness', 'liveness', 
        'valence', 'tempo'
    ]
    
    for col in numeric_cols:
        df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    df_processed = df_processed.dropna()
    
    # Normalizing numeric features
    print("\nNormalizing numeric features...")
    features_to_scale = df_processed[numeric_cols].copy()
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_to_scale)
    
    # Saving the scaler model
    with open('/Users/rafatorres/Desktop/Shaco-AI/data/scaler_model.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # New dataframe with the scaled features
    scaled_df = pd.DataFrame(features_scaled, columns=[f"{col}_scaled" for col in numeric_cols])
    
    # Adding scaled features to the processed dataframe
    df_processed = pd.concat([df_processed, scaled_df], axis=1)

    print("\nVerificando valores ausentes nas features escaladas...")
    missing_scaled = df_processed[[f"{col}_scaled" for col in numeric_cols]].isna().sum()
    if missing_scaled.sum() > 0:
        print(f"Encontrados {missing_scaled.sum()} valores ausentes nas colunas escaladas:")
        print(missing_scaled[missing_scaled > 0])
        print("\nRemovendo linhas com valores ausentes...")
        
        problem_rows = df_processed[df_processed[[f"{col}_scaled" for col in numeric_cols]].isna().any(axis=1)]
        print("\nMúsicas problemáticas que serão removidas:")
        print(problem_rows[['artist', 'title']].head())
        
        df_processed = df_processed.dropna(subset=[f"{col}_scaled" for col in numeric_cols])
        print(f"Restaram {len(df_processed)} músicas após remoção")
        
        # Saving processed data
        print(f"\nSaving processed data to {output_file}")
        df_processed.to_csv(output_file, index=False)
    
    print(f"Processing complete. Kept {df_processed.shape[0]} rows and {df_processed.shape[1]} columns.")
    print(f"Sample of processed data (showing scaled features):")
    print(df_processed[['title', 'artist'] + [f"{col}_scaled" for col in numeric_cols[:3]]].head())

if __name__ == "__main__":
    process_csv()