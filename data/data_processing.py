import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
input_file = os.path.join(base_dir, 'data', 'dataset.csv')
output_file = os.path.join(base_dir, 'data', 'processed_dataset.csv')
scaler_path = os.path.join(base_dir, 'model', 'scaler_model.pkl')


def process_csv():
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {input_file}")
        return

    print(f"Loading dataset from {input_file}...")
    try:
        # Use efficient data types and only load necessary columns initially
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    print("\nInitial dataset info:")
    print(df.info())

    # Selecting relevant columns
    relevant_columns = [
        'track_id', 'artists', 'track_name', 'popularity',
        'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo', 'track_genre'
    ]

    missing_relevant = [col for col in relevant_columns if col not in df.columns]
    if missing_relevant:
        print(f"Error: Relevant columns missing from dataset after loading: {missing_relevant}")
        print("Please ensure dataset.csv has the correct header row.")
        return

    # Select and rename columns in one operation
    df_processed = df[relevant_columns].rename(columns={
        'artists': 'artist',
        'track_name': 'title',
        'track_genre': 'category'
    })

    if 'track_id' not in df_processed.columns:
        print("Error: 'track_id' column not found after selecting columns.")
        return

    # Remove duplicates efficiently
    initial_rows = len(df_processed)
    df_processed.drop_duplicates(subset=['track_id'], inplace=True)
    rows_removed = initial_rows - len(df_processed)
    if rows_removed > 0:
        print(f"\nRemoved {rows_removed} duplicate track_id rows.")

    numeric_cols = [
        'popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
        'speechiness', 'acousticness', 'instrumentalness', 'liveness',
        'valence', 'tempo'
    ]

    missing_numeric = [col for col in numeric_cols if col not in df_processed.columns]
    if missing_numeric:
        print(f"Error: Numeric columns for scaling are missing: {missing_numeric}")
        return

    print("\nConverting numeric columns...")
    # Vectorized conversion of all numeric columns
    df_processed[numeric_cols] = df_processed[numeric_cols].apply(pd.to_numeric, errors='coerce')

    print("Handling missing numeric values (filling with mean)...")
    # More efficient handling of missing values
    for col in numeric_cols:
        if df_processed[col].isnull().any():
            mean_val = df_processed[col].mean()
            df_processed[col].fillna(mean_val, inplace=True)
            print(f" - Missing values in '{col}' filled with {mean_val:.4f}")

    # Drop rows with any remaining NaNs in numeric columns
    before_dropna = len(df_processed)
    df_processed.dropna(subset=numeric_cols, inplace=True)
    after_dropna = len(df_processed)
    if before_dropna != after_dropna:
        print(f"Dropped {before_dropna - after_dropna} rows with remaining NaN values")
    print(f"Remaining rows after handling numeric NaNs: {after_dropna}")

    print("\nScaling numeric features...")
    scaler = StandardScaler()
    df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    print("Features scaled.")

    print(f"\nSaving scaler to {scaler_path}...")
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    try:
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print("Scaler saved.")
    except Exception as e:
        print(f"Error saving scaler: {e}")

    print(f"\nSaving processed dataset to {output_file}...")
    try:
        df_processed.to_csv(output_file, index=False)
        print(f"Processed dataset saved with {len(df_processed)} unique rows and {len(df_processed.columns)} columns.")
        print(f"\nSample of processed data:")
        print(df_processed[['title', 'artist'] + numeric_cols[:3]].head())
    except Exception as e:
        print(f"Error saving processed dataset: {e}")

    print("\nData processing complete.")

if __name__ == "__main__":
    process_csv()
