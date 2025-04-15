import os
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler # Added for type hinting
import chromadb
import time
import sys
import traceback

# PCA embedding setup
def setup_embeddings(dataset: pd.DataFrame, scaler: StandardScaler, feature_cols: list, n_components: int, pca_model_path: str):
    print("Setting up PCA model...")
    pca_model = None
    embedding_col_names = [f'pca_{i}' for i in range(n_components)]

    # Verify necessary components
    if scaler is None:
        print("Error: Scaler object is required for PCA setup.")
        return None, embedding_col_names
    if not feature_cols:
        print("Error: feature_cols list is required for PCA setup.")
        return None, embedding_col_names
    missing_cols = [col for col in feature_cols if col not in dataset.columns]
    if missing_cols:
         print(f"Error: Dataset missing required feature columns for PCA setup: {missing_cols}")
         return None, embedding_col_names

    try:
        # Applying scaler to features
        print("Applying scaler to features for PCA setup...")
        X_features = dataset[feature_cols]
        if X_features.isna().any().any():
            print("Warning: NaNs found in features before scaling. Imputing with 0.")
            X_features = X_features.fillna(0.0)
        X_scaled = scaler.transform(X_features)
        print("Scaler applied.")

        # Load or Train PCA
        if os.path.exists(pca_model_path):
            print(f"Loading existing PCA model from {pca_model_path}")
            with open(pca_model_path, 'rb') as f:
                pca_model = pickle.load(f)
            if pca_model.n_components_ != n_components:
                print(f"Warning: Loaded PCA model has {pca_model.n_components_} components, but {n_components} were requested. Retraining...")
                pca_model = None
            else:
                print(f"PCA model loaded successfully with {pca_model.n_components_} components.")

        if pca_model is None:
            print(f"Training new PCA model with {n_components} components...")
            pca_model = PCA(n_components=n_components)
            pca_model.fit(X_scaled)
            print(f"PCA model trained. Explained variance: {sum(pca_model.explained_variance_ratio_):.4f}")

            # Save PCA model
            print(f"Saving PCA model to {pca_model_path}...")
            os.makedirs(os.path.dirname(pca_model_path), exist_ok=True)
            with open(pca_model_path, 'wb') as f:
                pickle.dump(pca_model, f)
            print("PCA model saved.")

        return pca_model, embedding_col_names

    except Exception as e:
        print(f"An error occurred during PCA setup: {e}")
        traceback.print_exc()
        return None, embedding_col_names


def initialize_chromadb(chromadb_path: str, collection_name: str, n_components: int):
    chroma_client = None
    chroma_collection = None
    try:
        print("Initializing ChromaDB...")
        if not chromadb_path:
             print("Error: ChromaDB path is required.")
             return None, None

        print(f"ChromaDB path: {chromadb_path}")
        os.makedirs(chromadb_path, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=chromadb_path)

        collection_metadata = {"hnsw:space": "cosine"}

        try:
            chroma_collection = chroma_client.get_collection(collection_name)
            print(f"Found existing ChromaDB collection '{collection_name}' with {chroma_collection.count()} items")
        except Exception:
            print(f"Creating new ChromaDB collection '{collection_name}'...")
            chroma_collection = chroma_client.create_collection(
                name=collection_name,
                metadata=collection_metadata
            )
            print(f"Collection created with dimension {n_components} and cosine distance.")

        return chroma_client, chroma_collection

    except Exception as e:
        print(f"Error initializing ChromaDB: {str(e)}")
        traceback.print_exc()
        print("Continuing without vector database...")
        return None, None


def load_embeddings_to_chromadb(chroma_collection, dataset: pd.DataFrame, scaler: StandardScaler, pca_model: PCA, feature_cols: list, metadata_cols: list):
    if chroma_collection is None:
        print("ChromaDB collection not initialized, skipping load.")
        return False
    if scaler is None or pca_model is None:
        print("Error: Scaler and PCA model are required for loading embeddings.")
        return False
    if not feature_cols or not metadata_cols:
        print("Error: feature_cols and metadata_cols are required.")
        return False

    required_load_cols = feature_cols + metadata_cols + ['track_id']
    missing_ds_cols = [col for col in required_load_cols if col not in dataset.columns]
    if missing_ds_cols:
        print(f"Error: Dataset missing required columns for ChromaDB load: {missing_ds_cols}")
        return False

    print("Loading music embeddings to ChromaDB...")

    # Prepare dataframe subset for loading
    tracks_df = dataset.dropna(subset=feature_cols + ['track_id']).copy()
    # Ensure metadata columns exist and fill NaNs appropriately
    for col in metadata_cols:
        if col not in tracks_df.columns:
            tracks_df[col] = "" # Add if missing
        elif pd.api.types.is_numeric_dtype(tracks_df[col]):
             tracks_df[col] = tracks_df[col].fillna(0) # Fill numeric NaNs with 0
        else:
             tracks_df[col] = tracks_df[col].fillna("") # Fill other NaNs with empty string

    if tracks_df.empty:
        print("No valid tracks found in the dataset to load into ChromaDB.")
        return False

    start_time = time.time()
    batch_size = 500
    total_batches = (len(tracks_df) + batch_size - 1) // batch_size
    loaded_count = 0

    print(f"Preparing to load {len(tracks_df)} tracks in {total_batches} batches...")

    for i in range(total_batches):
        try:
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(tracks_df))
            batch = tracks_df.iloc[start_idx:end_idx]

            if batch.empty:
                continue

            # Generating Embeddings for the batch
            X_batch_features = batch[feature_cols]
            if X_batch_features.isna().any().any():
                print(f"Warning: NaNs found in batch {i+1} features before scaling. Imputing with 0.")
                X_batch_features = X_batch_features.fillna(0.0)
            X_batch_scaled = scaler.transform(X_batch_features)
            embeddings_array = pca_model.transform(X_batch_scaled)

            # Preparing IDs and Metadatas
            ids = batch['track_id'].astype(str).tolist()
            embeddings_list = embeddings_array.tolist()
            metadatas = batch[metadata_cols].to_dict('records')

            if len(ids) != len(embeddings_list) or len(ids) != len(metadatas):
                 print(f"Error in batch {i+1}: Length mismatch between IDs, embeddings, and metadata. Skipping batch.")
                 continue

            # Adding data to ChromaDB
            chroma_collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas
            )
            loaded_count += len(ids)
            if (i + 1) % 10 == 0 or (i + 1) == total_batches: 
                 print(f"Loaded batch {i+1}/{total_batches} ({len(ids)} tracks)")

        except Exception as e:
            print(f"Error loading batch {i+1}/{total_batches}: {e}")
            traceback.print_exc()

    elapsed_time = time.time() - start_time
    print(f"Successfully loaded {loaded_count} tracks to ChromaDB in {elapsed_time:.2f} seconds")
    return loaded_count > 0


def is_vector_search_available(chroma_collection):
    return chroma_collection is not None

def get_chromadb_count(chroma_collection):
    if chroma_collection is not None:
        try:
            return chroma_collection.count()
        except Exception as e:
             print(f"Error getting ChromaDB count: {e}")
             return 0
    return 0

def rebuild_chromadb(chroma_client, collection_name: str, n_components: int):
    try:
        if chroma_client:
            print(f"Attempting to delete existing collection '{collection_name}'...")
            try:
                chroma_client.delete_collection(collection_name)
                print("Deleted existing collection.")
            except Exception as e:
                print(f"Collection '{collection_name}' might not exist or error deleting: {str(e)}")

            collection_metadata = {"hnsw:space": "cosine"}

            print(f"Creating new collection '{collection_name}'...")
            chroma_collection = chroma_client.create_collection(
                name=collection_name,
                metadata=collection_metadata
            )
            print(f"Collection created with dimension {n_components} and cosine distance.")
            return chroma_collection # Return the new, empty collection
        else:
            print("Error: Chroma client not available for rebuild.")
            return None
    except Exception as e:
        print(f"Error rebuilding ChromaDB: {str(e)}")
        traceback.print_exc()
        return None