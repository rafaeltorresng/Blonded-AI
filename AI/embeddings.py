import os
import numpy as np
import pandas as pd
import pickle
from sklearn.decomposition import PCA
import chromadb
import time

# PCA embeddings
def setup_embeddings(dataset, scaled_feature_cols, n_components=6, scaler_path=None):
    print("Using PCA embeddings...")
    
    pca_model_path = os.path.join(os.path.dirname(scaler_path), 'pca_model.pkl')
    if os.path.exists(pca_model_path):
        print(f"Loading PCA model from {pca_model_path}")
        with open(pca_model_path, 'rb') as f:
            pca_model = pickle.load(f)
    else:
        print("Training PCA model...")
        pca_model = PCA(n_components=n_components)
        X = dataset[scaled_feature_cols].values
        pca_model.fit(X)
        
        # Save PCA model
        with open(pca_model_path, 'wb') as f:
            pickle.dump(pca_model, f)
            
    # Apply PCA to get embeddings
    print("Generating embeddings...")
    X = dataset[scaled_feature_cols].values
    embeddings = pca_model.transform(X)
    
    # Add embeddings to dataset
    embedding_cols = [f"embedding_{i}" for i in range(n_components)]
    
    for i, col in enumerate(embedding_cols):
        dataset[col] = embeddings[:, i]
        
    return embedding_cols, pca_model

def initialize_chromadb(dataset, embedding_cols, chromadb_path=None):
    chroma_client = None
    chroma_collection = None
    
    try:
        print("Initializing ChromaDB...")
        
        if chromadb_path is None:
            import sys
            sys_path = sys.modules['AI'].__file__
            chromadb_path = os.path.join(os.path.dirname(os.path.dirname(sys_path)), "chroma_db")
            
        print(f"ChromaDB path: {chromadb_path}")
        
        os.makedirs(chromadb_path, exist_ok=True)
        
        chroma_client = chromadb.PersistentClient(path=chromadb_path)
        
        try:
            chroma_collection = chroma_client.get_collection("music_tracks")
            print(f"Found existing ChromaDB collection with {chroma_collection.count()} items")
            
            # If collection exists but is empty, reload data
            if chroma_collection.count() == 0:
                print("Collection exists but is empty. Loading data...")
                load_embeddings_to_chromadb(chroma_collection, dataset, embedding_cols)
                
        except Exception as e:
            print(f"Creating new ChromaDB collection: {str(e)}")
            chroma_collection = chroma_client.create_collection(
                name="music_tracks",
                metadata={"description": "Music track embeddings for recommendation"}
            )
            # Load data into the new collection
            load_embeddings_to_chromadb(chroma_collection, dataset, embedding_cols)
            
        return chroma_client, chroma_collection
        
    except Exception as e:
        print(f"Error initializing ChromaDB: {str(e)}")
        print("Continuing without vector database...")
        return None, None

def load_embeddings_to_chromadb(chroma_collection, dataset, embedding_cols):
    if chroma_collection is None:
        print("ChromaDB collection not initialized, skipping...")
        return False
    
    print("Loading music embeddings to ChromaDB...")
    
    tracks_df = dataset.dropna(subset=embedding_cols + ['track_id', 'artist', 'title'])
    
    if 'category' not in tracks_df.columns:
        tracks_df['category'] = ""
    
    # Preparing IDs, embeddings and metadata
    start_time = time.time()
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    total_batches = (len(tracks_df) + batch_size - 1) // batch_size
    
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(tracks_df))
        batch = tracks_df.iloc[start_idx:end_idx]
        
        ids = [str(idx) for idx in batch.index]
        embeddings = batch[embedding_cols].values.tolist()
        
        # Preparing metadata
        metadatas = []
        for _, row in batch.iterrows():
            metadata = {
                "track_id": row["track_id"],
                "artist": row["artist"],
                "title": row["title"],
                "category": row["category"] if row["category"] else "",
                "popularity": float(row["popularity"]) if "popularity" in row else 0.0
            }
            metadatas.append(metadata)
        
        # Add data to ChromaDB
        chroma_collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        print(f"Loaded batch {i+1}/{total_batches} ({len(ids)} tracks)")
    
    elapsed_time = time.time() - start_time
    print(f"Successfully loaded {len(tracks_df)} tracks to ChromaDB in {elapsed_time:.2f} seconds")
    return True

def is_vector_search_available(chroma_collection):
    return chroma_collection is not None
    
def get_chromadb_count(chroma_collection):
    if chroma_collection is not None:
        try:
            return chroma_collection.count()
        except:
            return 0
    return 0
    
def rebuild_chromadb(chroma_client, dataset, embedding_cols, chromadb_path):
    try:
        if chroma_client:
            try:
                # Delete existing collection
                chroma_client.delete_collection("music_tracks")
                print("Deleted existing collection")
            except Exception as e:
                print(f"Error deleting collection: {str(e)}")
            
            # Create new collection
            chroma_collection = chroma_client.create_collection(
                name="music_tracks",
                metadata={"description": "Music track embeddings for recommendation"}
            )
            
            # Load data into the new collection
            load_embeddings_to_chromadb(chroma_collection, dataset, embedding_cols)
            return chroma_client, chroma_collection
        else:
            # Initialize ChromaDB if it wasn't initialized before
            return initialize_chromadb(dataset, embedding_cols, chromadb_path)
    except Exception as e:
        print(f"Error rebuilding ChromaDB: {str(e)}")
        return None, None