import os
import pandas as pd
import numpy as np
from collections import Counter

from AI.data_loader import load_dataset, load_scaler
from AI.embeddings import setup_embeddings, initialize_chromadb, load_embeddings_to_chromadb
from AI.embeddings import is_vector_search_available, get_chromadb_count, rebuild_chromadb
from AI.user_profiling import create_user_profile, recommend_tracks, recommend_tracks_with_chromadb, recommend_artists
from AI.visualization import generate_user_profile_chart

class MusicRecommender:
    def __init__(self, dataset_path, scaler_path, n_components=6, use_chromadb=True, chromadb_path=None):
        # Load dataset and scaler
        self.dataset, self.feature_cols, self.scaled_feature_cols = load_dataset(dataset_path)
        self.scaler = load_scaler(scaler_path)
        
        # Setting up embeddings
        self.n_components = n_components
        self.embedding_cols, self.pca_model = setup_embeddings(
            self.dataset, 
            self.scaled_feature_cols,
            n_components, 
            scaler_path
        )
        
        # Initializing ChromaDB
        self.chroma_client = None
        self.chroma_collection = None
        self.use_chromadb = use_chromadb
        self.chromadb_path = chromadb_path
        
        if use_chromadb:
            self.chroma_client, self.chroma_collection = initialize_chromadb(
                self.dataset, 
                self.embedding_cols, 
                chromadb_path
            )
    
    def initialize_chromadb(self, chromadb_path=None):
        self.chroma_client, self.chroma_collection = initialize_chromadb(
            self.dataset, 
            self.embedding_cols, 
            chromadb_path
        )
        return self.chroma_collection is not None
        
    def load_embeddings_to_chromadb(self):
        return load_embeddings_to_chromadb(self.chroma_collection, self.dataset, self.embedding_cols)
    
    def is_vector_search_available(self):
        return is_vector_search_available(self.chroma_collection)
    
    def get_chromadb_count(self):
        return get_chromadb_count(self.chroma_collection)
    
    def rebuild_chromadb(self):
        self.chroma_client, self.chroma_collection = rebuild_chromadb(
            self.chroma_client, 
            self.dataset, 
            self.embedding_cols, 
            self.chromadb_path
        )
        return self.chroma_collection is not None
    
    def create_user_profile(self, user_tracks):
        return create_user_profile(self.dataset, self.embedding_cols, user_tracks)
    
    def recommend_tracks(self, user_profile, n=30, diversity_factor=0.3):
        return recommend_tracks(self.dataset, self.embedding_cols, user_profile, n, diversity_factor)
    
    def recommend_tracks_with_chromadb(self, user_profile, n=30, diversity_factor=0.3):
        return recommend_tracks_with_chromadb(
            self.dataset, 
            self.embedding_cols, 
            self.chroma_collection, 
            user_profile, 
            n, 
            diversity_factor
        )
    
    def recommend_artists(self, user_profile, n=5):
        return recommend_artists(self.dataset, self.embedding_cols, user_profile, n)
    
    def generate_playlist(self, user_profile, name="Your Personalized Playlist", tracks=30, use_vector_search=True):
        if use_vector_search and self.chroma_collection is not None:
            recommended_tracks = self.recommend_tracks_with_chromadb(user_profile, n=tracks)
        else:
            recommended_tracks = self.recommend_tracks(user_profile, n=tracks)
        
        recommended_artists = self.recommend_artists(user_profile, n=5)
        
        # Getting a genre distribution of recommended songs
        genre_distribution = Counter(recommended_tracks['category'])
        top_genres = [{"genre": genre, "count": count} 
                      for genre, count in genre_distribution.most_common()]
        
        # Building playlist object
        playlist = {
            "name": name,
            "tracks": recommended_tracks.to_dict('records'),
            "artists": recommended_artists,
            "genres": top_genres,
            "track_count": len(recommended_tracks),
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": {
                "vector_search_used": use_vector_search and self.chroma_collection is not None,
                "recommendation_engine": "ChromaDB" if (use_vector_search and self.chroma_collection is not None) else "In-memory similarity"
            }
        }
        
        return playlist
    
    def generate_user_profile_chart(self, user_profile, output_path=None):
        return generate_user_profile_chart(
            user_profile, 
            self.dataset, 
            self.feature_cols, 
            self.scaled_feature_cols,
            self.pca_model, 
            output_path
        )