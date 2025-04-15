import os
import pandas as pd
import numpy as np
from collections import Counter
import pickle

from AI.data_loader import load_dataset, load_scaler
from AI.embeddings import setup_embeddings, initialize_chromadb, load_embeddings_to_chromadb
from AI.embeddings import is_vector_search_available, get_chromadb_count, rebuild_chromadb
from AI.user_profiling import create_user_profile, recommend_tracks, recommend_tracks_with_chromadb, recommend_artists
from AI.visualization import generate_user_profile_chart

class MusicRecommender:
    def __init__(self, dataset_path, scaler_path, n_components=6, use_chromadb=True, chromadb_path=None, collection_name="music_tracks"):
        print("Initializing Music Recommender...")
        self.dataset, self.feature_cols = load_dataset(dataset_path)
        self.scaler = load_scaler(scaler_path)

        if self.dataset is None or self.scaler is None:
            raise ValueError("Failed to load dataset or scaler. Cannot initialize recommender.")

        self.n_components = n_components
        self.pca_model_path = os.path.join(os.path.dirname(scaler_path), 'pca_model.pkl')
        self.metadata_cols = ['artist', 'title', 'category', 'popularity']

        self.pca_model, self.embedding_col_names = setup_embeddings(
            self.dataset,
            self.scaler,
            self.feature_cols,
            self.n_components,
            self.pca_model_path
        )
        if self.pca_model is None:
            raise ValueError("PCA model setup failed. Recommendations cannot be generated.")

        self.chroma_client = None
        self.chroma_collection = None
        self.use_chromadb = use_chromadb
        self.chromadb_path = chromadb_path
        self.collection_name = collection_name

        if use_chromadb:
            self.initialize_chromadb_internal()

        print("Music Recommender initialized.")

    def initialize_chromadb_internal(self):
        if not self.use_chromadb or not self.chromadb_path:
            print("ChromaDB usage is disabled or path not provided.")
            self.chroma_client = None
            self.chroma_collection = None
            return False

        self.chroma_client, self.chroma_collection = initialize_chromadb(
            self.chromadb_path,
            self.collection_name,
            self.n_components
        )

        if self.chroma_collection is not None and self.get_chromadb_count() == 0 and not self.dataset.empty:
             print("ChromaDB collection is empty. Attempting to load embeddings...")
             self.load_embeddings_to_chromadb()

        return self.is_vector_search_available()

    def load_embeddings_to_chromadb(self):
        if not self.is_vector_search_available():
             print("Cannot load embeddings: ChromaDB is not available.")
             return False
        if self.pca_model is None or self.scaler is None:
             print("Cannot load embeddings: PCA model or Scaler is not available.")
             return False

        return load_embeddings_to_chromadb(
            self.chroma_collection,
            self.dataset,
            self.scaler,
            self.pca_model,
            self.feature_cols,
            self.metadata_cols
        )

    def is_vector_search_available(self):
        return is_vector_search_available(self.chroma_collection)

    def get_chromadb_count(self):
        return get_chromadb_count(self.chroma_collection)

    def rebuild_chromadb(self, load_after_rebuild=True):
        if not self.use_chromadb or not self.chroma_client:
            print("Cannot rebuild: ChromaDB usage is disabled or client not initialized.")
            return False

        new_collection = rebuild_chromadb(
            self.chroma_client,
            self.collection_name,
            self.n_components
        )
        self.chroma_collection = new_collection

        if self.chroma_collection is not None and load_after_rebuild:
             print("Rebuild complete. Loading embeddings into new collection...")
             return self.load_embeddings_to_chromadb()
        elif self.chroma_collection is not None:
             print("Rebuild complete. New collection is empty.")
             return True
        else:
             print("Rebuild failed.")
             return False

    def create_user_profile(self, user_tracks):
        if self.pca_model is None:
             print("Cannot create profile: PCA model not available.")
             return {
                 'feature_vector': np.zeros((1, self.n_components)),
                 'matched_tracks': pd.DataFrame(columns=['track_id', 'artist', 'title']),
                 'top_artists': [], 'top_categories': [], 'track_count': 0
             }

        return create_user_profile(
            self.dataset,
            self.pca_model,
            self.feature_cols,
            user_tracks
        )

    def recommend_tracks(self, user_profile, n=30, diversity_factor=0.3):
        if self.pca_model is None:
             print("Cannot recommend tracks (in-memory): PCA model not available.")
             return pd.DataFrame()

        return recommend_tracks(
            self.dataset,
            self.pca_model,
            self.feature_cols,
            user_profile,
            n,
            diversity_factor
        )

    def recommend_tracks_with_chromadb(self, user_profile, n=30, diversity_factor=0.3):
        return recommend_tracks_with_chromadb(
            self.dataset,
            self.pca_model,
            self.feature_cols,
            self.chroma_collection,
            user_profile,
            n,
            diversity_factor
        )

    def recommend_artists(self, user_profile, n=5):
        if self.pca_model is None:
             print("Cannot recommend artists: PCA model not available.")
             return []

        return recommend_artists(
            self.dataset,
            self.pca_model,
            self.feature_cols,
            user_profile,
            n
        )

    def generate_playlist(self, user_profile, name="Your Personalized Playlist", tracks=30, use_vector_search=True):
        if user_profile is None or user_profile.get('track_count', 0) == 0:
             print("Cannot generate playlist: Invalid user profile.")
             return {
                 "name": name, "tracks": [], "artists": [], "genres": [],
                 "track_count": 0, "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "metadata": {"vector_search_used": False, "recommendation_engine": "None - Empty Profile"}
             }

        recommended_tracks_df = pd.DataFrame()
        vector_search_actually_used = False

        if use_vector_search and self.is_vector_search_available():
            print("Generating playlist using ChromaDB...")
            recommended_tracks_df = self.recommend_tracks_with_chromadb(user_profile, n=tracks)
            vector_search_actually_used = True
            if recommended_tracks_df.empty and user_profile.get('track_count', 0) > 0:
                 print("ChromaDB search yielded no results or failed, trying in-memory...")
                 recommended_tracks_df = self.recommend_tracks(user_profile, n=tracks)
                 vector_search_actually_used = False
        else:
            print("Generating playlist using in-memory similarity...")
            recommended_tracks_df = self.recommend_tracks(user_profile, n=tracks)
            vector_search_actually_used = False

        if recommended_tracks_df.empty:
             print("No tracks could be recommended.")
             return {
                 "name": name, "tracks": [], "artists": [], "genres": [],
                 "track_count": 0, "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                 "metadata": {"vector_search_used": vector_search_actually_used, "recommendation_engine": "None - No Recommendations Found"}
             }

        recommended_artists = self.recommend_artists(user_profile, n=5)

        genre_distribution = Counter(recommended_tracks_df['category'].dropna())
        top_genres = [{"genre": genre, "count": count}
                      for genre, count in genre_distribution.most_common() if genre]

        playlist = {
            "name": name,
            "tracks": recommended_tracks_df.to_dict('records'),
            "artists": recommended_artists,
            "genres": top_genres,
            "track_count": len(recommended_tracks_df),
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            "metadata": {
                "vector_search_used": vector_search_actually_used,
                "recommendation_engine": "ChromaDB" if vector_search_actually_used else "In-memory similarity"
            }
        }

        return playlist

    def generate_user_profile_chart(self, user_profile, output_path=None):
        if self.pca_model is None:
             print("Cannot generate chart: PCA model not available.")
             return None
        if user_profile is None or user_profile.get('track_count', 0) == 0:
             print("Cannot generate chart: Invalid user profile.")
             return None

        return generate_user_profile_chart(
            user_profile,
            self.dataset,
            self.feature_cols,
            self.pca_model,
            output_path
        )