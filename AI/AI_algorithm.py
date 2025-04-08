import matplotlib
matplotlib.use('Agg') 
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from collections import Counter

class MusicRecommender:
    def __init__(self, dataset_path, scaler_path, use_pca=True, n_components=6):
        print(f"Loading dataset from {dataset_path}...")
        self.dataset = pd.read_csv(dataset_path)
        
        print(f"Loading scaler from {scaler_path}...")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        self.feature_cols = ['popularity', 'danceability', 'energy', 'loudness', 
                           'acousticness', 'instrumentalness', 'liveness', 
                           'valence', 'tempo']
        self.scaled_feature_cols = [f"{col}_scaled" for col in self.feature_cols]
        
        # Check for missing values in scaled features
        missing_counts = self.dataset[self.scaled_feature_cols].isna().sum()
        if missing_counts.sum() > 0:
            print(f"Found {missing_counts.sum()} missing values in scaled features")
            print(missing_counts[missing_counts > 0])
            
            original_count = len(self.dataset)
            self.dataset = self.dataset.dropna(subset=self.scaled_feature_cols)
            print(f"Removed {original_count - len(self.dataset)} rows with missing values")
            
        print(f"Dataset loaded with {len(self.dataset)} tracks")
        
        # PCA Embeddings Implementation
        if use_pca:
            # Check if PCA embeddings already exist
            pca_cols = [f'pca_emb_{i}' for i in range(n_components)]
            if all(col in self.dataset.columns for col in pca_cols):
                print(f"Using existing PCA embeddings")
                self.embedding_cols = pca_cols
                
                # Load existing PCA model
                pca_path = os.path.join(os.path.dirname(scaler_path), 'pca_model.pkl')
                if os.path.exists(pca_path):
                    with open(pca_path, 'rb') as f:
                        self.pca = pickle.load(f)
                        print("Loaded existing PCA model")
            else:
                # Create new PCA embeddings
                self.create_pca_embeddings(n_components)
        else:
            # Use original features
            print("Using original scaled features as embeddings")
            self.embedding_cols = self.scaled_feature_cols
    
    # Plotting the cumulative explained variance of PCA components
    def plot_pca_variance(self, pca, max_components=None):
        if max_components is None:
            max_components = len(pca.explained_variance_ratio_)
        else:
            max_components = min(max_components, len(pca.explained_variance_ratio_))
            
        # Calculate cumulative explained variance
        explained_variance = pca.explained_variance_ratio_[:max_components]
        cumulative_variance = np.cumsum(explained_variance)
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot individual and cumulative explained variance
        components = range(1, max_components + 1)
        plt.bar(components, explained_variance, alpha=0.5, label='Individual explained variance')
        plt.step(components, cumulative_variance, where='mid', label='Cumulative explained variance')
        plt.scatter(components, cumulative_variance, s=50)
        
        # Add reference lines for common thresholds
        plt.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% threshold')
        plt.axhline(y=0.9, color='g', linestyle='--', alpha=0.5, label='90% threshold')
        plt.axhline(y=0.95, color='b', linestyle='--', alpha=0.5, label='95% threshold')
        
        # Format plot
        plt.title('Explained Variance by PCA Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Explained Variance Ratio')
        plt.xticks(components)
        plt.ylim([0, 1.05])
        plt.grid(True)
        plt.legend(loc='best')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '/../pca_variance.png')
        plt.close()
        print(f"PCA variance plot saved to pca_variance.png")

    # Createing PCA embeddings from audio features
    def create_pca_embeddings(self, n_components=6):
        print("Creating new PCA embeddings...")
        
        # Get scaled features
        features = self.dataset[self.scaled_feature_cols].values
        
        # Primeiro, vamos criar um PCA com mais componentes para análise
        pca_analysis = PCA()
        pca_analysis.fit(features)
        
        # Create and train PCA model
        pca = PCA(n_components=n_components)
        embeddings = pca.fit_transform(features)
        
        # Save embeddings to dataset
        for i in range(n_components):
            self.dataset[f'pca_emb_{i}'] = embeddings[:, i]
        
        # Update embedding columns
        self.embedding_cols = [f'pca_emb_{i}' for i in range(n_components)]
        
        # Display variance explained
        explained_variance = pca.explained_variance_ratio_
        print(f"Variance explained by components: {explained_variance}")
        print(f"Total variance explained: {sum(explained_variance):.2f}")
        
        # Visualize components
        self.visualize_pca_components(pca)
        
        # Save PCA model
        os.makedirs(os.path.dirname(os.path.abspath(__file__)) + '/../model', exist_ok=True)
        pca_path = os.path.dirname(os.path.abspath(__file__)) + '/../model/pca_model.pkl'
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)
        
        self.pca = pca
        return embeddings
    
    # Visualizing PCA components and their relationship to original feature
    def visualize_pca_components(self, pca):
        plt.figure(figsize=(12, 8))
        components = pd.DataFrame(
            pca.components_, 
            columns=self.scaled_feature_cols,
            index=[f'Component {i+1}' for i in range(pca.n_components_)]
        )
        
        sns.heatmap(components, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('PCA Components')
        plt.tight_layout()
        
        plt.savefig(os.path.dirname(os.path.abspath(__file__)) + '/../pca_components.png')
        plt.close()
        print(f"PCA components visualization saved to pca_components.png")
    
    def create_user_profile(self, user_tracks_df):
        # Find user tracks that exist in our dataset
        matched_tracks = self.dataset[self.dataset['track_id'].isin(user_tracks_df['id'])]
        
        if len(matched_tracks) == 0:
            print("No tracks from user history found in dataset")
            matched_tracks = self.dataset.sort_values('popularity', ascending=False).head(10)
        else:
            print(f"Found {len(matched_tracks)} tracks in dataset that match user's history")
        
        # Ensure we don't have any NaN values in the matched tracks
        matched_tracks = matched_tracks.dropna(subset=self.embedding_cols)
        if len(matched_tracks) == 0:
            print("All matched tracks had missing values, using popular tracks instead")
            matched_tracks = self.dataset.sort_values('popularity', ascending=False).head(10)
            matched_tracks = matched_tracks.dropna(subset=self.embedding_cols)
        
        # Calculate average feature vector (user's taste profile)
        user_mean_features = matched_tracks[self.embedding_cols].mean().values.reshape(1, -1)
        
        # Verify the user vector doesn't contain NaN values
        if np.isnan(user_mean_features).any():
            print("User profile contains NaN values, using fallback profile")
            # Fallback to a safe profile
            safe_tracks = self.dataset.dropna(subset=self.embedding_cols).head(100)
            user_mean_features = safe_tracks[self.embedding_cols].mean().values.reshape(1, -1)
        
        # Identify user's preferred genres
        category_counts = Counter(matched_tracks['category'])
        top_categories = [cat for cat, _ in category_counts.most_common(3)]
        
        user_profile = {
            'feature_vector': user_mean_features,
            'matched_tracks': matched_tracks,
            'top_categories': top_categories,
            'matched_count': len(matched_tracks)
        }
        
        return user_profile
    
    def recommend_tracks(self, user_profile, n=30, diversity_factor=0.3):
        # Get user's feature vector
        user_vector = user_profile['feature_vector']
        
        # Ensure the dataset has no NaN values
        valid_tracks = self.dataset.dropna(subset=self.embedding_cols)
        
        # Calculate similarity between user profile and valid tracks
        all_track_features = np.array(valid_tracks[self.embedding_cols])
        similarities = cosine_similarity(user_vector, all_track_features)[0]
        
        # Add similarity scores to the valid tracks
        recommendation_df = valid_tracks.copy()
        recommendation_df['similarity'] = similarities
        
        # Exclude tracks the user already knows
        known_tracks = set(user_profile['matched_tracks']['track_id'])
        recommendation_df = recommendation_df[~recommendation_df['track_id'].isin(known_tracks)]
        
        # Blend similarity with popularity for better recommendations
        recommendation_df['score'] = (
            (1 - diversity_factor) * recommendation_df['similarity'] + 
            diversity_factor * recommendation_df['popularity_scaled']
        )
        
        # Boost tracks from user's favorite categories
        if user_profile['top_categories']:
            category_boost = 0.5
            is_preferred = recommendation_df['category'].isin(user_profile['top_categories'])
            recommendation_df.loc[is_preferred, 'score'] += category_boost
        
        # Remove duplicate tracks (keeping only the highest scored instance)
        recommendation_df = recommendation_df.sort_values('score', ascending=False)
        recommendation_df = recommendation_df.drop_duplicates(subset=['track_id'], keep='first')
        
        # Select top recommendations
        recommendations = recommendation_df.head(n)

        if not recommendations.empty:
            # Normalize similarity scores
            max_similarity = recommendations['similarity'].max()
            recommendations.loc[:, 'similarity'] = recommendations['similarity'] / max_similarity
        return recommendations[['track_id', 'artist', 'title', 'category', 
                               'similarity', 'score']]

    def recommend_artists(self, user_profile, n=5):
        track_recommendations = self.recommend_tracks(user_profile, n=100)
        # Count artist occurrences in recommendations, weighted by similarity
        artist_scores = {}
        
        for _, track in track_recommendations.iterrows():
            artist = track['artist']
            
            # Handle multiple artists (semicolon-separated)
            for single_artist in artist.split(';'):
                single_artist = single_artist.strip()
                
                if single_artist in artist_scores:
                    artist_scores[single_artist] += track['score']
                else:
                    artist_scores[single_artist] = track['score']
        
        # Get user's existing artists to exclude
        known_artists = set()
        for _, track in user_profile['matched_tracks'].iterrows():
            for artist in track['artist'].split(';'):
                known_artists.add(artist.strip())
        
        # Filter out known artists
        artist_scores = {a: s for a, s in artist_scores.items() if a not in known_artists}

        # Return top artists
        top_artists = sorted(artist_scores.items(), key=lambda x: x[1], reverse=True)[:n]
        # Get the maximum score for normalization
        if top_artists:
            max_score = top_artists[0][1]
            normalized_artists = [{"artist": artist, "score": score / max_score} 
                                for artist, score in top_artists]
            return normalized_artists
        else:
            return []
    
    def generate_playlist(self, user_profile, name="Your Personalized Playlist", tracks=30):
        recommended_tracks = self.recommend_tracks(user_profile, n=tracks)
        recommended_artists = self.recommend_artists(user_profile, n=5)
        
        # Get a genre distribution of recommended songs
        genre_distribution = Counter(recommended_tracks['category'])
        top_genres = [{"genre": genre, "count": count} 
                      for genre, count in genre_distribution.most_common()]
        
        # Build playlist object
        playlist = {
            "name": name,
            "tracks": recommended_tracks.to_dict('records'),
            "artists": recommended_artists,
            "genres": top_genres,
            "track_count": len(recommended_tracks),
            "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return playlist


def main():
    # Demo code for testing
    dataset_path = '/Users/rafatorres/Desktop/Shaco-AI/data/processed_dataset.csv'
    scaler_path = '/Users/rafatorres/Desktop/Shaco-AI/model/scaler_model.pkl'
    
    # Create recommender with PCA embeddings
    recommender = MusicRecommender(dataset_path, scaler_path, use_pca=True, n_components=6)
    
    # Simulate user data
    mock_user_data = pd.DataFrame({
        'id': [
            '6Vc5wAMmXdKIAM7WUoEb7N',  # Say Something - A Great Big World
            '1EzrEOXmMH3G43AXT1y7pA',  # I'm Yours - Jason Mraz
            '0IktbUcnAGrvD03AWnz3Q8'   # Lucky - Jason Mraz & Colbie Caillat
        ],
        'name': ['Say Something', "I'm Yours", 'Lucky'],
        'artist_name': ['A Great Big World', 'Jason Mraz', 'Jason Mraz']
    })
    
    print("\nCreating user profile...")
    user_profile = recommender.create_user_profile(mock_user_data)
    
    print("\nGenerating track recommendations...")
    track_recommendations = recommender.recommend_tracks(user_profile, n=10)
    print(track_recommendations[['artist', 'title', 'similarity']].head())
    
    print("\nGenerating artist recommendations...")
    artist_recommendations = recommender.recommend_artists(user_profile)
    for i, artist in enumerate(artist_recommendations):
        print(f"{i+1}. {artist['artist']} (score: {artist['score']:.3f})")
    
    print("\nGenerating complete playlist...")
    playlist = recommender.generate_playlist(user_profile)
    print(f"Playlist '{playlist['name']}' with {playlist['track_count']} tracks")
    print(f"Top genres: {', '.join([g['genre'] for g in playlist['genres'][:3]])}")
    
    import json
    with open('recommendation_results.json', 'w') as f:
        json.dump(playlist, f, indent=2)
    print("\n✅ Saved recommendations to recommendation_results.json")


if __name__ == "__main__":
    main()