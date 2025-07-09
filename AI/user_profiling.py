import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA  # Import PCA for type hinting


def create_user_profile(dataset: pd.DataFrame, pca_model: PCA, feature_cols: list, user_tracks: pd.DataFrame):
    # Matching user tracks with dataset
    print("Matching user tracks with dataset...")

    # Ensuring feature_cols exist in the main dataset
    missing_dataset_cols = [col for col in feature_cols if col not in dataset.columns]
    if missing_dataset_cols:
        print(f"Error: Main dataset is missing required feature columns: {missing_dataset_cols}")
        # Return an empty profile matching the expected structure but with zero vector
        n_components = pca_model.n_components_ if pca_model else 0
        return {
            'feature_vector': np.zeros((1, n_components)),
            'matched_tracks': pd.DataFrame(columns=['track_id', 'artist', 'title']),
            'top_artists': [],
            'top_categories': [],
            'track_count': 0
        }

    # More efficient matching using merge instead of iterating
    user_track_ids = set(user_tracks['id'].values)
    matched_tracks = dataset[dataset['track_id'].isin(user_track_ids)].copy()

    if not matched_tracks.empty:
        cols_to_keep = list(set(feature_cols + ['track_id', 'artist', 'title', 'category']))
        cols_present = [col for col in cols_to_keep if col in matched_tracks.columns]
        matched_tracks = matched_tracks[cols_present]

    matched_count = len(matched_tracks)

    print(f"Matched {matched_count} out of {len(user_tracks)} user tracks")

    n_components = pca_model.n_components_ if pca_model else 0
    if matched_count == 0 or pca_model is None:
        print("Warning: No tracks matched or PCA model missing! Unable to create user profile vector.")
        return {
            'feature_vector': np.zeros((1, n_components)), 
            'matched_tracks': pd.DataFrame(columns=['track_id', 'artist', 'title']),
            'top_artists': [],
            'top_categories': [],
            'track_count': 0
        }

    print("Generating embeddings for matched tracks...")
    # Extract features
    matched_features = matched_tracks[feature_cols].values
    if np.isnan(matched_features).any():
        print("Warning: NaNs found in matched track features before PCA transform. Imputing with 0.")
        matched_features = np.nan_to_num(matched_features, nan=0.0)

    # Apply PCA transform
    try:
        matched_embeddings = pca_model.transform(matched_features)
    except Exception as e:
        print(f"Error applying PCA transform to matched tracks: {e}")
        return { 
            'feature_vector': np.zeros((1, n_components)),
            'matched_tracks': pd.DataFrame(columns=['track_id', 'artist', 'title']),
            'top_artists': [],
            'top_categories': [],
            'track_count': 0
        }

    # Calculate user profile vector
    feature_vector = np.mean(matched_embeddings, axis=0).reshape(1, -1)
    print("User profile vector created.")

    # Getting top artists and categories efficiently
    top_artists = matched_tracks['artist'].value_counts().head(10).index.tolist() if 'artist' in matched_tracks.columns else []

    top_categories = []
    if 'category' in matched_tracks.columns:
        top_categories = matched_tracks['category'].dropna().value_counts().head(5).index.tolist()

    # Creating user profile
    user_profile = {
        'feature_vector': feature_vector,
        'matched_tracks': matched_tracks[['track_id', 'artist', 'title']],
        'top_artists': top_artists,
        'top_categories': top_categories,
        'track_count': matched_count
    }

    return user_profile
def recommend_tracks(dataset: pd.DataFrame, pca_model: PCA, feature_cols: list, user_profile: dict, n=30, diversity_factor=0.3):
    if user_profile['track_count'] == 0 or pca_model is None:
        print("Warning: Empty user profile or missing PCA model. Cannot recommend tracks.")
        return pd.DataFrame(columns=['track_id', 'artist', 'title', 'category', 'similarity', 'score'])

    print("Calculating track similarities (In-Memory - Optimized)...")

    # Pre-filter known tracks to avoid processing them
    known_tracks = set(user_profile['matched_tracks']['track_id'])
    candidate_tracks = dataset[~dataset['track_id'].isin(known_tracks)].copy()

    if candidate_tracks.empty:
        print("No new tracks to recommend after filtering known tracks.")
        return pd.DataFrame(columns=['track_id', 'artist', 'title', 'category', 'similarity', 'score'])

    # Generating Embeddings for candidate tracks only
    try:
        candidate_features = candidate_tracks[feature_cols].values
        # Handle NaN values efficiently
        if np.isnan(candidate_features).any():
            print("Warning: NaNs found in dataset features before PCA transform. Imputing with 0.")
            candidate_features = np.nan_to_num(candidate_features, nan=0.0)
        candidate_embeddings = pca_model.transform(candidate_features)
    except Exception as e:
        print(f"Error applying PCA transform to candidate tracks: {e}")
        return pd.DataFrame(columns=['track_id', 'artist', 'title', 'category', 'similarity', 'score'])

    # Calculate cosine similarity between user profile and candidate tracks
    user_vector = user_profile['feature_vector']
    similarities = cosine_similarity(user_vector, candidate_embeddings)[0]

    # Add similarity scores to candidates
    candidate_tracks = candidate_tracks.copy()  # Avoid SettingWithCopyWarning
    candidate_tracks['similarity'] = similarities

    # Calculate composite score efficiently
    if 'popularity' in candidate_tracks.columns:
        # Vectorized normalization
        popularity_norm = candidate_tracks['popularity'] / 100.0
        candidate_tracks['score'] = (1 - diversity_factor) * similarities + diversity_factor * popularity_norm
    else:
        print("Warning: 'popularity' column not found for diversity calculation. Using similarity only.")
        candidate_tracks['score'] = similarities

    # Apply category boost efficiently
    if user_profile['top_categories'] and 'category' in candidate_tracks.columns:
        category_boost = 0.5
        is_preferred = candidate_tracks['category'].isin(user_profile['top_categories'])
        candidate_tracks.loc[is_preferred, 'score'] += category_boost

    # Sort and get top recommendations
    recommendations = candidate_tracks.nlargest(n, 'score')

    # Normalize similarity scores efficiently
    if not recommendations.empty and recommendations['similarity'].max() > recommendations['similarity'].min():
        min_sim = recommendations['similarity'].min()
        max_sim = recommendations['similarity'].max()
        recommendations = recommendations.copy()  # Avoid SettingWithCopyWarning
        recommendations['similarity'] = (recommendations['similarity'] - min_sim) / (max_sim - min_sim)
    elif not recommendations.empty:
        recommendations = recommendations.copy()
        recommendations['similarity'] = 1.0 if recommendations['similarity'].iloc[0] > 0 else 0.0

    return recommendations[['track_id', 'artist', 'title', 'category', 'similarity', 'score']]


def recommend_tracks_with_chromadb(dataset: pd.DataFrame, pca_model: PCA, feature_cols: list, chroma_collection, user_profile: dict, n=30, diversity_factor=0.3):
    if chroma_collection is None:
        print("ChromaDB not available, falling back to in-memory search")
        return recommend_tracks(dataset, pca_model, feature_cols, user_profile, n, diversity_factor)

    if user_profile['track_count'] == 0 or pca_model is None:
        print("Warning: Empty user profile or missing PCA model. Cannot recommend tracks.")
        return pd.DataFrame(columns=['track_id', 'artist', 'title', 'category', 'similarity', 'score'])

    print("Generating recommendations using ChromaDB vector search...")

    user_vector = user_profile['feature_vector'][0].tolist()
    known_track_ids = set(user_profile['matched_tracks']['track_id'])

    try:
        results = chroma_collection.query(
            query_embeddings=[user_vector],
            n_results=n * 3,  # Get more results to filter known tracks
            include=["metadatas", "distances"]
        )
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        print("Falling back to in-memory search...")
        return recommend_tracks(dataset, pca_model, feature_cols, user_profile, n, diversity_factor)

    # Process results efficiently
    if not results or not results.get('ids') or not results['ids'][0]:
        print("No search results from ChromaDB")
        return pd.DataFrame(columns=['track_id', 'artist', 'title', 'category', 'similarity', 'score'])

    # Prepare data for DataFrame creation
    recommendations_data = []
    seen_ids = set()

    for i, track_id in enumerate(results['ids'][0]):
        if track_id in seen_ids or track_id in known_track_ids:
            continue

        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]

        # Clamp distance and convert to similarity
        clamped_distance = max(0.0, min(distance, 2.0))
        similarity = 1.0 - (clamped_distance / 2.0)

        recommendations_data.append({
            'track_id': metadata.get('track_id', track_id),
            'artist': metadata.get('artist', 'Unknown Artist'),
            'title': metadata.get('title', 'Unknown Title'),
            'category': metadata.get('category', ''),
            'similarity': similarity,
            'popularity': metadata.get('popularity', 0.0)
        })
        seen_ids.add(track_id)

        if len(recommendations_data) >= n * 2:  # Stop when we have enough candidates
            break

    if not recommendations_data:
        print("No valid recommendations after processing ChromaDB results.")
        return pd.DataFrame(columns=['track_id', 'artist', 'title', 'category', 'similarity', 'score'])

    # Create DataFrame and process efficiently
    recommendation_df = pd.DataFrame(recommendations_data)

    # Vectorized operations for better performance
    recommendation_df['popularity'] = pd.to_numeric(recommendation_df['popularity'], errors='coerce').fillna(0.0)
    max_pop = recommendation_df['popularity'].max()
    if max_pop > 0:
        popularity_norm = recommendation_df['popularity'] / (100.0 if max_pop <= 100 else max_pop)
    else:
        popularity_norm = 0.0

    # Calculate composite score
    recommendation_df['score'] = (
        (1 - diversity_factor) * recommendation_df['similarity'] +
        diversity_factor * popularity_norm
    )

    # Apply category boost efficiently
    if user_profile['top_categories']:
        category_boost = 0.2
        is_preferred = recommendation_df['category'].isin(user_profile['top_categories'])
        recommendation_df.loc[is_preferred, 'score'] += category_boost

    # Get top N recommendations
    recommendations_final = recommendation_df.nlargest(n, 'score')

    # Normalize similarity scores efficiently
    if not recommendations_final.empty and len(recommendations_final) > 1:
        min_sim = recommendations_final['similarity'].min()
        max_sim = recommendations_final['similarity'].max()
        if max_sim > min_sim:
            recommendations_final = recommendations_final.copy()
            recommendations_final['similarity'] = (recommendations_final['similarity'] - min_sim) / (max_sim - min_sim)
        elif max_sim > 0:
            recommendations_final = recommendations_final.copy()
            recommendations_final['similarity'] = 1.0
        else:
            recommendations_final = recommendations_final.copy()
            recommendations_final['similarity'] = 0.0

    return recommendations_final[['track_id', 'artist', 'title', 'category', 'similarity', 'score']]


def recommend_artists(dataset: pd.DataFrame, pca_model: PCA, feature_cols: list, user_profile: dict, n=5):
    if user_profile['track_count'] == 0 or pca_model is None:
        print("Warning: Empty user profile or missing PCA model. Cannot recommend artists.")
        return []

    # Getting known artists to exclude efficiently
    known_artists = set(user_profile['top_artists'])

    # Get track recommendations more efficiently (fewer tracks needed for artist recommendations)
    track_recommendations = recommend_tracks(dataset, pca_model, feature_cols, user_profile, n=50)

    if track_recommendations.empty:
        print("No track recommendations found to base artist recommendations on.")
        return []

    # Use pandas groupby for more efficient aggregation
    artist_stats = track_recommendations.groupby('artist').agg({
        'similarity': ['mean', 'count']
    }).round(6)

    # Flatten column names
    artist_stats.columns = ['avg_similarity', 'track_count']
    artist_stats = artist_stats.reset_index()

    # Filter out known artists and sort by average similarity
    artist_stats = artist_stats[~artist_stats['artist'].isin(known_artists)]
    artist_stats = artist_stats.sort_values('avg_similarity', ascending=False)

    # Convert to required format
    recommended_artists = [
        {
            'artist': row['artist'],
            'score': float(row['avg_similarity']),
            'track_count': int(row['track_count'])
        }
        for _, row in artist_stats.head(n).iterrows()
    ]

    return recommended_artists

