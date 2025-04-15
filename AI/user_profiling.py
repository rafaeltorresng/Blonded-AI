import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA # Import PCA for type hinting

def create_user_profile(dataset: pd.DataFrame, pca_model: PCA, feature_cols: list, user_tracks: pd.DataFrame):
    # Matching user tracks with dataset
    print("Matching user tracks with dataset...")
    matched_tracks = pd.DataFrame()
    matched_count = 0

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

    for _, user_track in user_tracks.iterrows():
        # Try matching by track_id
        track_match = dataset[dataset['track_id'] == user_track['id']]

        if len(track_match) > 0:
            matched_count += 1
            cols_to_keep = list(set(feature_cols + ['track_id', 'artist', 'title', 'category']))
            cols_present = [col for col in cols_to_keep if col in track_match.columns]
            matched_tracks = pd.concat([matched_tracks, track_match[cols_present]], ignore_index=True)

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

    # Getting top artists
    artist_counts = Counter(matched_tracks['artist'])
    top_artists = [artist for artist, _ in artist_counts.most_common(10)]

    top_categories = []
    if 'category' in matched_tracks.columns:
        category_counts = Counter(matched_tracks['category'].dropna()) # Ensure NaNs are dropped before counting
        top_categories = [category for category, _ in category_counts.most_common(5) if category]

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

    print("Calculating track similarities (In-Memory - May be slow/memory intensive)...")

    # Generating Embeddings for all tracks
    try:
        all_features = dataset[feature_cols].values
        if np.isnan(all_features).any():
            print("Warning: NaNs found in dataset features before PCA transform. Imputing with 0.")
            all_features = np.nan_to_num(all_features, nan=0.0)
        all_tracks_embeddings = pca_model.transform(all_features)
    except Exception as e:
        print(f"Error applying PCA transform to all tracks: {e}")
        return pd.DataFrame(columns=['track_id', 'artist', 'title', 'category', 'similarity', 'score'])

    # Calculate cosine similarity between user profile and all tracks
    user_vector = user_profile['feature_vector']
    similarities = cosine_similarity(user_vector, all_tracks_embeddings)[0]

    recommendations = dataset.copy()
    recommendations['similarity'] = similarities

    known_tracks = set(user_profile['matched_tracks']['track_id'])
    recommendations = recommendations[~recommendations['track_id'].isin(known_tracks)]

    if 'popularity' in recommendations.columns:
         # Simple normalization (0-100 assumed)
         recommendations['popularity_norm'] = recommendations['popularity'] / 100.0
         recommendations['score'] = (1 - diversity_factor) * recommendations['similarity'] + diversity_factor * recommendations['popularity_norm']
    else:
         print("Warning: 'popularity' column not found for diversity calculation. Using similarity only.")
         recommendations['score'] = recommendations['similarity']


    if user_profile['top_categories'] and 'category' in recommendations.columns:
        category_boost = 0.5
        is_preferred = recommendations['category'].isin(user_profile['top_categories'])
        recommendations.loc[is_preferred, 'score'] += category_boost

    recommendations = recommendations.sort_values('score', ascending=False).head(n)

    # Normalize similarity scores (0 to 1) for final output
    if not recommendations.empty:
        min_sim = recommendations['similarity'].min()
        max_sim = recommendations['similarity'].max()
        if max_sim > min_sim: # Avoid division by zero if all similarities are the same
            recommendations['similarity'] = (recommendations['similarity'] - min_sim) / (max_sim - min_sim)
        elif max_sim != 0: # Handle case where all similarities are the same non-zero value
             recommendations['similarity'] = 1.0
        else: # All similarities are zero
             recommendations['similarity'] = 0.0


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

    known_track_ids = list(user_profile['matched_tracks']['track_id'])

    try:
        results = chroma_collection.query(
            query_embeddings=[user_vector],
            n_results=n * 3,  
            include=["metadatas", "distances"]
        )
    except Exception as e:
        print(f"Error querying ChromaDB: {e}")
        print("Falling back to in-memory search...")
        return recommend_tracks(dataset, pca_model, feature_cols, user_profile, n, diversity_factor)


    # Process results
    if not results or not results.get('ids') or not results['ids'][0]:
        print("No search results from ChromaDB")
        return pd.DataFrame(columns=['track_id', 'artist', 'title', 'category', 'similarity', 'score'])

    recommendations = []
    processed_ids = set() 

    for i, track_id in enumerate(results['ids'][0]):
        if track_id in processed_ids or track_id in known_track_ids:
            continue 

        metadata = results['metadatas'][0][i]
        distance = results['distances'][0][i]

        clamped_distance = max(0.0, min(distance, 2.0))
        similarity = 1.0 - (clamped_distance / 2.0)

        recommendations.append({
            'track_id': metadata.get('track_id', track_id), 
            'artist': metadata.get('artist', 'Unknown Artist'),
            'title': metadata.get('title', 'Unknown Title'),
            'category': metadata.get('category', ''),
            'similarity': similarity,
            'popularity': metadata.get('popularity', 0.0) 
        })
        processed_ids.add(track_id)


    if not recommendations:
         print("No valid recommendations after processing ChromaDB results.")
         return pd.DataFrame(columns=['track_id', 'artist', 'title', 'category', 'similarity', 'score'])

    recommendation_df = pd.DataFrame(recommendations)

    recommendation_df['popularity'] = pd.to_numeric(recommendation_df['popularity'], errors='coerce').fillna(0.0)
    max_pop = recommendation_df['popularity'].max()
    recommendation_df['popularity_norm'] = recommendation_df['popularity'] / 100.0 if max_pop <= 100 else recommendation_df['popularity'] / max_pop if max_pop > 0 else 0.0

    recommendation_df['score'] = (
        (1 - diversity_factor) * recommendation_df['similarity'] +
        diversity_factor * recommendation_df['popularity_norm']
    )

    if user_profile['top_categories']:
        category_boost = 0.2
        is_preferred = recommendation_df['category'].isin(user_profile['top_categories'])
        recommendation_df.loc[is_preferred, 'score'] += category_boost

    recommendation_df = recommendation_df.sort_values('score', ascending=False)

    # Selecting top recommendations
    recommendations_final = recommendation_df.head(n)

    # Normalize similarity scores (0 to 1) for final output
    if not recommendations_final.empty:
        min_sim = recommendations_final['similarity'].min()
        max_sim = recommendations_final['similarity'].max()
        if max_sim > min_sim:
            recommendations_final.loc[:, 'similarity'] = (recommendations_final['similarity'] - min_sim) / (max_sim - min_sim)
        elif max_sim != 0:
             recommendations_final.loc[:, 'similarity'] = 1.0
        else:
             recommendations_final.loc[:, 'similarity'] = 0.0


    return recommendations_final[['track_id', 'artist', 'title', 'category', 'similarity', 'score']]

def recommend_artists(dataset: pd.DataFrame, pca_model: PCA, feature_cols: list, user_profile: dict, n=5):
    if user_profile['track_count'] == 0 or pca_model is None:
        print("Warning: Empty user profile or missing PCA model. Cannot recommend artists.")
        return []

    # Getting known artists to exclude
    known_artists = set(user_profile['top_artists'])

    track_recommendations = recommend_tracks(dataset, pca_model, feature_cols, user_profile, n=100)

    if track_recommendations.empty:
        print("No track recommendations found to base artist recommendations on.")
        return []

    artist_counts = Counter(track_recommendations['artist'])

    recommended_artists = []
    for artist, count in artist_counts.most_common(n*2):
        if artist not in known_artists and len(recommended_artists) < n:
            artist_tracks = track_recommendations[track_recommendations['artist'] == artist]
            if not artist_tracks.empty:
                 avg_similarity = artist_tracks['similarity'].mean()
                 recommended_artists.append({
                     'artist': artist,
                     'score': float(avg_similarity),
                     'track_count': int(count)
                 })

    # Sort by score (average similarity)
    recommended_artists.sort(key=lambda x: x['score'], reverse=True)

    return recommended_artists[:n]
