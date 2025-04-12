import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

def create_user_profile(dataset, embedding_cols, user_tracks):
    # Matching user tracks with dataset
    print("Matching user tracks with dataset...")
    matched_tracks = pd.DataFrame()
    matched_count = 0
    
    for _, user_track in user_tracks.iterrows():
        # Try matching by track_id
        track_match = dataset[dataset['track_id'] == user_track['id']]
        
        if len(track_match) > 0:
            matched_count += 1
            matched_tracks = pd.concat([matched_tracks, track_match], ignore_index=True)
    
    print(f"Matched {matched_count} out of {len(user_tracks)} user tracks")
    
    if matched_count == 0:
        print("Warning: No tracks matched! Unable to create user profile.")
        return {
            'feature_vector': np.zeros((1, len(embedding_cols))),
            'matched_tracks': pd.DataFrame(columns=['track_id', 'artist', 'title']),
            'top_artists': [],
            'top_categories': [],
            'track_count': 0
        }
    
    missing_cols = [col for col in embedding_cols if col not in matched_tracks.columns]
    if missing_cols:
        print(f"Warning: Missing embedding columns: {missing_cols}")
        print(f"Available columns: {matched_tracks.columns.tolist()}")
    
    # Calculate average feature vector from matched tracks
    feature_vector = matched_tracks[embedding_cols].mean().values.reshape(1, -1)
    
    # Getting top artists
    artist_counts = Counter(matched_tracks['artist'])
    top_artists = [artist for artist, _ in artist_counts.most_common(10)]
    
    top_categories = []
    if 'category' in matched_tracks.columns:
        category_counts = Counter(matched_tracks['category'])
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

def recommend_tracks(dataset, embedding_cols, user_profile, n=30, diversity_factor=0.3):
    if user_profile['track_count'] == 0:
        print("Warning: Empty user profile. Cannot recommend tracks.")
        return pd.DataFrame(columns=['track_id', 'artist', 'title', 'similarity'])
    
    print("Calculating track similarities...")
    
    # Calculate cosine similarity between user profile and all tracks
    user_vector = user_profile['feature_vector']
    all_tracks_matrix = dataset[embedding_cols].values
    
    similarities = cosine_similarity(user_vector, all_tracks_matrix)[0]
    
    recommendations = dataset.copy()
    recommendations['similarity'] = similarities
    
    # Exclude tracks that the user already knows
    known_tracks = set(user_profile['matched_tracks']['track_id'])
    recommendations = recommendations[~recommendations['track_id'].isin(known_tracks)]
    
    # Balance similarity with popularity for diversity
    recommendations['score'] = (1 - diversity_factor) * recommendations['similarity'] + diversity_factor * (recommendations['popularity'] / 100)
    
    # Boost tracks from user's favorite categories
    if user_profile['top_categories'] and 'category' in recommendations.columns:
        category_boost = 0.2
        is_preferred = recommendations['category'].isin(user_profile['top_categories'])
        recommendations.loc[is_preferred, 'score'] += category_boost
    
    # Select top tracks by combined score
    recommendations = recommendations.sort_values('score', ascending=False).head(n)
    
    # Normalize similarity scores
    max_similarity = recommendations['similarity'].max()
    if max_similarity > 0:
        recommendations['similarity'] = recommendations['similarity'] / max_similarity
    
    return recommendations[['track_id', 'artist', 'title', 'category', 'similarity', 'score']]

# Recommending tracks using ChromaDB vector search
def recommend_tracks_with_chromadb(dataset, embedding_cols, chroma_collection, user_profile, n=30, diversity_factor=0.3):
    if chroma_collection is None:
        print("ChromaDB not available, falling back to in-memory search")
        return recommend_tracks(dataset, embedding_cols, user_profile, n, diversity_factor)
    
    if user_profile['track_count'] == 0:
        print("Warning: Empty user profile. Cannot recommend tracks.")
        return pd.DataFrame(columns=['track_id', 'artist', 'title', 'category', 'similarity', 'score'])
    
    print("Generating recommendations using ChromaDB vector search...")
    
    user_vector = user_profile['feature_vector'][0].tolist()
    
    known_track_ids = list(user_profile['matched_tracks']['track_id'])
    
    results = chroma_collection.query(
        query_embeddings=[user_vector],
        n_results=n * 3,  # Get more results to allow filtering
        include=["metadatas", "distances"]
    )
    
    # Process results
    if not results or not results['ids'] or not results['ids'][0]:
        print("No search results from ChromaDB")
        return pd.DataFrame(columns=['track_id', 'artist', 'title', 'category', 'similarity', 'score'])
    
    recommendations = []
    
    for i, (metadata, distance) in enumerate(zip(results['metadatas'][0], results['distances'][0])):
        if distance <= 2.0:
            similarity = 1.0 - (distance / 2.0)
        else:  
            similarity = np.exp(-distance / 2.0)  # Convert distance to similarity
        
        recommendations.append({
            'track_id': metadata['track_id'],
            'artist': metadata['artist'],
            'title': metadata['title'],
            'category': metadata.get('category', ''),
            'similarity': similarity,
            'popularity': metadata.get('popularity', 50.0)
        })
    
    recommendation_df = pd.DataFrame(recommendations)
    
    # Filter out known tracks
    recommendation_df = recommendation_df[~recommendation_df['track_id'].isin(known_track_ids)]
    
    # Blend similarity with popularity for better recommendations
    recommendation_df['score'] = (
        (1 - diversity_factor) * recommendation_df['similarity'] + 
        diversity_factor * (recommendation_df['popularity'] / 100)  # Normalize popularity
    )
    
    # Boost tracks from user's favorite categories
    if user_profile['top_categories']:
        category_boost = 0.2
        is_preferred = recommendation_df['category'].isin(user_profile['top_categories'])
        recommendation_df.loc[is_preferred, 'score'] += category_boost
    
    recommendation_df = recommendation_df.sort_values('score', ascending=False)
    recommendation_df = recommendation_df.drop_duplicates(subset=['track_id'], keep='first')
    
    # Selecting top recommendations
    recommendations = recommendation_df.head(n)

    if not recommendations.empty:
        # Normalize similarity scores
        max_similarity = recommendations['similarity'].max()
        if max_similarity > 0:
            recommendations.loc[:, 'similarity'] = recommendations['similarity'] / max_similarity
    
    return recommendations[['track_id', 'artist', 'title', 'category', 'similarity', 'score']]

def recommend_artists(dataset, embedding_cols, user_profile, n=5):
    if user_profile['track_count'] == 0:
        print("Warning: Empty user profile. Cannot recommend artists.")
        return []
    
    # Getting known artists to exclude
    known_artists = set(user_profile['top_artists'])
    
    track_recommendations = recommend_tracks(dataset, embedding_cols, user_profile, n=100)
    
    artist_counts = Counter(track_recommendations['artist'])
    
    recommended_artists = []
    for artist, count in artist_counts.most_common(n*2): 
        if artist not in known_artists and len(recommended_artists) < n:
            # Calculate average similarity for this artist
            artist_tracks = track_recommendations[track_recommendations['artist'] == artist]
            avg_similarity = artist_tracks['similarity'].mean()
            
            recommended_artists.append({
                'artist': artist,
                'score': float(avg_similarity),
                'track_count': int(count)
            })
    
    return recommended_artists[:n]  