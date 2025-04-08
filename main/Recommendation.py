import os
import pandas as pd
import json
from datetime import datetime
from .user_auth import SpotifyUserAuth
from AI.AI_algorithm import MusicRecommender

def main():
    print("🎷 Blonded AI Music Recommendation System") 
    # Paths
    dataset_path = '/Users/rafatorres/Desktop/Shaco-AI/data/processed_dataset.csv'
    scaler_path = '/Users/rafatorres/Desktop/Shaco-AI/model/scaler_model.pkl'
    output_dir = '/Users/rafatorres/Desktop/Shaco-AI/data/recommendations'
    
    # Authenticating with Spotify and collect user data
    print("\n1️⃣ Authenticating with Spotify...")
    auth = SpotifyUserAuth()
    
    if not auth.authenticate():
        print(" Authentication failed. Please check your credentials and try again.")
        return
        
    print("\n2️⃣ Collecting your music preferences from Spotify...")
    if not auth.collect_user_music_data():
        print(" Failed to collect user data.")
        return
        
    print("\n3️⃣ Processing your music profile...")
    tracks_file, user_file = auth.save_user_data()
    
    # Preparing user data for recommendation system
    user_tracks_df = pd.read_csv(tracks_file)
    
    # Formating the DF
    user_tracks = pd.DataFrame({
        'id': user_tracks_df['id'],
        'name': user_tracks_df['name'],
        'artist_name': user_tracks_df['artist_name']
    })
    
    # Runnng recommendation system
    print("\n4️⃣ Initializing recommendation...")
    try:
        recommender = MusicRecommender(dataset_path, scaler_path)
        
        print("\n5️⃣ Creating your personalized music profile...")
        user_profile = recommender.create_user_profile(user_tracks)
        
        print("\n6️⃣ Generating recommendations based on your taste...")
        playlist = recommender.generate_playlist(
            user_profile, 
            name=f"{auth.user_data['display_name']}'s Recommended Playlist",
            tracks=30
        )
        
        # Saving results (JSON)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{output_dir}/recommendations_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(playlist, f, indent=2)
            
        print("\n Recommendations generated successfully!")
        print(f"\n Your Personalized Playlist: {playlist['name']}")
        print(f"   • {len(playlist['tracks'])} tracks")
        print(f"   • Top genres: {', '.join([g['genre'] for g in playlist['genres'][:3]])}")
        
        print("\n Recommended Artists:")
        for i, artist in enumerate(playlist['artists'][:5]):
            print(f"   {i+1}. {artist['artist']}")
            
        print(f"\n Sample Tracks:")
        for i, track in enumerate(playlist['tracks'][:5]):
            print(f"   {i+1}. \"{track['title']}\" by {track['artist']}")
            
        print(f"\n Full recommendations saved to: {results_file}")
        
        export_response = input("\nWould you like to export this playlist to your Spotify account? (y/n): ")
        
        if export_response.lower() in ['y', 'yes']:
            print("\n Creating playlist on Spotify...")
            
            # Extracting track IDs from the recommendations
            track_ids = [track['track_id'] for track in playlist['tracks']]
            
            # Creating a description for the playlist
            description = f"Recommended by Blonded AI based on your listening history. Top genres: {', '.join([g['genre'] for g in playlist['genres'][:3]])}"
            
            # Create the playlist
            spotify_playlist = auth.create_spotify_playlist(
                playlist_name=playlist['name'],
                track_ids=track_ids,
                description=description
            )
            
            if spotify_playlist:
                print(f"\nPlaylist created successfully!")
                print(f"Open in Spotify: {spotify_playlist['url']}")
        
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        
if __name__ == "__main__":
    main()