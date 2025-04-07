import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
import pandas as pd
import json
from datetime import datetime

load_dotenv()

class SpotifyUserAuth:
    def __init__(self):
        self.client_id = os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
        self.redirect_uri = os.getenv("SPOTIFY_REDIRECT_URI")
        
        if not all([self.client_id, self.client_secret, self.redirect_uri]):
            raise ValueError("Missing Spotify API credentials in .env file")
        
        self.scope = "user-top-read user-read-recently-played user-library-read"
        
        self.sp = None
        self.user_data = {
            'user_id': None,
            'display_name': None,
            'top_tracks': [],
            'top_artists': []
        }

    def authenticate(self):
        try:
            auth_manager = SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope=self.scope,
                cache_path=".spotify_cache"
            )
            
            self.sp = spotipy.Spotify(auth_manager=auth_manager)
            
            # Getting basic user info
            user_info = self.sp.current_user()
            self.user_data['user_id'] = user_info['id']
            self.user_data['display_name'] = user_info['display_name']
            
            print(f"✅ Successfully authenticated as {self.user_data['display_name']}")
            return True
            
        except Exception as e:
            print(f"❌ Authentication failed: {str(e)}")
            return False

    def collect_user_music_data(self):
        if not self.sp:
            print("Not authenticated. Call authenticate() first.")
            return False
            
        try:
            # Collecting top tracks 
            time_ranges = ['short_term', 'medium_term', 'long_term']
            
            for time_range in time_ranges:
                results = self.sp.current_user_top_tracks(limit=50, time_range=time_range)
                
                for item in results['items']:
                    track_data = {
                        'id': item['id'],
                        'name': item['name'],
                        'artist_id': item['artists'][0]['id'],
                        'artist_name': item['artists'][0]['name'],
                        'popularity': item['popularity'],
                        'time_range': time_range
                    }
                    self.user_data['top_tracks'].append(track_data)
                
                print(f"✅ Collected {len(results['items'])} top tracks ({time_range})")
            
            # Collecting top artists
            for time_range in time_ranges:
                results = self.sp.current_user_top_artists(limit=20, time_range=time_range)
                
                for item in results['items']:
                    artist_data = {
                        'id': item['id'],
                        'name': item['name'],
                        'genres': item['genres'],
                        'popularity': item['popularity'],
                        'time_range': time_range
                    }
                    self.user_data['top_artists'].append(artist_data)
                
                print(f"✅ Collected {len(results['items'])} top artists ({time_range})")
                
            return True
            
        except Exception as e:
            print(f"❌ Error collecting user data: {str(e)}")
            return False

    def save_user_data(self, output_dir="data"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Saving top tracks as CSV
        if self.user_data['top_tracks']:
            tracks_df = pd.DataFrame(self.user_data['top_tracks'])
            tracks_file = f"{output_dir}/user_top_tracks_{timestamp}.csv"
            tracks_df.to_csv(tracks_file, index=False)
            print(f"✅ Saved user top tracks to {tracks_file}")
        
        # Saving all user data as JSON 
        user_file = f"{output_dir}/user_data_{timestamp}.json"
        with open(user_file, 'w') as f:
            json.dump(self.user_data, f, indent=2)
        print(f"✅ Saved complete user data to {user_file}")
        
        return tracks_file, user_file

def main():
    print("🎵 Initializing Spotify authentication...")
    auth = SpotifyUserAuth()
    
    if auth.authenticate():
        print("\n📊 Collecting your music preferences...")
        if auth.collect_user_music_data():
            print("\n💾 Saving your music profile...")
            tracks_file, user_file = auth.save_user_data()
            
            print("\n✨ Summary:")
            print(f"  • Collected data for: {auth.user_data['display_name']}")
            print(f"  • Top tracks: {len(auth.user_data['top_tracks'])}")
            print(f"  • Top artists: {len(auth.user_data['top_artists'])}")
            print(f"  • Files saved: {tracks_file}, {user_file}")
            print("\nYour music profile is ready for recommendation processing! 🚀")
    
if __name__ == "__main__":
    main()