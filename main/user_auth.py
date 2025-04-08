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
        self.redirect_uri = "http://127.0.0.1:5000/callback"  
        
        if not all([self.client_id, self.client_secret, self.redirect_uri]):
            raise ValueError("Missing Spotify API credentials in .env file")
        
        self.scope = "user-top-read user-read-recently-played user-library-read playlist-modify-public playlist-modify-private"
        
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
            
            user_info = self.sp.current_user()
            self.user_data['user_id'] = user_info['id']
            self.user_data['display_name'] = user_info['display_name']
            
            print(f"✅ Successfully authenticated as {self.user_data['display_name']}")
            return True
            
        except Exception as e:
            print(f"❌ Authentication failed: {str(e)}")
            return False

    # Getting authorization URL for Spotify OAuth flow
    def get_auth_url(self, force_refresh=False):
        if force_refresh:
            import uuid
            cache_path = f".spotify_cache_{uuid.uuid4()}"
        else:
            cache_path = ".spotify_cache"
            
        auth_manager = SpotifyOAuth(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope=self.scope,
            cache_path=cache_path,
            show_dialog=True
        )
        
        return auth_manager.get_authorize_url()

    # Completing Spotify authentication with auth code
    def complete_authentication(self, code):
        try:
            auth_manager = SpotifyOAuth(
                client_id=self.client_id,
                client_secret=self.client_secret,
                redirect_uri=self.redirect_uri,
                scope=self.scope,
                show_dialog=True
            )
        
            token_info = auth_manager.get_access_token(code)
            self.set_token(token_info)
            
            user_profile = self.sp.current_user()
            
            self.user_data['user_id'] = user_profile['id']
            self.user_data['display_name'] = user_profile['display_name'] or user_profile['id']
            
            if user_profile['images'] and len(user_profile['images']) > 0:
                self.user_data['profile_image'] = user_profile['images'][0]['url']
            else:
                self.user_data['profile_image'] = None
                
            return True
            
        except Exception as e:
            print(f"Authentication error: {str(e)}")
            return False

    # Setting token from existing session
    def set_token(self, token_info):
        self.sp = spotipy.Spotify(auth=token_info['access_token'])
        self._token_info = token_info
        
        user_info = self.sp.current_user()
        self.user_data['user_id'] = user_info['id']
        self.user_data['display_name'] = user_info['display_name']
        
        return True

    # Getting the current auth token info for session storage
    def get_auth_token(self):
        return self._token_info
    
    def collect_user_music_data(self, tracks_per_period=100, artists_per_period=50):
        if not self.sp:
            print("Not authenticated. Call authenticate() first.")
            return False
            
        try:
            time_ranges = ['short_term', 'medium_term', 'long_term']
            
            # Coletar faixas para cada período
            for time_range in time_ranges:
                collected_tracks = []
                offset = 0
                
                while len(collected_tracks) < tracks_per_period:
                    # Spotify allows a max of 50 items per request
                    current_limit = min(50, tracks_per_period - len(collected_tracks))
                    
                    results = self.sp.current_user_top_tracks(
                        limit=current_limit, 
                        offset=offset,
                        time_range=time_range
                    )
                    
                    if len(results['items']) == 0:
                        break
                    
                    # Process returned items
                    for item in results['items']:
                        track_data = {
                            'id': item['id'],
                            'name': item['name'],
                            'artist_id': item['artists'][0]['id'],
                            'artist_name': item['artists'][0]['name'],
                            'popularity': item['popularity'],
                            'time_range': time_range
                        }
                        collected_tracks.append(track_data)
                    
                    offset += len(results['items'])
                    
                    if len(results['items']) < current_limit:
                        break
                        
                # Add collected tracks to user data
                self.user_data['top_tracks'].extend(collected_tracks)
                print(f"✅ Collected {len(collected_tracks)} top tracks ({time_range})")
            
            # Similar implementation for artists
            for time_range in time_ranges:
                collected_artists = []
                offset = 0
                
                while len(collected_artists) < artists_per_period:
                    current_limit = min(50, artists_per_period - len(collected_artists))
                    
                    results = self.sp.current_user_top_artists(
                        limit=current_limit, 
                        offset=offset,
                        time_range=time_range
                    )
                    
                    if len(results['items']) == 0:
                        break
                        
                    for item in results['items']:
                        artist_data = {
                            'id': item['id'],
                            'name': item['name'],
                            'genres': item['genres'],
                            'popularity': item['popularity'],
                            'time_range': time_range
                        }
                        collected_artists.append(artist_data)
                    
                    offset += len(results['items'])
                    
                    if len(results['items']) < current_limit:
                        break
                
                self.user_data['top_artists'].extend(collected_artists)
                print(f"✅ Collected {len(collected_artists)} top artists ({time_range})")
                
            return True
            
        except Exception as e:
            print(f"❌ Error collecting user data: {str(e)}")
            return False

    def save_user_data(self, output_dir="data"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        tracks_df = pd.DataFrame(self.user_data['top_tracks'])
        tracks_file = f"{output_dir}/user_top_tracks_{timestamp}.csv"
        tracks_df.to_csv(tracks_file, index=False)
        print(f"✅ Saved user top tracks to {tracks_file}")
        
        user_file = f"{output_dir}/user_data_{timestamp}.json"
        with open(user_file, 'w') as f:
            json.dump(self.user_data, f, indent=2)
        print(f"✅ Saved complete user data to {user_file}")
        
        return tracks_file, user_file

    def create_spotify_playlist(self, playlist_name, track_ids, description=""):
        if not self.sp:
            print("Not authenticated. Call authenticate() first.")
            return None
            
        try:
            playlist = self.sp.user_playlist_create(
                user=self.user_data['user_id'],
                name=playlist_name,
                public=False,
                description=description
            )
            
            playlist_id = playlist['id']
            batch_size = 100
            
            for i in range(0, len(track_ids), batch_size):
                batch = track_ids[i:min(i + batch_size, len(track_ids))]
                self.sp.playlist_add_items(playlist_id, batch)
            
            print(f"✅ Created playlist '{playlist_name}' with {len(track_ids)} tracks")
            
            return {
                'id': playlist_id,
                'name': playlist_name,
                'url': playlist['external_urls']['spotify'],
                'tracks_added': len(track_ids)
            }
            
        except Exception as e:
            print(f"❌ Error creating playlist: {str(e)}")
            return None
    
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