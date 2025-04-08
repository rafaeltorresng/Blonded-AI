from flask import Flask, render_template, redirect, request, session, url_for, jsonify
from flask_session import Session  # For server-side sessions
import os
import sys
import json
import pickle
import pandas as pd
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from main.user_auth import SpotifyUserAuth
from main.Recommendation import MusicRecommender

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure server-side sessions to avoid cookie size limitations
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = os.path.join(parent_dir, 'flask_sessions')
app.config["SESSION_PERMANENT"] = False
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
Session(app)

# Path configurations
DATASET_PATH = os.path.join(parent_dir, 'data', 'processed_dataset.csv')
SCALER_PATH = os.path.join(parent_dir, 'model', 'scaler_model.pkl')  # Changed from 'model' to 'data'
OUTPUT_DIR = os.path.join(parent_dir, 'recommendations')

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)

@app.route('/')
def index():
    """Home page with login button"""
    # Clear any existing session data
    session.clear()
    return render_template('index.html')

@app.route('/login')
def login():
    """Initiate Spotify OAuth flow"""
    auth = SpotifyUserAuth()
    session['auth_pending'] = True
    return redirect(auth.get_auth_url())

@app.route('/callback')
def callback():
    """Handle Spotify OAuth callback"""
    if not session.get('auth_pending'):
        return redirect('/')
    
    code = request.args.get('code')
    error = request.args.get('error')
    
    if error:
        return render_template('error.html', 
                              error_message=f"Authorization failed: {error}")
    
    if not code:
        return render_template('error.html', 
                              error_message="Authorization failed. No code provided.")
    
    # Complete authentication with Spotify
    auth = SpotifyUserAuth()
    success = auth.complete_authentication(code)
    
    if not success:
        return render_template('error.html', 
                              error_message="Failed to authenticate with Spotify")
    
    # Store minimal user info in session
    session['user_id'] = auth.user_data['user_id']
    session['display_name'] = auth.user_data['display_name']
    session['profile_image'] = auth.user_data['profile_image']
    session['auth_token'] = auth.get_auth_token()
    
    # Clear authentication flag
    session.pop('auth_pending', None)
    
    # Redirect to loading page
    return redirect(url_for('loading'))

@app.route('/loading')
def loading():
    """Show loading screen during recommendation generation"""
    if 'user_id' not in session:
        return redirect('/')
    
    # Check if recommendations already exist to avoid redundant processing
    if 'recommendation_file' in session and os.path.exists(session['recommendation_file']):
        # If recommendations already exist, go directly to recommendations page
        return redirect(url_for('recommendations'))
    
    return render_template('loading.html')

@app.route('/api/generate-recommendations', methods=['POST'])
def generate_recommendations():
    """API endpoint to generate music recommendations"""
    if 'auth_token' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    # Check if required files exist
    if not os.path.exists(DATASET_PATH):
        return jsonify({'success': False, 
                       'error': 'Dataset file not found. Please run data processing first.'}), 500
    
    
    try:
        # Initialize auth with stored token
        auth = SpotifyUserAuth()
        auth.set_token(session['auth_token'])
        
        # Collect user music data
        auth.collect_user_music_data()
        
        # Save user data
        tracks_file, _ = auth.save_user_data()
        
        # Load user tracks
        user_tracks_df = pd.read_csv(tracks_file)
        
        # Format for recommendation engine
        user_tracks = pd.DataFrame({
            'id': user_tracks_df['id'],
            'name': user_tracks_df['name'],
            'artist_name': user_tracks_df['artist_name']
        })
        
        # Initialize recommender
        recommender = MusicRecommender(DATASET_PATH, SCALER_PATH)
        
        # Create user profile
        user_profile = recommender.create_user_profile(user_tracks)
        
        # Generate recommendations
        playlist = recommender.generate_playlist(
            user_profile,
            name=f"{session['display_name']}'s Recommended Playlist"
        )
        
        # Get artist images from Spotify
        for artist in playlist['artists']:
            try:
                artist_name = artist['artist']
                print(f"Searching for artist: {artist_name}")
                
                # Try exact search first
                artist_data = auth.sp.search(q=f"artist:\"{artist_name}\"", type='artist', limit=5)
                
                # Check for exact name matches first
                exact_match = None
                for item in artist_data['artists']['items']:
                    if item['name'].lower() == artist_name.lower():
                        exact_match = item
                        break
                
                # If no exact match, try to find closest match
                if not exact_match and artist_data['artists']['items']:
                    # Use the first result but log a warning
                    exact_match = artist_data['artists']['items'][0]
                    print(f"⚠️ No exact match for '{artist_name}', using '{exact_match['name']}' instead")
                
                # If we found a match, use it
                if exact_match:
                    artist['id'] = exact_match['id']
                    artist['image_url'] = exact_match['images'][0]['url'] if exact_match['images'] else None
                    artist['genres'] = exact_match.get('genres', [])
                    print(f"✅ Found match for '{artist_name}': {exact_match['name']}")
                    
                    # Handle special case for Adele if that's the problem
                    if artist_name.lower() == "adele" and artist['id'] != "4dpARuHxo51G3z768sgnrY":
                        # Hardcode Adele's correct Spotify ID and fetch data
                        adele_id = "4dpARuHxo51G3z768sgnrY"
                        try:
                            adele_data = auth.sp.artist(adele_id)
                            artist['id'] = adele_id
                            artist['image_url'] = adele_data['images'][0]['url'] if adele_data['images'] else None
                            artist['genres'] = adele_data.get('genres', [])
                            print("ℹ️ Used hardcoded ID for Adele")
                        except Exception as e:
                            print(f"Error fetching Adele with hardcoded ID: {str(e)}")
                else:
                    print(f"❌ No match found for '{artist_name}'")
            except Exception as e:
                print(f"Error fetching artist data for {artist['artist']}: {str(e)}")
                continue
        
        # Save recommendations to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(OUTPUT_DIR, f"recommendations_{session['user_id']}_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(playlist, f, indent=2)
        
        # Store only the file path in session, not the entire playlist
        session['recommendation_file'] = results_file
        
        # Success response
        return jsonify({'success': True, 'redirect': url_for('recommendations')})
        
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/recommendations')
def recommendations():
    """Show recommendation results page"""
    # Check if user is authenticated
    if 'user_id' not in session:
        return redirect('/')
    
    # Check if we have a recommendation file
    if 'recommendation_file' not in session:
        return redirect(url_for('loading'))
    
    # Load recommendations from file
    try:
        # Debug prints
        print(f"Loading recommendation file: {session['recommendation_file']}")
        print(f"File exists: {os.path.exists(session['recommendation_file'])}")
        
        # Verify the file exists and is a valid path
        if not os.path.exists(session['recommendation_file']):
            raise FileNotFoundError(f"Recommendation file not found: {session['recommendation_file']}")
        
        # Try to load and parse the JSON
        try:
            with open(session['recommendation_file'], 'r') as f:
                playlist = json.load(f)
                
            # Debug what we loaded
            print(f"Loaded playlist with keys: {playlist.keys()}")
            
        except json.JSONDecodeError as je:
            print(f"JSON decode error: {str(je)}")
            raise
        
        # Check that we got valid data
        if not isinstance(playlist, dict):
            raise ValueError(f"Invalid playlist type: {type(playlist)}")
        
        if 'tracks' not in playlist:
            raise ValueError(f"Missing 'tracks' key in playlist. Keys: {playlist.keys()}")
        
        # Debug template rendering
        try:
            return render_template('recommendations.html', 
                                 user=session.get('display_name'),
                                 playlist=playlist)
        except Exception as template_error:
            print(f"Template rendering error: {str(template_error)}")
            raise
                              
    except FileNotFoundError as e:
        print(f"File not found error: {str(e)}")
        # If file is missing, redirect back to loading to regenerate
        session.pop('recommendation_file', None)
        return redirect(url_for('loading'))
        
    except Exception as e:
        print(f"Error loading recommendations: {str(e)}")
        print(f"Session contains: {list(session.keys())}")
        
        # Get the actual exception info
        import traceback
        trace = traceback.format_exc()
        
        return render_template('error.html', 
                             error_message="Could not load recommendations", 
                             technical_details=f"{str(e)}\n\n{trace}")

@app.route('/api/export-playlist', methods=['POST'])
def export_playlist():
    """API endpoint to export playlist to Spotify"""
    # Check if user is authenticated
    if 'auth_token' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    # Check if we have a recommendation file
    if 'recommendation_file' not in session:
        return jsonify({'success': False, 'error': 'No recommendations found'}), 404
    
    try:
        # Initialize auth with stored token
        auth = SpotifyUserAuth()
        auth.set_token(session['auth_token'])
        
        # Load playlist from file
        with open(session['recommendation_file'], 'r') as f:
            playlist = json.load(f)
        
        if not playlist:
            return jsonify({'success': False, 'error': 'No playlist found'}), 404
        
        # Extract track IDs
        track_ids = [track['track_id'] for track in playlist['tracks']]
        
        # Create playlist description
        genres = [g['genre'] for g in playlist.get('genres', [])[:3]] if playlist.get('genres') else []
        genre_text = f"Top genres: {', '.join(genres)}" if genres else ""
        description = f"Recommended by Blonded AI based on your listening history. {genre_text}"
        
        # Create the playlist
        result = auth.create_spotify_playlist(
            playlist_name=playlist['name'],
            track_ids=track_ids,
            description=description
        )
        
        if not result:
            return jsonify({'success': False, 'error': 'Failed to create playlist'}), 500
        
        return jsonify({'success': True, 'playlist_url': result['url']})
        
    except Exception as e:
        print(f"Error exporting playlist: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/logout')
def logout():
    """Log out and clear session"""
    session.clear()
    return redirect(url_for('index'))

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_message="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', 
                          error_message="Server error", 
                          technical_details=str(e)), 500

if __name__ == '__main__':
    # Make sure directories exist
    os.makedirs('data', exist_ok=True)
    app.run(host='127.0.0.1', port=5000, debug=True)