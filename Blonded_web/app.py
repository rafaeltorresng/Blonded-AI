from flask import Flask, render_template, redirect, request, session, url_for, jsonify
import os
import sys
import json
import pandas as pd
from datetime import datetime

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from main.user_auth import SpotifyUserAuth
from main.Recommendation import MusicRecommender

app = Flask(__name__)
app.secret_key = os.urandom(24) 

DATASET_PATH = os.path.join(parent_dir, 'data', 'processed_dataset.csv')
SCALER_PATH = os.path.join(parent_dir, 'data', 'scaler_model.pkl')
OUTPUT_DIR = os.path.join(parent_dir, 'recommendations')
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login')
def login():
    auth = SpotifyUserAuth()
    
    session['auth_pending'] = True
    
    return redirect(auth.get_auth_url())

@app.route('/callback')
def callback():
    if not session.get('auth_pending'):
        return redirect('/')
    
    code = request.args.get('code')
    
    if not code:
        return render_template('error.html', error_message="Authorization failed. No code provided.")
    
    # Complete authentication
    auth = SpotifyUserAuth()
    success = auth.complete_authentication(code)
    
    if not success:
        return render_template('error.html', error_message="Failed to authenticate with Spotify")
    
    # Store user info in session
    session['user_id'] = auth.user_data['user_id']
    session['display_name'] = auth.user_data['display_name']
    session['auth_token'] = auth.get_auth_token()
    
    session.pop('auth_pending', None)
    
    # Redirect to loading page
    return redirect(url_for('loading'))

@app.route('/loading')
def loading():
    # Check if user is authenticated
    if 'user_id' not in session:
        return redirect('/')
    
    return render_template('loading.html')

@app.route('/api/generate-recommendations', methods=['POST'])
def generate_recommendations():
    # Check if user is authenticated
    if 'auth_token' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
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
        
        user_profile = recommender.create_user_profile(user_tracks)
        
        # Generate recommendations
        playlist = recommender.generate_playlist(
            user_profile,
            name=f"{session['display_name']}'s Recommended Playlist"
        )
        
        # Get artist images from Spotify
        for artist in playlist['artists']:
            artist_data = auth.sp.search(q=f"artist:{artist['artist']}", type='artist', limit=1)
            if artist_data['artists']['items']:
                artist_obj = artist_data['artists']['items'][0]
                artist['id'] = artist_obj['id']
                artist['image_url'] = artist_obj['images'][0]['url'] if artist_obj['images'] else None
                artist['genres'] = artist_obj.get('genres', [])
        
        # Save recommendations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(OUTPUT_DIR, f"recommendations_{session['user_id']}_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(playlist, f, indent=2)
        
        # Store in session for later access
        session['recommendation_file'] = results_file
        session['playlist'] = playlist
        
        return jsonify({'success': True, 'redirect': url_for('recommendations')})
        
    except Exception as e:
        print(f"Error generating recommendations: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/recommendations')
def recommendations():
    # Check if user is authenticated
    if 'user_id' not in session:
        return redirect('/')
    
    # Check if we have recommendations
    if 'playlist' not in session:
        return redirect(url_for('loading'))
    
    return render_template('recommendations.html', 
                          user=session.get('display_name'),
                          playlist=session.get('playlist'))

@app.route('/api/export-playlist', methods=['POST'])
def export_playlist():
    # Check if user is authenticated
    if 'auth_token' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    try:
        # Initialize auth with stored token
        auth = SpotifyUserAuth()
        auth.set_token(session['auth_token'])
        
        # Get playlist from session
        playlist = session.get('playlist')
        
        if not playlist:
            return jsonify({'success': False, 'error': 'No playlist found'}), 404
        
        # Extract track IDs
        track_ids = [track['track_id'] for track in playlist['tracks']]
        
        # Create playlist description
        genres = [g['genre'] for g in playlist['genres'][:3]]
        description = f"Recommended by Blonded AI based on your listening history. Top genres: {', '.join(genres)}"
        
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

# Error handler for 404
@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_message="Page not found"), 404

# Error handler for 500
@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', 
                          error_message="Server error", 
                          technical_details=str(e)), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)