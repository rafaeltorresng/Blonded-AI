from flask import Flask, render_template, redirect, request, session, url_for, jsonify
from flask_session import Session
import os
import sys
import json
import pickle
import pandas as pd
from datetime import datetime

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from main.user_auth import SpotifyUserAuth
from AI.recommender import MusicRecommender

app = Flask(__name__)
app.secret_key = os.urandom(24)

app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_FILE_DIR"] = os.path.join(parent_dir, 'flask_sessions')
app.config["SESSION_PERMANENT"] = False
os.makedirs(app.config["SESSION_FILE_DIR"], exist_ok=True)
Session(app)

DATASET_PATH = os.path.join(parent_dir, 'data', 'processed_dataset.csv')
SCALER_PATH = os.path.join(parent_dir, 'model', 'scaler_model.pkl')
OUTPUT_DIR = os.path.join(parent_dir, 'recommendations')
CHROMA_DIR = os.path.join(parent_dir, 'chroma_db')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

try:
    print("Inicializando o recomendador com ChromaDB...")
    global_recommender = MusicRecommender(
        dataset_path=DATASET_PATH,
        scaler_path=SCALER_PATH,
        n_components=6,
        use_chromadb=True,
        chromadb_path=CHROMA_DIR
    )
    print(f"Recomendador inicializado com sucesso. ChromaDB disponível: {global_recommender.is_vector_search_available()}")
except Exception as e:
    print(f"Erro ao inicializar recomendador: {str(e)}")
    print("Continuando sem ChromaDB...")
    global_recommender = None

@app.route('/')
def index():
    session.clear()
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
    error = request.args.get('error')
    if error:
        return render_template('error.html', error_message=f"Authorization failed: {error}")
    if not code:
        return render_template('error.html', error_message="Authorization failed. No code provided.")
    auth = SpotifyUserAuth()
    success = auth.complete_authentication(code)
    if not success:
        return render_template('error.html', error_message="Failed to authenticate with Spotify")
    session['user_id'] = auth.user_data['user_id']
    session['display_name'] = auth.user_data['display_name']
    session['profile_image'] = auth.user_data['profile_image']
    session['auth_token'] = auth.get_auth_token()
    session.pop('auth_pending', None)
    return redirect(url_for('loading'))

@app.route('/loading')
def loading():
    if 'user_id' not in session:
        return redirect('/')
    if 'recommendation_file' in session and os.path.exists(session['recommendation_file']):
        return redirect(url_for('recommendations'))
    return render_template('loading.html')

@app.route('/api/generate-recommendations', methods=['POST'])
def generate_recommendations():
    if 'auth_token' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    if not os.path.exists(DATASET_PATH):
        return jsonify({'success': False, 'error': 'Dataset file not found'}), 500
    try:
        auth = SpotifyUserAuth()
        auth.set_token(session['auth_token'])
        auth.collect_user_music_data()
        tracks_file, _ = auth.save_user_data()
        user_tracks_df = pd.read_csv(tracks_file)
        user_tracks = pd.DataFrame({
            'id': user_tracks_df['id'],
            'name': user_tracks_df['name'],
            'artist_name': user_tracks_df['artist_name']
        })
        recommender = global_recommender
        if not recommender:
            print("Recomendador global não disponível. Criando nova instância...")
            recommender = MusicRecommender(DATASET_PATH, SCALER_PATH)
        user_profile = recommender.create_user_profile(user_tracks)
        use_vector_search = recommender.is_vector_search_available()
        print(f"Gerando playlist com busca vetorial: {use_vector_search}")
        playlist = recommender.generate_playlist(
            user_profile,
            name=f"{session['display_name']}'s Recommended Playlist",
            use_vector_search=use_vector_search
        )
        for artist in playlist['artists']:
            try:
                artist_name = artist['artist']
                artist_data = auth.sp.search(q=f"artist:\"{artist_name}\"", type='artist', limit=5)
                exact_match = None
                for item in artist_data['artists']['items']:
                    if item['name'].lower() == artist_name.lower():
                        exact_match = item
                        break
                if not exact_match and artist_data['artists']['items']:
                    exact_match = artist_data['artists']['items'][0]
                if exact_match:
                    artist['id'] = exact_match['id']
                    artist['image_url'] = exact_match['images'][0]['url'] if exact_match['images'] else None
                    artist['genres'] = exact_match.get('genres', [])
                    if artist_name.lower() == "adele" and artist['id'] != "4dpARuHxo51G3z768sgnrY":
                        adele_id = "4dpARuHxo51G3z768sgnrY"
                        try:
                            adele_data = auth.sp.artist(adele_id)
                            artist['id'] = adele_id
                            artist['image_url'] = adele_data['images'][0]['url'] if adele_data['images'] else None
                            artist['genres'] = adele_data.get('genres', [])
                        except Exception:
                            pass
            except Exception:
                continue
        playlist['metadata'] = {
            'vector_search_used': use_vector_search,
            'recommendation_engine': 'ChromaDB' if use_vector_search else 'In-memory similarity'
        }
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(OUTPUT_DIR, f"recommendations_{session['user_id']}_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(playlist, f, indent=2)
        session['recommendation_file'] = results_file
        return jsonify({'success': True, 'redirect': url_for('recommendations')})
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        print(f"Erro ao gerar recomendações: {str(e)}\n{trace}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/recommendations')
def recommendations():
    if 'user_id' not in session:
        return redirect('/')
    if 'recommendation_file' not in session:
        return redirect(url_for('loading'))
    try:
        if not os.path.exists(session['recommendation_file']):
            raise FileNotFoundError(f"Recommendation file not found: {session['recommendation_file']}")
        with open(session['recommendation_file'], 'r') as f:
            playlist = json.load(f)
        if not isinstance(playlist, dict):
            raise ValueError(f"Invalid playlist type: {type(playlist)}")
        if 'tracks' not in playlist:
            raise ValueError(f"Missing 'tracks' key in playlist")
        vector_search_used = playlist.get('metadata', {}).get('vector_search_used', False)
        recommendation_engine = playlist.get('metadata', {}).get('recommendation_engine', 'Unknown')
        return render_template('recommendations.html', 
                             user=session.get('display_name'),
                             playlist=playlist,
                             vector_search_used=vector_search_used,
                             recommendation_engine=recommendation_engine)
    except FileNotFoundError:
        session.pop('recommendation_file', None)
        return redirect(url_for('loading'))
    except Exception as e:
        import traceback
        trace = traceback.format_exc()
        return render_template('error.html', 
                             error_message="Could not load recommendations", 
                             technical_details=f"{str(e)}\n\n{trace}")

@app.route('/api/export-playlist', methods=['POST'])
def export_playlist():
    if 'auth_token' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    if 'recommendation_file' not in session:
        return jsonify({'success': False, 'error': 'No recommendations found'}), 404
    try:
        auth = SpotifyUserAuth()
        auth.set_token(session['auth_token'])
        with open(session['recommendation_file'], 'r') as f:
            playlist = json.load(f)
        if not playlist:
            return jsonify({'success': False, 'error': 'No playlist found'}), 404
        track_ids = [track['track_id'] for track in playlist['tracks']]
        genres = [g['genre'] for g in playlist.get('genres', [])[:3]] if playlist.get('genres') else []
        genre_text = f"Top genres: {', '.join(genres)}" if genres else ""
        description = f"Recommended by Blonded AI based on your listening history. {genre_text}"
        result = auth.create_spotify_playlist(
            playlist_name=playlist['name'],
            track_ids=track_ids,
            description=description
        )
        if not result:
            return jsonify({'success': False, 'error': 'Failed to create playlist'}), 500
        return jsonify({'success': True, 'playlist_url': result['url']})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/logout')
def logout():
    user_id = session.get('user_id')
    session.clear()
    for cache_path in [".spotify_cache", f".cache-{user_id}" if user_id else None]:
        if cache_path and os.path.exists(cache_path):
            try:
                os.remove(cache_path)
            except Exception:
                pass
    return redirect(url_for('index', _cache_bust=datetime.now().timestamp()))

@app.route('/status')
def status():
    if not global_recommender:
        return jsonify({
            'status': 'error',
            'message': 'Recomendador não inicializado'
        }), 500
    chroma_available = global_recommender.is_vector_search_available()
    chroma_count = global_recommender.get_chromadb_count() if chroma_available else 0
    return jsonify({
        'status': 'ok',
        'chromadb_available': chroma_available,
        'chromadb_items': chroma_count,
        'dataset_size': len(global_recommender.dataset) if hasattr(global_recommender, 'dataset') else 0,
        'embedding_dimensions': global_recommender.n_components if hasattr(global_recommender, 'n_components') else 0,
        'pca_enabled': True
    })

@app.route('/admin/rebuild-chromadb', methods=['POST'])
def rebuild_chromadb():
    global global_recommender
    try:
        if not global_recommender:
            return jsonify({'success': False, 'error': 'Recomendador não inicializado'}), 500
        success = global_recommender.rebuild_chromadb()
        if success:
            return jsonify({
                'success': True,
                'message': 'ChromaDB reconstruído com sucesso',
                'collection_size': global_recommender.get_chromadb_count()
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Falha ao reconstruir ChromaDB'
            }), 500
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error_message="Page not found"), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('error.html', 
                          error_message="Server error", 
                          technical_details=str(e)), 500

if __name__ == '__main__':
    os.makedirs('data', exist_ok=True)
    app.run(host='127.0.0.1', port=5000, debug=True)