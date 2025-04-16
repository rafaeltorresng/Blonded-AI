#  Blonded AI
Blonded AI is my personal project to blend the worlds of AI and music. It utilizes the power of music embeddings to understand your taste at a deeper level based on Spotify data, creating highly personalized recommendations.

## ✨ Features
### 🎯 Embedding-Based Recommendations
- Uses advanced vector representations of songs through PCA (Principal Component Analysis)
- Optimally reduces 9 audio features to 6 dimensions while preserving 90% of information
- Enhances recommendation quality by capturing correlated musical patterns
### 🔍 Vector Database Search
- Leverages ChromaDB for high-performance similarity search
- Indexes over 113,000 songs for instant retrieval
- Enables sub-second recommendations even with complex preference vectors
### 🎸 Artists and Tracks Discovery
- Finds similar artists based on musical style
- Suggests tracks matching your taste profile
- Explores new genres aligned with your preferences
### 🎵 Spotify Integration
- Seamless connection with your Spotify account
- One-click playlist creation
- Real-time synchronization with your library

## 🛠️ Project Building Steps
- Data Collection: Gathered a comprehensive dataset of Spotify Tracks with audio features and metadata.
- Embedding Generation: Created song embeddings using PCA to efficiently represent musical characteristics.
- Vector Database: Implemented ChromaDB to enable efficient similarity search across thousands of songs.
- Recommendation Engine: Built an algorithm that balances similarity, diversity, and popularity.
- Web App Development: Built a Flask-based web application to provide a user-friendly interface.
- Spotify Integration: Integrated with the Spotify API to collect user data and export playlists.

## 💻 Technologies Used
All the skills and technologies used in this project:

<p align="left">
<a href="https://www.python.org/" target="_blank" rel="noreferrer"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/python/python-original.svg" width="36" height="36" alt="Python" /></a>
<a href="https://flask.palletsprojects.com/" target="_blank" rel="noreferrer"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/flask/flask-original.svg" width="36" height="36" alt="Flask" /></a>
<a href="https://developer.mozilla.org/en-US/docs/Web/JavaScript" target="_blank" rel="noreferrer"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/javascript/javascript-original.svg" width="36" height="36" alt="JavaScript" /></a>
<a href="https://vuejs.org/" target="_blank" rel="noreferrer"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/vuejs/vuejs-original.svg" width="36" height="36" alt="Vue" /></a>
<a href="https://developer.mozilla.org/en-US/docs/Glossary/HTML5" target="_blank" rel="noreferrer"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/html5/html5-original.svg" width="36" height="36" alt="HTML5" /></a>
<a href="https://www.w3.org/TR/CSS/#css" target="_blank" rel="noreferrer"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/css3/css3-original.svg" width="36" height="36" alt="CSS3" /></a>
<a href="https://tailwindcss.com/" target="_blank" rel="noreferrer"><img src="https://raw.githubusercontent.com/danielcranney/readme-generator/main/public/icons/skills/tailwindcss-colored.svg" width="36" height="36" alt="TailwindCSS" /></a>
<a href="https://numpy.org/" target="_blank" rel="noreferrer"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/numpy/numpy-original.svg" width="36" height="36" alt="NumPy" /></a>
<a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/pandas/pandas-original.svg" width="36" height="36" alt="Pandas" /></a>
<a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"><img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" width="36" height="36" alt="scikit-learn" /></a>
<a href="https://www.tensorflow.org/" target="_blank" rel="noreferrer"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/tensorflow/tensorflow-original.svg" width="36" height="36" alt="TensorFlow" /></a>
<a href="https://www.sqlite.org/" target="_blank" rel="noreferrer"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/sqlite/sqlite-original.svg" width="36" height="36" alt="SQLite" /></a>
<a href="https://developer.spotify.com/" target="_blank" rel="noreferrer"><img src="https://www.vectorlogo.zone/logos/spotify/spotify-icon.svg" width="36" height="36" alt="Spotify API" /></a>
</p>

### Home Page
![Home Page](/Blonded_web/static/img/02Blond.png)

### Recommendation
![Home Page](/Blonded_web/static/img/BLONDDD01.png)

## ⚙️ Workflow
```bash
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│                     │     │                     │     │                     │
│  User Authentication│────►│  Track Matching     │────►│  Feature            │
│  & Data Collection  │     │  with Dataset       │     │  Extraction         │
│                     │     │                     │     │                     │
└─────────────────────┘     └─────────────────────┘     └──────────┬──────────┘
                                                                   │
                                                                   ▼
┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
│                     │     │                     │     │                     │
│  Recommendation     │◄────│  Result             │◄────│  Vector Database    │
│  Generation         │     │  Processing         │     │  Similarity Search  │
│                     │     │                     │     │                     │
└──────────┬──────────┘     └─────────────────────┘     └─────────────────────┘
           │
           ▼
┌─────────────────────┐
│                     │
│  Export to          │
│  Spotify            │
│                     │
└─────────────────────┘
```
### 📊 Dataset Audio Features
The recommendation system analyzes songs using these Spotify audio features:

- **Popularity** (0-100)
- **Danceability** (0-1) 
- **Energy** (0-1)
- **Loudness** (-60-0 dB)
- **Acousticness** (0-1)
- **Instrumentalness** (0-1) 
- **Liveness** (0-1)
- **Valence** (0-1)
- **Tempo** (BPM)

These features form the foundation of our embeddings, which are then optimized through PCA to create more meaningful song representations for recommendations.

## 🖥️  Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Configure Environment Variables:
fill into .env your Spotify API credentials:
   ```bash
    SPOTIPY_CLIENT_ID=your_client_id
    SPOTIPY_CLIENT_SECRET=your_client_secret
    SPOTIPY_REDIRECT_URI=http://localhost:5000/callback
To get your Spotify API credentials:
- Go to the Spotify Developer Dashboard.
- Log in with your Spotify account.
- Create a new app.
- Note the Client ID and Client Secret.
- Set the Redirect URI to http://localhost:5000/callback.
4. Run the Flask Application:
Navigate to the Blonded_web directory and run the Flask application:
   ```bash
    cd Blonded_web
    flask run
5. Access the Application:
Open your web browser and go to http://127.0.0.1:5000 to access the Blonded AI application.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request or contact me if you liked the idea and want develop something.
