# SentiTunes: AI-Based Mood Playlist Generator

SentiTunes is an AI-powered web app that analyzes your mood from text or voice and recommends personalized music playlists. Whether you’re feeling happy, sad, nostalgic, or anything in between, SentiTunes finds the perfect tunes for your emotions—instantly!

---

## 🚀 Features

- **Multi-Modal Mood Detection:**  
  Detects your mood using advanced NLP (VADER, KNN, keyword, and fuzzy matching).
- **Voice & Text Input:**  
  Speak or type your feelings—SentiTunes understands both!
- **Personalized Playlist Recommendations:**  
  Suggests YouTube Music playlists tailored to your mood, genre, and time preferences.
- **Modern, Responsive UI:**  
  Built with Streamlit for a smooth and interactive experience.
- **Region & Genre Awareness:**  
  Supports regional music and genre-based filtering.
- **Confidence Scores:**  
  See how confident the AI is in its mood predictions.

---

## 🖥️ Demo

<!-- Add a screenshot if available -->
<!-- ![SentiTunes Screenshot](static/images/demo_screenshot.png) -->

*Describe your mood or speak it out—get instant playlist recommendations!*

---

## 🛠️ Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/mood_playlist_app.git
    cd mood_playlist_app
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows:
    venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(If you don’t have a requirements.txt, install manually:)*  
    ```bash
    pip install streamlit speechrecognition pyaudio ytmusicapi vaderSentiment fuzzywuzzy python-Levenshtein
    ```

4. **Run the app:**
    ```bash
    streamlit run app.py
    ```

---

## 🎤 Usage

- **Text Input:**  
  Type your feelings and click “Analyze My Mood & Find Music”.
- **Voice Input:**  
  Click “Speak Your Feelings”, record your voice, and let SentiTunes do the rest!
- **Get Playlists:**  
  Instantly receive music playlists that match your mood.

---

## 🧠 How It Works

- **Mood Detection:**  
  Uses VADER sentiment analysis, KNN classification, keyword, and fuzzy matching to detect your mood from input.
- **Playlist Recommendation:**  
  Fetches relevant playlists from YouTube Music based on detected mood, genre, region, and time period.
- **Confidence Display:**  
  Shows how confident each AI method is in its prediction.

---

## 📦 Project Structure

```
mood_playlist_app/
│
├── app.py                # Streamlit frontend
├── main.py               # Core logic: mood detection, playlist fetching
├── voice_input.py        # Voice-to-text logic
├── static/images/        # App images
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

---

## 🙏 Credits

- [Streamlit](https://streamlit.io/)
- [VADER Sentiment](https://github.com/cjhutto/vaderSentiment)
- [YTMusicAPI](https://ytmusicapi.readthedocs.io/)
- [FuzzyWuzzy](https://github.com/seatgeek/fuzzywuzzy)
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)

---

## 📣 Future Enhancements

- Webcam-based emotion detection (DeepFace + OpenCV)
- Spotify/Apple Music integration
- User mood analytics and history
- Multi-language support

---