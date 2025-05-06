import streamlit as st
import time
import os
from main import detect_mood_genre_keywords, determine_final_mood, extract_time_range, fetch_playlists, recommend_music
from main import train_knn_model

# Set page config
st.set_page_config(
    page_title="SentiTunes",
    page_icon="🎵",
    layout="wide"
)

# Custom CSS styles
st.markdown(
    """
    <style>
    .main {
        background-color: #1c1c1e;
        color: white;
        text-align: center;
        padding: 30px 10px;
    }
    .square-img {
        width: 200px;
        height: 200px;
        object-fit: cover;
        border: 3px solid lightgrey;
    }
    .title {
        font-size: 4.5rem;
        font-weight: bold;
        color: #ff4b94;
        margin-bottom: 0;
    }
    .subtitle {
        font-size: 2rem;
        color: #f5c2d0;
    }
    .mood-card {
        background-color: #292930;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .playlist-card {
        background-color: #292930;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        transition: transform 0.3s;
    }
    .playlist-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    .song-card {
        background-color: #1F1F23;
        border-radius: 8px;
        padding: 10px 15px;
        margin: 8px 0;
        border-left: 4px solid #ff4b94;
    }
    .recommendation-title {
        font-size: 24px;
        color: #f5c2d0;
        margin: 15px 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)
@st.cache_resource  # This will cache the model
def load_knn_model():
    knn_model, _, _ = train_knn_model()
    return knn_model

knn_model = load_knn_model()

# Layout
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.image("static/images/bawra.jpg", width=300)

with col2:
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="title">SentiTunes</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">An AI powered playlist generating platform based on your mood</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.image("static/images/gustakh.jpg", width=300)
    
# Input section with enlarged title
st.markdown(
    '<p style="font-size:35px; font-weight:600; color:white;">How was your day?</p>',
    unsafe_allow_html=True
)
user_feelings = st.text_area(
    "Share your thoughts and emotions...", 
    height=200, 
    placeholder="Describe about your day here... add info about any specific genre you want to hear or music from a particular timeperiod. If in dilemma leave it to us we will do that job for you!!!"
)
# Submit button
# This section in app.py needs to be updated to properly handle the playlist results

# Inside the submit button section:
# Inside the submit button section:
if st.button("Analyze My Mood", key="submit"):
    if user_feelings:
        with st.spinner("Analyzing your mood and finding the perfect tunes..."):
            try:
                # First get mood analysis results
                mood, mood_results, confidence_scores = determine_final_mood(user_feelings, knn_model)
                
                # Then extract genre and other preferences
                _, genre, _ = detect_mood_genre_keywords(user_feelings)
                years = extract_time_range(user_feelings)
                country = 'IN'  # Default to India
                
                # Display mood analysis
                st.markdown('<div class="mood-card">', unsafe_allow_html=True)
                st.markdown("### 🎭 Mood Analysis Results")
                st.markdown(f"**Detected Mood:** {mood.title()}")
                st.markdown("#### Analysis Details:")
                
                method_names = {
                    'vader': 'Sentiment Analysis',
                    'keyword': 'Keyword Detection',
                    'fuzzy': 'Fuzzy Matching',
                    'knn': 'ML Classification'
                }

                # Create two columns for the scores
                col1, col2 = st.columns(2)

                with col1:
                    for method, result in mood_results.items():
                        if result:
                            confidence = confidence_scores.get(method, 0)
                            confidence_str = f"{confidence:.0%}" if confidence else "N/A"
                            st.markdown(f"• {method_names[method]}: **{result.upper()}** (confidence: {confidence_str})")

                with col2:
                    if genre:
                        st.markdown(f"**Genre Detected:** {genre.title()}")
                    st.markdown(f"**Time Range:** {'Recent' if years == 1 else f'Last {years} years'}")
                    st.markdown(f"**Region:** {country}")

                st.markdown('</div>', unsafe_allow_html=True)

                # Get playlist recommendations
                playlists = fetch_playlists(mood, genre, country, years)
                
                # Rest of your existing playlist display code...
                if playlists and len(playlists) > 0:
                    # ...existing playlist display code...
                    st.markdown('<div class="mood-card">', unsafe_allow_html=True)
                    st.markdown(f"## 🎵 Music For Your {mood.title()} Mood")
                    
                    for playlist in playlists:
                        if isinstance(playlist, dict) and 'title' in playlist and 'url' in playlist:
                            st.markdown(
                                f"""
                                <div class="playlist-card">
                                    <h3>🎵 {playlist['title']}</h3>
                                    <a href="{playlist['url']}" target="_blank">Open in YouTube Music</a>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )
                    
                    st.markdown(
                        f"<p style='color: #888; font-size: 12px; margin-top: 20px;'>Analysis completed on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>", 
                        unsafe_allow_html=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.warning("No playlists found for your mood. Try describing your feelings differently.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                import traceback
                st.error(traceback.format_exc())  # Show the full traceback for debugging
    else:
        st.warning("Please enter your feelings before submitting.")