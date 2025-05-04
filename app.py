import streamlit as st
import time
import os
from mood_playlist_app.sentiment_analyzer import detect_mood_genre, fetch_recommendations

# Set page config
st.set_page_config(layout="wide")

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
    .sentiment-score {
        font-size: 18px;
        font-weight: bold;
        margin: 5px 0;
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

# Layout
col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    st.image("images/bawra.jpg", width=300)

with col2:
    st.markdown('<div class="main">', unsafe_allow_html=True)
    st.markdown('<div class="title">SentiTunes</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">An AI powered playlist generating platform based on your mood</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.image("images/gustakh.png", width=300)

# Input section with enlarged title
st.markdown(
    '<p style="font-size:35px; font-weight:600; color:white;">How was your day?</p>',
    unsafe_allow_html=True
)
user_feelings = st.text_area("Share your thoughts and emotions...", height=200, placeholder="Describe about your day here... add info about any specific genre you want to hear or music from a particular timeperiod. If in dilemma leave it to us we will do that job for you!!!")

# Submit button
if st.button("Analyze My Mood", key="submit"):
    if user_feelings:
        # Show spinner while analyzing
        with st.spinner("Analyzing your mood and finding the perfect tunes..."):
            # Get sentiment analysis results
            analysis_results = detect_mood_genre(user_feelings)
            
            # Display mood analysis in a stylish card
            st.markdown('<div class="mood-card">', unsafe_allow_html=True)
            st.markdown(f"### Mood Analysis Results")
            st.markdown(f"**Detected Mood:** {analysis_results['mood'].title()}")
            if analysis_results['genre']:
                st.markdown(f"**Preferred Genre:** {analysis_results['genre'].title()}")
            
            # Display VADER scores
            st.markdown("**Sentiment Scores:**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                score = analysis_results['vader_scores']['pos']
                st.markdown(f'<p class="sentiment-score">Positive: <span style="color:{"#4CAF50" if score > 0.2 else "#757575"}">{score:.3f}</span></p>', unsafe_allow_html=True)
            
            with col2:
                score = analysis_results['vader_scores']['neg']
                st.markdown(f'<p class="sentiment-score">Negative: <span style="color:{"#F44336" if score > 0.2 else "#757575"}">{score:.3f}</span></p>', unsafe_allow_html=True)
            
            with col3:
                score = analysis_results['vader_scores']['neu']
                st.markdown(f'<p class="sentiment-score">Neutral: <span style="color:{"#2196F3" if score > 0.5 else "#757575"}">{score:.3f}</span></p>', unsafe_allow_html=True)
            
            with col4:
                score = analysis_results['vader_scores']['compound']
                color = "#4CAF50" if score > 0 else "#F44336" if score < 0 else "#757575"
                st.markdown(f'<p class="sentiment-score">Compound: <span style="color:{color}">{score:.3f}</span></p>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Fetch music recommendations
            recommendations = fetch_recommendations(
                analysis_results['mood'],
                analysis_results['genre'], 
                analysis_results['country'],
                analysis_results['years']
            )
            
            # Display recommendations
            st.markdown('<div class="mood-card">', unsafe_allow_html=True)
            st.markdown(f"## Music For Your {analysis_results['mood'].title()} Mood")
            
            # Display playlists if available
            if recommendations['playlists']:
                st.markdown(f'<p class="recommendation-title">{recommendations["playlist_type"]}</p>', unsafe_allow_html=True)
                for idx, playlist in enumerate(recommendations['playlists']):
                    st.markdown(
                        f"""
                        <div class="playlist-card">
                            <h3>🎵 {playlist['title']}</h3>
                            <a href="{playlist['url']}" target="_blank">Open in YouTube Music</a>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
            
            # Display songs
            if recommendations['songs']:
                st.markdown(f'<p class="recommendation-title">{recommendations["chart_type"]}</p>', unsafe_allow_html=True)
                for idx, song in enumerate(recommendations['songs']):
                    # Always create YouTube Music search URL in the format 'https://music.youtube.com/search?q=song+artist'
                    # Format search query: replace spaces with '+' and combine song title and artist
                    search_query = f"{song['title']} {song['artist']}".replace(' ', '+')
                    url = f"https://music.youtube.com/search?q={search_query}"
                    
                    url_html = f"<a href=\"{url}\" target=\"_blank\">Listen on YouTube Music</a>"
                    st.markdown(
                        f"""
                        <div class="song-card">
                            <h4>{idx+1}. {song['title']}</h4>
                            <p>Artist: {song['artist']}</p>
                            {url_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            
            # Add a note about the current date and time
            st.markdown(f"<p style='color: #888; font-size: 12px; margin-top: 20px;'>Analysis completed on {time.strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
    else:
        st.warning("Please enter your feelings before submitting.")