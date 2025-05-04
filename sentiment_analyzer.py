import re
import geocoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ytmusicapi import YTMusic
from fuzzywuzzy import process

# Initialize analyzers
analyzer = SentimentIntensityAnalyzer()
ytmusic = YTMusic()

# Expanded mood keywords from main(2).py
mood_keywords = {
    "depressed": [
        "depressed", "numb", "hopeless", "empty", "lost", "crying", "worthless", "broken", "drowning", "done with life",
        "i feel so low", "nothing matters", "i'm not okay", "want to disappear", "tired of everything",
        "can't take it anymore", "my heart feels heavy", "everything feels dark", "can't feel anything", "life is meaningless",
        "suicidal", "despair", "bleak", "devastated", "forsaken", "inconsolable", 
        "miserable", "despondent", "desolate", "rock bottom", "grief-stricken",
        "somber", "defeated", "anguish", "spiritless", "lifeless",
        "soul-crushing", "i've given up", "nothing feels right", "what's the point"
    ],
    "sad": [
        "sad", "blue", "down", "unhappy", "melancholy", "gloomy", "teary", "feeling low", "heavy-hearted", "lonely",
        "i miss someone", "not feeling good", "it's a bad day", "feeling down", "i need a hug",
        "emotional mess", "heartbroken", "i want to cry", "feeling hopeless", "lost in thoughts",
        "upset", "disappointed", "dismal", "hurting", "grieving", "sorrowful", 
        "weeping", "mourning", "nostalgic", "regretful", "wistful", "homesick",
        "feeling blue", "disheartened", "troubled", "heartache", "forlorn",
        "bummed out", "feeling grey", "under the weather", "not in a good place"
    ],
    "happy": [
        "happy", "joyful", "cheerful", "excited", "great", "delighted", "thrilled", "on cloud nine", "overjoyed", "smiling",
        "best day ever", "feeling awesome", "everything's perfect", "i'm glowing", "grateful heart",
        "laughing out loud", "can't stop smiling", "loving life", "so pumped", "pure joy",
        "elated", "ecstatic", "blissful", "content", "delightful", "pleasant", 
        "jubilant", "gleeful", "merry", "jolly", "carefree", "uplifted",
        "blessed", "on top of the world", "walking on sunshine", "beaming", "radiant"
    ],
    "chill": [
        "chill", "relax", "calm", "soothing", "peaceful", "laid back", "serene", "zen", "cool vibes", "easygoing",
        "winding down", "just chilling", "slow day", "peaceful mind", "need to relax",
        "soft mood", "sunday vibes", "taking it slow", "just breathing", "mellow mood",
        "tranquil", "gentle", "restful", "quiet", "composed", "untroubled", 
        "cozy", "comfortable", "at ease", "unwinding", "relaxation time", "me time",
        "mindful", "balanced", "harmony", "stress-free", "no worries", "steady"
    ],
    "hype": [
        "hype", "energetic", "party", "pumped", "upbeat", "excited", "wild", "crazy night", "full power", "let's goooo",
        "turn up", "ready to rock", "dance time", "feeling electric", "get lit",
        "hyped up", "supercharged", "blast the beats", "on fire", "adrenaline rush",
        "turnt", "lit", "amped up", "stoked", "fired up", "ready to party", 
        "full throttle", "high energy", "going hard", "feeling alive", "unstoppable",
        "bouncing off walls", "celebration", "festival vibes", "night out", "dancing"
    ],
    "romantic": [
        "romantic", "love", "affection", "crush", "valentine", "passion", "cuddles", "date night", "sweetheart", "heartbeats",
        "thinking of you", "miss my babe", "in love", "sweet vibes", "roses and kisses",
        "i found the one", "love songs", "romance in the air", "sweet memories", "emotional love",
        "infatuated", "smitten", "adore", "cherish", "intimate", "tender", 
        "dreamy", "butterflies", "enamored", "loving feelings", "heart eyes",
        "loving", "relationship", "soulmate", "significant other", "true love"
    ],
    "angry": [
        "angry", "mad", "furious", "irritated", "annoyed", "rage", "pissed", "losing it", "fuming", "short-tempered",
        "i'm done", "sick of it", "boiling inside", "mad as hell", "frustrated as ever",
        "nothing's right", "get out of my way", "not today", "i need space", "can't hold back",
        "livid", "seething", "irate", "outraged", "heated", "enraged", 
        "bitter", "indignant", "incensed", "steaming", "hot-headed", "inflamed",
        "triggered", "had enough", "exploding", "raging", "hostile", "agitated"
    ],
    "anxious": [
        "anxious", "worried", "nervous", "stressed", "overwhelmed", "panicking",
        "restless", "uneasy", "tense", "frantic", "dreading", "freaking out",
        "overthinking", "on edge", "can't relax", "butterflies", "fearing",
        "apprehensive", "troubled", "agitated", "scared", "frightened",
        "paranoid", "jittery", "jumpy", "distressed", "concerned"
    ],
    "nostalgic": [
        "nostalgic", "memories", "throwback", "good old days", "childhood", "reminiscing",
        "remember when", "missing the past", "retro", "vintage vibes", "time machine",
        "back then", "oldies", "classics", "feeling sentimental", "take me back", 
        "flashback", "old school", "golden era", "memory lane", "yearning"
    ],
    "motivated": [
        "motivated", "inspired", "determined", "focused", "productive", "driven",
        "ready to conquer", "ambitious", "goal-oriented", "disciplined", "dedicated",
        "pumped to work", "getting things done", "on a mission", "crushing it",
        "unstoppable", "resilient", "pushing forward", "hustle mode", "grind time",
        "achieving", "success mindset", "winning", "determined"
    ]
}

# Expanded genre keywords from main(2).py
genre_keywords = {
    "pop": ["pop", "mainstream", "top hits", "popular songs", "radio hits", "billboard", "charts", "trending"],
    "rock": ["rock", "guitar", "bands", "classic rock", "alt rock", "alternative", "punk", "hard rock", "grunge"],
    "hip hop": ["hip hop", "rap", "bars", "beats", "trap", "freestyle", "rhymes", "mc", "flow", "hip-hop"],
    "lofi": ["lofi", "study music", "chill beats", "focus", "low fidelity", "beats to relax", "ambient", "background"],
    "classical": ["classical", "symphony", "orchestra", "beethoven", "mozart", "concerto", "sonata", "piano", "violin"],
    "jazz": ["jazz", "saxophone", "blues", "smooth jazz", "soulful", "trumpet", "bebop", "swing", "improvisation"],
    "edm": ["edm", "electronic", "dance", "club music", "house", "techno", "dubstep", "trance", "drop", "beat"],
    "indie": ["indie", "alternative", "underground", "indie rock", "indie pop", "hipster", "folk", "acoustic"],
    "bollywood": ["bollywood", "indian songs", "hindi music", "desi vibes", "filmy", "hindi songs", "punjabi", "indian pop", "bhangra"],
    "metal": ["metal", "heavy metal", "headbang", "screamo", "thrash", "death metal", "black metal", "hardcore", "metalcore"],
    "rnb": ["rnb", "r&b", "rhythm and blues", "soul", "neo soul", "contemporary r&b", "slow jams"],
    "country": ["country", "western", "nashville", "folk", "americana", "cowboy", "southern"],
    "kpop": ["kpop", "k-pop", "korean pop", "korean", "idols", "bts", "blackpink", "twice"]
}

def extract_time_range(text):
    """Extract time range from user input"""
    match = re.search(r'(\d{1,2})\s*(year|years)', text.lower())
    if match:
        years = int(match.group(1))
        if years in [1, 5, 10, 20]:
            return years
    return 1  # default

def get_user_country():
    """Get user's country from IP"""
    try:
        g = geocoder.ip('me')
        country = g.country
        if country:
            return country
    except:
        pass
    return 'IN'  # Default to India

def detect_mood_genre_keywords(text):
    """
    Improved function to detect mood and genre using a scoring system
    """
    text_lower = text.lower()
    mood_scores = {}
    genre_scores = {}

    # Score each mood by counting keyword matches
    for m, keys in mood_keywords.items():
        mood_scores[m] = sum(1 for k in keys if k in text_lower)
    
    # Score each genre by counting keyword matches
    for g, keys in genre_keywords.items():
        genre_scores[g] = sum(1 for k in keys if k in text_lower)
    
    # Find mood and genre with highest scores, if any
    mood = max(mood_scores.items(), key=lambda x: x[1])[0] if any(mood_scores.values()) else None
    genre = max(genre_scores.items(), key=lambda x: x[1])[0] if any(genre_scores.values()) else None
    
    # Only return matches if they scored at least 1
    mood = mood if mood_scores.get(mood, 0) > 0 else None
    genre = genre if genre_scores.get(genre, 0) > 0 else None
    
    return mood, genre

def fuzzy_mood_match(text):
    """
    Use fuzzy matching to find close matches to mood keywords
    """
    # Create a list of all keywords across all moods
    all_keywords = []
    keyword_to_mood = {}
    for mood, keywords in mood_keywords.items():
        for kw in keywords:
            all_keywords.append(kw)
            keyword_to_mood[kw] = mood
    
    # Extract words from the input text
    words = text.lower().split()
    
    # For each word, find the best matching keyword
    matches = []
    for word in words:
        if len(word) > 3:  # Only consider words longer than 3 chars
            match, score = process.extractOne(word, all_keywords)
            if score > 80:  # Only accept good matches
                matches.append((match, keyword_to_mood[match], score))
    
    # Return the highest scoring mood
    if matches:
        matches.sort(key=lambda x: x[2], reverse=True)
        return matches[0][1]
    return None

def detect_vader_mood(text):
    """
    Detect mood based on VADER sentiment analysis with context awareness
    """
    # Count negation words
    negations = ['not', 'no', "don't", "doesn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't", "hadn't"]
    has_negation = any(neg in text.lower().split() for neg in negations)
    
    # Check for intensity modifiers
    intensifiers = ['very', 'extremely', 'so', 'really', 'incredibly', 'absolutely', 'totally']
    has_intensifier = any(intens in text.lower().split() for intens in intensifiers)
    
    # Get base sentiment scores
    score = analyzer.polarity_scores(text)
    
    # If there are negations but VADER didn't catch them well
    if has_negation and score['compound'] > 0:
        score['compound'] -= 0.3
    
    # If there are intensifiers, amplify the sentiment
    if has_intensifier:
        score['compound'] = score['compound'] * 1.5
        # Cap at -1 to 1
        score['compound'] = max(-1, min(1, score['compound']))
    
    # Determine mood from adjusted scores
    comp = score['compound']
    if comp >= 0.5:
        return 'happy', score
    elif comp <= -0.5:
        return 'depressed', score
    elif 0.2 < comp < 0.5:
        return 'chill', score
    elif -0.5 < comp < -0.2:
        return 'sad', score
    else:
        return 'neutral', score

def determine_final_mood(text):
    """
    Combine different mood detection methods for better accuracy
    """
    # Get results from all detection methods
    vader_mood, vader_scores = detect_vader_mood(text)
    keyword_mood, _ = detect_mood_genre_keywords(text)
    fuzzy_mood = fuzzy_mood_match(text)
    
    # If keyword detection found something specific, prioritize it
    if keyword_mood:
        # But double-check with VADER for contradictions
        if keyword_mood == "happy" and vader_scores['compound'] < -0.3:
            return "chill"  # Compromise
        elif keyword_mood == "depressed" and vader_scores['compound'] > 0.3:
            return "sad"    # Compromise
        else:
            return keyword_mood
    
    # If fuzzy matching found something, use it
    if fuzzy_mood:
        return fuzzy_mood
    
    # Otherwise fall back to VADER
    return vader_mood

def detect_mood_genre(text):
    """Main function to analyze user text and extract mood information"""
    # Analyze text using the improved methods
    vader_mood, vader_scores = detect_vader_mood(text)
    keyword_mood, keyword_genre = detect_mood_genre_keywords(text)
    fuzzy_mood = fuzzy_mood_match(text)
    
    # Use the combined approach to get a final mood
    mood = determine_final_mood(text)
    
    # Get other important information
    genre = keyword_genre
    years = extract_time_range(text)
    country = get_user_country()

    return {
        'mood': mood,
        'genre': genre,
        'years': years,
        'country': country,
        'vader_scores': vader_scores
    }

def create_search_url(title, artist):
    """Create a YouTube Music search URL from song title and artist"""
    search_query = f"{title} {artist}".replace(' ', '+')
    return f"https://music.youtube.com/search?q={search_query}"

def fetch_recommendations(mood, genre, country, years):
    """Fetch playlist or song recommendations based on mood analysis"""
    result = {
        'playlists': [],
        'songs': [],
        'playlist_type': '',
        'chart_type': ''
    }
    
    # Special case: If genre is Bollywood or country is India, prioritize Hindi music
    is_india = country == 'IN'
    is_bollywood = genre and 'bollywood' in genre.lower()
    
    # Try Bollywood first if appropriate
    if is_bollywood or is_india:
        try:
            # First try to find Bollywood/Hindi playlists related to the mood
            bollywood_query = f"bollywood {mood}" if mood else "bollywood"
            search_results = ytmusic.search(bollywood_query, filter="playlists", limit=5)
            
            if search_results and len(search_results) > 0:
                result['playlist_type'] = f"Hindi/Bollywood Playlists for '{mood.title() if mood else 'Your Mood'}'"
                for item in search_results[:3]:
                    if 'playlistId' in item:
                        result['playlists'].append({
                            'title': item.get('title', 'Unknown Playlist'),
                            'url': f"https://music.youtube.com/playlist?list={item['playlistId']}"
                        })
                
                if result['playlists']:
                    return result
        except Exception as e:
            print(f"Error fetching Bollywood playlists: {e}")
    
    # Try mood with direct search (more reliable than mood categories)
    if mood:
        try:
            mood_query = f"{mood} music playlist"
            if genre:
                mood_query = f"{genre} {mood} music playlist"
                
            search_results = ytmusic.search(mood_query, filter="playlists", limit=5)
            
            if search_results and len(search_results) > 0:
                result['playlist_type'] = f"Playlists for '{mood.title()}' {genre.title() if genre else ''} Music"
                for item in search_results[:3]:
                    if 'playlistId' in item:
                        result['playlists'].append({
                            'title': item.get('title', 'Unknown Playlist'),
                            'url': f"https://music.youtube.com/playlist?list={item['playlistId']}"
                        })
                
                if result['playlists']:
                    return result
        except Exception as e:
            print(f"Error fetching mood playlists: {e}")
    
    # Try genre with direct search if no mood playlists found
    if genre and not result['playlists']:
        try:
            genre_query = f"{genre} music playlist"
            search_results = ytmusic.search(genre_query, filter="playlists", limit=5)
            
            if search_results and len(search_results) > 0:
                result['playlist_type'] = f"{genre.title()} Music Playlists"
                for item in search_results[:3]:
                    if 'playlistId' in item:
                        result['playlists'].append({
                            'title': item.get('title', 'Unknown Playlist'),
                            'url': f"https://music.youtube.com/playlist?list={item['playlistId']}"
                        })
                
                if result['playlists']:
                    return result
        except Exception as e:
            print(f"Error fetching genre playlists: {e}")
    
    # Try country-specific charts
    try:
        charts = ytmusic.get_charts(country=country)
        if 'songs' in charts and 'items' in charts['songs'] and len(charts['songs']['items']) > 0:
            result['chart_type'] = f"Top Charts in {country} for past {years} year(s)"
            for item in charts['songs']['items'][:5]:
                try:
                    title = item['title']
                    artists = item.get('artists', [])
                    artist_names = ', '.join([a.get('name', 'Unknown Artist') for a in artists])
                    
                    song = {
                        'title': title,
                        'artist': artist_names
                    }
                    
                    # Create search URL
                    song['url'] = create_search_url(title, artist_names)
                    result['songs'].append(song)
                except Exception as e:
                    continue
            
            if result['songs']:
                return result
    except Exception as e:
        print(f"Error fetching country charts: {e}")
    
    # Fallback: mood-based song search
    try:
        mood_query = f"{mood} music" if mood else "popular music"
        if genre:
            mood_query = f"{genre} {mood} music" if mood else f"{genre} music"
            
        song_results = ytmusic.search(mood_query, filter="songs", limit=5)
        result['chart_type'] = f"Songs for '{mood_query}'"
        
        if song_results:
            for item in song_results[:5]:
                try:
                    title = item.get('title', 'Unknown Title')
                    artists = item.get('artists', [])
                    artist_names = ', '.join([a.get('name', 'Unknown Artist') for a in artists])
                    
                    song_info = {
                        'title': title,
                        'artist': artist_names
                    }
                    
                    # Always use search URL format
                    song_info['url'] = create_search_url(title, artist_names)
                    
                    result['songs'].append(song_info)
                except Exception as e:
                    continue
            
            if result['songs']:
                return result
    except Exception as e:
        print(f"Error searching for songs: {e}")
    
    # Ultimate fallback: hardcoded recommendations based on mood
    result['chart_type'] = f"Suggested {mood.title() if mood else 'Recommended'} Songs"
    mood_recommendations = {
        'happy': [
            {'title': 'Happy', 'artist': 'Pharrell Williams'},
            {'title': 'Uptown Funk', 'artist': 'Mark Ronson ft. Bruno Mars'},
            {'title': 'Can\'t Stop the Feeling!', 'artist': 'Justin Timberlake'},
            {'title': 'Good as Hell', 'artist': 'Lizzo'},
            {'title': 'Walking on Sunshine', 'artist': 'Katrina & The Waves'}
        ],
        'sad': [
            {'title': 'Someone Like You', 'artist': 'Adele'},
            {'title': 'Fix You', 'artist': 'Coldplay'},
            {'title': 'When the Party\'s Over', 'artist': 'Billie Eilish'},
            {'title': 'Tears in Heaven', 'artist': 'Eric Clapton'},
            {'title': 'Hurt', 'artist': 'Johnny Cash'}
        ],
        'chill': [
            {'title': 'Sunday Morning', 'artist': 'Maroon 5'},
            {'title': 'Waves', 'artist': 'Chill Harris'},
            {'title': 'Another Day in Paradise', 'artist': 'Quinn XCII'},
            {'title': 'Don\'t Know Why', 'artist': 'Norah Jones'},
            {'title': 'Lowkey', 'artist': 'NIKI'}
        ],
        'depressed': [
            {'title': 'Let It Be', 'artist': 'The Beatles'},
            {'title': 'Everybody Hurts', 'artist': 'R.E.M.'},
            {'title': 'Praying', 'artist': 'Kesha'},
            {'title': 'The Scientist', 'artist': 'Coldplay'},
            {'title': 'Breathe Me', 'artist': 'Sia'}
        ],
        'hype': [
            {'title': 'All I Do Is Win', 'artist': 'DJ Khaled'},
            {'title': 'Stronger', 'artist': 'Kanye West'},
            {'title': 'Eye of the Tiger', 'artist': 'Survivor'},
            {'title': 'Level Up', 'artist': 'Ciara'},
            {'title': 'Can\'t Hold Us', 'artist': 'Macklemore & Ryan Lewis'}
        ],
        'romantic': [
            {'title': 'All of Me', 'artist': 'John Legend'},
            {'title': 'Perfect', 'artist': 'Ed Sheeran'},
            {'title': 'At Last', 'artist': 'Etta James'},
            {'title': 'Just the Way You Are', 'artist': 'Bruno Mars'},
            {'title': 'I Will Always Love You', 'artist': 'Whitney Houston'}
        ],
        'angry': [
            {'title': 'Break Stuff', 'artist': 'Limp Bizkit'},
            {'title': 'Given Up', 'artist': 'Linkin Park'},
            {'title': 'Bulls on Parade', 'artist': 'Rage Against the Machine'},
            {'title': 'Back in Black', 'artist': 'AC/DC'},
            {'title': 'All Downhill From Here', 'artist': 'New Found Glory'}
        ],
        'anxious': [
            {'title': 'Breathe (2 AM)', 'artist': 'Anna Nalick'},
            {'title': 'Weightless', 'artist': 'Marconi Union'},
            {'title': 'Unsteady', 'artist': 'X Ambassadors'},
            {'title': 'Heavy', 'artist': 'Linkin Park ft. Kiiara'},
            {'title': 'Breathe Me', 'artist': 'Sia'}
        ],
        'nostalgic': [
            {'title': '1979', 'artist': 'Smashing Pumpkins'},
            {'title': 'Someday', 'artist': 'The Strokes'},
            {'title': 'Good Old Days', 'artist': 'Macklemore ft. Kesha'},
            {'title': 'In My Life', 'artist': 'The Beatles'},
            {'title': 'Summer of 69', 'artist': 'Bryan Adams'}
        ],
        'motivated': [
            {'title': 'Till I Collapse', 'artist': 'Eminem'},
            {'title': 'Believer', 'artist': 'Imagine Dragons'},
            {'title': 'Hall of Fame', 'artist': 'The Script ft. will.i.am'},
            {'title': 'Rise Up', 'artist': 'Andra Day'},
            {'title': 'Unstoppable', 'artist': 'Sia'}
        ]
    }
    
    # Default to happy if mood not found in our hardcoded list
    default_mood = 'happy' if not mood or mood not in mood_recommendations else mood
    result['songs'] = mood_recommendations[default_mood]
    
    # Add search URLs to the hardcoded recommendations
    for song in result['songs']:
        song['url'] = create_search_url(song['title'], song['artist'])
    
    return result