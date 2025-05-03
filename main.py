# main.py

from voice_input import get_voice_input
import re
import geocoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ytmusicapi import YTMusic
from fuzzywuzzy import process

analyzer = SentimentIntensityAnalyzer()
ytmusic = YTMusic()

# Expanded mood keywords
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
    match = re.search(r'(\d{1,2})\s*(year|years)', text.lower())
    if match:
        years = int(match.group(1))
        if years in [1, 5, 10, 20]:
            return years
    return 1  # default

def get_user_country():
    try:
        g = geocoder.ip('me')
        country = g.country
        if country:
            return country
    except:
        pass
    return 'IN'  # Default to India instead of falling back

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

def get_bollywood_charts():
    """
    Get popular Bollywood playlists
    """
    try:
        # Use search API to find popular Bollywood playlists
        results = ytmusic.search("bollywood top songs", filter="playlists")
        print("\n🎵 Bollywood Top Charts:")
        for i in results[:5]:
            if 'playlistId' in i:
                print(f"- {i['title']} — https://music.youtube.com/playlist?list={i['playlistId']}")
        return results
    except Exception as e:
        print(f"Error getting Bollywood charts: {e}")
        return []

def fetch_playlists(mood, genre, country, years):
    """
    Improved playlist fetching with better error handling and multiple fallbacks
    to ensure playlists are always returned.
    """
    found_playlists = False
    
    # Special case: If genre is Bollywood or country is India, prioritize Hindi music
    is_india = country == 'IN'
    is_bollywood = genre and 'bollywood' in genre.lower()
    
    # Try Bollywood first if appropriate
    if is_bollywood or is_india:
        try:
            # First try to find Bollywood/Hindi playlists related to the mood
            bollywood_query = f"bollywood {mood}" if mood else "bollywood"
            search_results = ytmusic.search(bollywood_query, filter="playlists")
            
            if search_results and len(search_results) > 0:
                print(f"\n🎧 Hindi/Bollywood Playlists for '{mood.title() if mood else 'Your Mood'}':")
                playlist_count = 0
                for i in search_results[:5]:  # Try up to 5 results
                    try:
                        if 'playlistId' in i:
                            print(f"- {i['title']} — https://music.youtube.com/playlist?list={i['playlistId']}")
                            playlist_count += 1
                            if playlist_count >= 3:  # Show at least 3 playlists
                                found_playlists = True
                                break
                    except Exception as e:
                        continue  # Skip problematic entries
                
                if playlist_count > 0:
                    found_playlists = True
                    return  # Return only if we found at least one playlist
        except Exception as e:
            print(f"Error fetching Bollywood playlists, trying alternatives: {e}")

    # Try mood with direct search (more reliable than mood categories)
    if mood and not found_playlists:
        try:
            mood_query = f"{mood} music playlist"
            search_results = ytmusic.search(mood_query, filter="playlists")
            
            if search_results and len(search_results) > 0:
                print(f"\n🎧 Mood Playlists for '{mood.title()}':")
                playlist_count = 0 
                for i in search_results[:5]:
                    try:
                        if 'playlistId' in i:
                            print(f"- {i['title']} — https://music.youtube.com/playlist?list={i['playlistId']}")
                            playlist_count += 1
                            if playlist_count >= 3:
                                found_playlists = True
                                break
                    except Exception as e:
                        continue
                
                if playlist_count > 0:
                    found_playlists = True
                    return
        except Exception as e:
            print(f"Error fetching mood playlists via search, trying alternatives: {e}")

    # Use mood categories if direct search didn't work
    if mood and not found_playlists:
        try:
            moods = ytmusic.get_mood_categories()
            for cat in moods.get('Moods & moments', []):
                if mood.lower() in cat['title'].lower():
                    pl = ytmusic.get_mood_playlists(cat['params'])
                    if pl and len(pl) > 0:
                        print(f"\n🎧 Mood Playlists for '{mood.title()}':")
                        playlist_count = 0
                        for i in pl[:5]:
                            try:
                                if 'playlistId' in i:
                                    print(f"- {i['title']} — https://music.youtube.com/playlist?list={i['playlistId']}")
                                    playlist_count += 1
                                    if playlist_count >= 3:
                                        found_playlists = True
                                        break
                            except Exception as e:
                                continue
                        
                        if playlist_count > 0:
                            found_playlists = True
                            return
        except Exception as e:
            print(f"Error fetching mood category playlists, trying alternatives: {e}")

    # Try genre with direct search
    if genre and not found_playlists:
        try:
            genre_query = f"{genre} music playlist"
            search_results = ytmusic.search(genre_query, filter="playlists")
            
            if search_results and len(search_results) > 0:
                print(f"\n🎧 Genre Playlists for '{genre.title()}':")
                playlist_count = 0
                for i in search_results[:5]:
                    try:
                        if 'playlistId' in i:
                            print(f"- {i['title']} — https://music.youtube.com/playlist?list={i['playlistId']}")
                            playlist_count += 1
                            if playlist_count >= 3:
                                found_playlists = True
                                break
                    except Exception as e:
                        continue
                
                if playlist_count > 0:
                    found_playlists = True
                    return
        except Exception as e:
            print(f"Error fetching genre playlists via search, trying alternatives: {e}")

    # Try genre categories if direct search didn't work
    if genre and not found_playlists:
        try:
            genres = ytmusic.get_mood_categories()
            for cat in genres.get('Genres', []):
                if genre.lower() in cat['title'].lower():
                    pl = ytmusic.get_mood_playlists(cat['params'])
                    if pl and len(pl) > 0:
                        print(f"\n🎧 Genre Playlists for '{genre.title()}':")
                        playlist_count = 0
                        for i in pl[:5]:
                            try:
                                if 'playlistId' in i:
                                    print(f"- {i['title']} — https://music.youtube.com/playlist?list={i['playlistId']}")
                                    playlist_count += 1
                                    if playlist_count >= 3:
                                        found_playlists = True
                                        break
                            except Exception as e:
                                continue
                        
                        if playlist_count > 0:
                            found_playlists = True
                            return
        except Exception as e:
            print(f"Error fetching genre category playlists, trying alternatives: {e}")

    # Try country-specific charts
    if not found_playlists:
        try:
            charts = ytmusic.get_charts(country=country)
            if 'songs' in charts and 'items' in charts['songs'] and len(charts['songs']['items']) > 0:
                print(f"\n🎶 Top Charts in {country} for past {years} year(s):")
                song_count = 0
                for i in charts['songs']['items'][:5]:
                    try:
                        title = i['title']
                        artist = ', '.join([a['name'] for a in i['artists']])
                        print(f"- {title} by {artist}")
                        song_count += 1
                        if song_count >= 3:
                            found_playlists = True
                            break
                    except Exception as e:
                        continue
                
                if song_count > 0:
                    found_playlists = True
                    return
        except Exception as e:
            print(f"Error fetching country charts, trying alternatives: {e}")

    # Try bollywood charts as a special fallback for Indian users
    if not found_playlists and (is_india or is_bollywood):
        try:
            results = ytmusic.search("bollywood top songs", filter="playlists")
            if results and len(results) > 0:
                print("\n🎵 Bollywood Top Charts:")
                playlist_count = 0
                for i in results[:5]:
                    try:
                        if 'playlistId' in i:
                            print(f"- {i['title']} — https://music.youtube.com/playlist?list={i['playlistId']}")
                            playlist_count += 1
                            if playlist_count >= 3:
                                found_playlists = True
                                break
                    except Exception as e:
                        continue
                
                if playlist_count > 0:
                    found_playlists = True
                    return
        except Exception as e:
            print(f"Error fetching Bollywood chart fallback, trying alternatives: {e}")

    # General mood playlists as fallback
    if not found_playlists:
        try:
            general_moods = ["happy", "sad", "chill", "energetic", "focus"]
            # Choose a mood that's closest to the user's mood, or use a default one
            fallback_mood = mood if mood in general_moods else "chill"
            
            search_results = ytmusic.search(f"{fallback_mood} playlist", filter="playlists")
            if search_results and len(search_results) > 0:
                print(f"\n🎧 Recommended Playlists (Fallback):")
                playlist_count = 0
                for i in search_results[:5]:
                    try:
                        if 'playlistId' in i:
                            print(f"- {i['title']} — https://music.youtube.com/playlist?list={i['playlistId']}")
                            playlist_count += 1
                            if playlist_count >= 3:
                                found_playlists = True
                                break
                    except Exception as e:
                        continue
                
                if playlist_count > 0:
                    found_playlists = True
                    return
        except Exception as e:
            print(f"Error fetching general mood fallback, trying US charts: {e}")

    # Final fallback: US charts
    if not found_playlists:
        try:
            print("\n⚠️ Using fallback US charts:")
            fallback_charts = ytmusic.get_charts(country='US')
            if 'songs' in fallback_charts and 'items' in fallback_charts['songs']:
                song_count = 0
                for i in fallback_charts['songs']['items'][:5]:
                    try:
                        title = i['title']
                        artist = ', '.join([a['name'] for a in i['artists']])
                        print(f"- {title} by {artist}")
                        song_count += 1
                        if song_count >= 3:
                            found_playlists = True
                            break
                    except Exception as e:
                        continue
                
                if song_count > 0:
                    found_playlists = True
            else:
                raise Exception("No US charts available")
        except Exception as e:
            print(f"Error fetching US charts: {e}")

    # Ultimate hardcoded fallback (always show something)
    if not found_playlists:
        print("\n🎵 Recommended Playlists (Hardcoded Fallback):")
        print("- Top Global Hits — https://music.youtube.com/playlist?list=PLw-VjHDlEOgs658kAHR_LAaILBXb-s6Q5")
        print("- Today's Hits — https://music.youtube.com/playlist?list=RDCLAK5uy_nx8aCr7bJUCj4LKqJ9W1i2ejegRcZIyUM")
        print("- Chill Vibes — https://music.youtube.com/playlist?list=RDCLAK5uy_lx9KXPC6712pQn-BV4XIBaNHnKsRtQNSo")

def main():
    print("\n🎵 Welcome to Mood Playlist Generator 🎵\n")
    
    choice = input("Choose input mode (type 'voice' or 'text'): ").strip().lower()

    if choice == 'voice':
        user_input = get_voice_input()
        if not user_input:
            print("Fallback to text input.")
            user_input = input("→ Type your mood or genre: ")
    else:
        user_input = input("→ Type your mood or genre: ")

    # Improved mood detection
    vader_mood, vader_scores = detect_vader_mood(user_input)
    keyword_mood, keyword_genre = detect_mood_genre_keywords(text=user_input)
    fuzzy_mood = fuzzy_mood_match(user_input)
    
    # Get final mood using combined approach
    mood = determine_final_mood(user_input)
    genre = keyword_genre
    years = extract_time_range(user_input)
    country = get_user_country()

    print("\n🔍 Analysis Results:")
    print(f"VADER Sentiment Scores: {vader_scores}")
    if keyword_mood: 
        print(f"Keyword Mood Detection: {keyword_mood}")
    if fuzzy_mood:
        print(f"Fuzzy Match Mood: {fuzzy_mood}")
    print(f"🧠 Final Detected Mood: {mood}")
    print(f"🎼 Detected Genre: {genre if genre else 'None'}")
    print(f"🌍 Detected Country: {country}")
    print(f"🕒 Time Range: Last {years} year(s)")

    fetch_playlists(mood, genre, country, years)

if __name__ == "__main__":
    print("✅ mood_playlist_app is running!")
    print("Available mood categories:")
    for mood in mood_keywords:
        print(f"  - {mood}")
    try:
        main()
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Please try again with different input.")