# main.py

from voice_input import get_voice_input
import re
import geocoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from ytmusicapi import YTMusic

analyzer = SentimentIntensityAnalyzer()
ytmusic = YTMusic()

mood_keywords = {
    "depressed": [
        "depressed", "numb", "hopeless", "empty", "lost", "crying", "worthless", "broken", "drowning", "done with life",
        "i feel so low", "nothing matters", "i'm not okay", "want to disappear", "tired of everything",
        "can’t take it anymore", "my heart feels heavy", "everything feels dark", "can’t feel anything", "life is meaningless"
    ],
    "sad": [
        "sad", "blue", "down", "unhappy", "melancholy", "gloomy", "teary", "feeling low", "heavy-hearted", "lonely",
        "i miss someone", "not feeling good", "it's a bad day", "feeling down", "i need a hug",
        "emotional mess", "heartbroken", "i want to cry", "feeling hopeless", "lost in thoughts"
    ],
    "happy": [
        "happy", "joyful", "cheerful", "excited", "great", "delighted", "thrilled", "on cloud nine", "overjoyed", "smiling",
        "best day ever", "feeling awesome", "everything’s perfect", "i’m glowing", "grateful heart",
        "laughing out loud", "can’t stop smiling", "loving life", "so pumped", "pure joy"
    ],
    "chill": [
        "chill", "relax", "calm", "soothing", "peaceful", "laid back", "serene", "zen", "cool vibes", "easygoing",
        "winding down", "just chilling", "slow day", "peaceful mind", "need to relax",
        "soft mood", "sunday vibes", "taking it slow", "just breathing", "mellow mood"
    ],
    "hype": [
        "hype", "energetic", "party", "pumped", "upbeat", "excited", "wild", "crazy night", "full power", "let’s goooo",
        "turn up", "ready to rock", "dance time", "feeling electric", "get lit",
        "hyped up", "supercharged", "blast the beats", "on fire", "adrenaline rush"
    ],
    "romantic": [
        "romantic", "love", "affection", "crush", "valentine", "passion", "cuddles", "date night", "sweetheart", "heartbeats",
        "thinking of you", "miss my babe", "in love", "sweet vibes", "roses and kisses",
        "i found the one", "love songs", "romance in the air", "sweet memories", "emotional love"
    ],
    "angry": [
        "angry", "mad", "furious", "irritated", "annoyed", "rage", "pissed", "losing it", "fuming", "short-tempered",
        "i’m done", "sick of it", "boiling inside", "mad as hell", "frustrated as ever",
        "nothing’s right", "get out of my way", "not today", "i need space", "can’t hold back"
    ]
}

genre_keywords = {
    "pop": ["pop", "mainstream", "top hits", "popular songs", "radio hits"],
    "rock": ["rock", "guitar", "bands", "classic rock", "alt rock"],
    "hip hop": ["hip hop", "rap", "bars", "beats", "trap", "freestyle"],
    "lofi": ["lofi", "study music", "chill beats", "focus", "low fidelity"],
    "classical": ["classical", "symphony", "orchestra", "beethoven", "mozart"],
    "jazz": ["jazz", "saxophone", "blues", "smooth jazz", "soulful"],
    "edm": ["edm", "electronic", "dance", "club music", "house", "techno"],
    "indie": ["indie", "alternative", "underground", "indie rock", "indie pop"],
    "bollywood": ["bollywood", "indian songs", "hindi music", "desi vibes", "filmy"],
    "metal": ["metal", "heavy metal", "headbang", "screamo", "thrash"]
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
    return 'IN'  # fallback

def detect_mood_genre_keywords(text):
    text_lower = text.lower()
    mood = None
    genre = None

    for m, keys in mood_keywords.items():
        if any(k in text_lower for k in keys):
            mood = m
            break

    for g, keys in genre_keywords.items():
        if any(k in text_lower for k in keys):
            genre = g
            break

    return mood, genre


def detect_vader_mood(text):
    score = analyzer.polarity_scores(text)
    comp = score['compound']
    if comp >= 0.5:
        return 'happy', score
    elif comp <= -0.5:
        return 'depressed', score
    elif 0 < comp < 0.5:
        return 'chill', score
    elif -0.5 < comp < 0:
        return 'sad', score
    else:
        return 'neutral', score


def fetch_playlists(mood, genre, country, years):
    # Try mood
    if mood:
        moods = ytmusic.get_mood_categories()
        for cat in moods.get('Moods & moments', []):
            if mood.lower() in cat['title'].lower():
                pl = ytmusic.get_mood_playlists(cat['params'])
                print(f"\n🎧 Mood Playlists for '{mood.title()}':")
                for i in pl[:3]:
                    print(f"- {i['title']} — https://music.youtube.com/playlist?list={i['playlistId']}")
                return

    # Try genre
    if genre:
        genres = ytmusic.get_mood_categories()
        for cat in genres.get('Genres', []):
            if genre.lower() in cat['title'].lower():
                pl = ytmusic.get_mood_playlists(cat['params'])
                print(f"\n🎧 Genre Playlists for '{genre.title()}':")
                for i in pl[:3]:
                    print(f"- {i['title']} — https://music.youtube.com/playlist?list={i['playlistId']}")
                return
    # Try mood
    if mood:
        moods = ytmusic.get_mood_categories()
        for cat in moods.get('Moods & moments', []):
            if mood.lower() in cat['title'].lower():
                pl = ytmusic.get_mood_playlists(cat['params'])
                print(f"\n🎧 Mood Playlists for '{mood.title()}':")
                for i in pl[:3]:
                    print(f"- {i['title']} — https://music.youtube.com/playlist?list={i['playlistId']}")
                return

    # Try genre
    if genre:
        genres = ytmusic.get_mood_categories()
        for cat in genres.get('Genres', []):
            if genre.lower() in cat['title'].lower():
                pl = ytmusic.get_mood_playlists(cat['params'])
                print(f"\n🎧 Genre Playlists for '{genre.title()}':")
                for i in pl[:3]:
                    print(f"- {i['title']} — https://music.youtube.com/playlist?list={i['playlistId']}")
                return

    # Try country-specific charts
    print(f"\n🎶 Top Charts in {country} for past {years} year(s):")
    charts = ytmusic.get_charts(country=country)
    if 'songs' in charts and 'items' in charts['songs']:
        for i in charts['songs']['items'][:5]:
            title = i['title']
            artist = ', '.join([a['name'] for a in i['artists']])
            print(f"- {title} by {artist}")
        return

    # Final fallback: US charts
    print("⚠️ No song charts available for this country. Trying fallback to US...")
    fallback_charts = ytmusic.get_charts(country='US')
    if 'songs' in fallback_charts and 'items' in fallback_charts['songs']:
        for i in fallback_charts['songs']['items'][:5]:
            title = i['title']
            artist = ', '.join([a['name'] for a in i['artists']])
            print(f"- {title} by {artist}")
        return

    # Ultimate fallback
    print("❌ No charts available. Please try again later.")

def main():
    choice = input("Choose input mode (type 'voice' or 'text'): ").strip().lower()

    if choice == 'voice':
        user_input = get_voice_input()
        if not user_input:
            print("Fallback to text input.")
            user_input = input("→ Type your mood or genre: ")
    else:
        user_input = input("→ Type your mood or genre: ")

    # (everything below this point stays as it was in your Colab code)
    vader_mood, vader_scores = detect_vader_mood(user_input)
    keyword_mood, keyword_genre = detect_mood_genre_keywords(user_input)
    mood = keyword_mood if keyword_mood else vader_mood
    genre = keyword_genre
    years = extract_time_range(user_input)
    country = get_user_country()

    print("\n🔍 VADER Sentiment Scores:", vader_scores)
    print(f"🧠 Final Detected Mood: {mood}")
    print(f"🎼 Detected Genre: {genre if genre else 'None'}")
    print(f"🌍 Detected Country: {country}")
    print(f"🕒 Time Range: Last {years} year(s)")
    print(ytmusic.get_charts())

    fetch_playlists(mood, genre, country, years)

if __name__ == "__main__":
    print("✅ main.py is running!")
    print("Available mood keywords:")
    for mood in mood_keywords:
        print(f"  - {mood}")
main()

