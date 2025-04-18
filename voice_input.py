# voice_input.py

import speech_recognition as sr

def get_voice_input():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Speak now (max 10 sec)...")
        audio = recognizer.listen(source, phrase_time_limit=10)
    try:
        print("🔄 Recognizing...")
        text = recognizer.recognize_google(audio)
        print("✅ You said:", text)
        return text
    except sr.UnknownValueError:
        print("❌ Sorry, could not understand your voice.")
        return None
    except sr.RequestError:
        print("❌ Voice service not available.")
        return None
