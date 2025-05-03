# voice_input.py

import speech_recognition as sr

def get_voice_input():
    """
    Get voice input from the user using the microphone
    
    Returns:
        str: The recognized text from the user's voice input
             None if recognition fails
    """
    recognizer = sr.Recognizer()
    
    # Adjust for ambient noise
    try:
        with sr.Microphone() as source:
            print("🔊 Adjusting for ambient noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("🎙️ Speak now (max 10 sec)...")
            audio = recognizer.listen(source, phrase_time_limit=10)
    except sr.RequestError as e:
        print(f"❌ Error with speech recognition service: {e}")
        return None
    except Exception as e:
        print(f"❌ Error accessing microphone: {e}")
        print("Make sure your microphone is connected and working.")
        return None
    
    # Try to recognize using multiple APIs
    try:
        print("🔄 Recognizing...")
        # Try Google first
        try:
            text = recognizer.recognize_google(audio)
            print("✅ You said:", text)
            return text
        except sr.UnknownValueError:
            # If Google fails, try other recognizers
            try:
                print("🔄 Trying alternative recognition...")
                text = recognizer.recognize_sphinx(audio)
                print("✅ You said:", text)
                return text
            except:
                print("❌ Sorry, could not understand your voice.")
                return None
    except sr.RequestError as e:
        print(f"❌ Voice service error: {e}")
        print("Please check your internet connection.")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None