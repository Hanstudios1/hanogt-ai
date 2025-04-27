# sesli_asistan.py

import speech_recognition as sr
import pyttsx3

def listen_to_microphone():
    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        print("Dinliyorum...")
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio, language="tr-TR")
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        return None

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()