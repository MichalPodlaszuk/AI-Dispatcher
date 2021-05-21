import speech_recognition as sr
from gtts import gTTS
import random
from playsound import playsound
import os


def speech_recognizer():
    print('talk now')
    r = sr.Recognizer()
    recording = sr.Microphone()
    with recording as source:
        audio = r.listen(source)
    output = r.recognize_google(audio)
    print(output)
    return output


def speak(audio_string):
    tts = gTTS(text=audio_string, lang='en') # text to speech(voice)
    r = random.randint(1,20000000)
    audio_file = 'audio' + str(r) + '.mp3'
    tts.save(audio_file)
    playsound(audio_file)
    print(f"kiri: {audio_string}")
    os.remove(audio_file)