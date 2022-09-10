from gtts import gTTS
from playsound import playsound
from io import BytesIO

import uuid, os

def speak(text: str, lang="en"):
    if text == "" or text is None:
        return

    tts = gTTS(text, lang=lang, tld="com")
    filename = f"/tmp/{uuid.uuid4()}.mp3"
    tts.save(filename)
    
    playsound(filename)
    print("audio end")
    os.remove(filename)
