from gtts import gTTS
from playsound import playsound
from io import BytesIO

import uuid, os

def speak(text: str, lang="en"):
    tts = gTTS(text, lang=lang)
    filename = f"/tmp/{uuid.uuid4()}.mp3"
    tts.save(filename)
    
    playsound(filename)
    os.remove(filename)
