import unicodedata
import os
from TTS.api import TTS

def get_synthesizer():
    tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False, gpu=False)
    return SpeechSynthesizer(tts_model)

def remove_unrecognized_chars(text):
    return ''.join(c for c in text if unicodedata.category(c)[0] != "C" and ord(c) != 0x0361)

def remove_emojis(text):
    return ''.join(c for c in text if c.isascii())

class SpeechSynthesizer:
    def __init__(self, tts_model):
        self.tts = tts_model

    def speak(self, text):
        text = remove_emojis(text.strip())
        text = remove_unrecognized_chars(text)

        if len(text.split()) < 2:
            text = "Please speak a little more so I can help you better."

        os.makedirs("static/audio", exist_ok=True)
        file_path = os.path.join("static/audio", "output.mp3")
        self.tts.tts_to_file(text=text, file_path=file_path)
        return file_path
