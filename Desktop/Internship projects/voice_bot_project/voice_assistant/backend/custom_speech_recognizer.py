import os
import json
import pyaudio
from vosk import Model, KaldiRecognizer

class CustomSpeechRecognizer:
    def __init__(self, model_path="model"):
        if not os.path.exists(model_path):
            raise Exception(f"Model not found at {model_path}")
        self.model = Model(model_path)

    def recognize_speech(self):
        print("🎤 Say something to analyze your communication skills...")

        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8192)
        stream.start_stream()

        rec = KaldiRecognizer(self.model, 16000)
        final_text = ""

        while True:
            data = stream.read(4096, exception_on_overflow=False)
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()
                if text:
                    print("🗣️ You said:", text)
                    final_text = text
                    break  # Exit once valid speech is detected

        stream.stop_stream()
        stream.close()
        p.terminate()

        return final_text
