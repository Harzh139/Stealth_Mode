from vosk import Model, KaldiRecognizer
import wave
import json

# Load model
model = Model("model")  # Make sure this is the correct path to your Vosk model

# Open your audio file
wf = wave.open("recordings/output.wav", "rb")
rec = KaldiRecognizer(model, wf.getframerate())

results = []

# Read and process audio
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        results.append(json.loads(rec.Result()))

# Get final result
results.append(json.loads(rec.FinalResult()))

# Print results
for r in results:
    print("🧠 Recognized:", r.get("text", "[NO TEXT]"))
