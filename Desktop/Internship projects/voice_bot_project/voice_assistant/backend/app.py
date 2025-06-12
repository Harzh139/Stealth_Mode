from flask import Flask, render_template, request, jsonify
from voice_bot import VoiceBot
import os
import json
import wave
from vosk import Model, KaldiRecognizer
import tempfile
import subprocess

app = Flask(__name__)
bot = VoiceBot()
model = Model("model")  # Vosk model directory

@app.route('/')
def index():
    return render_template('index.html', greeting=True)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return "No audio file uploaded", 400

    audio_file = request.files['audio']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_webm:
        audio_file.save(temp_webm.name)
        webm_path = temp_webm.name

    wav_path = webm_path.replace(".webm", ".wav")
    try:
        subprocess.run([
            'ffmpeg', '-y', '-loglevel', 'error',
            '-i', webm_path,
            '-ar', '16000',
            '-ac', '1',
            '-f', 'wav',
            '-acodec', 'pcm_s16le',
            wav_path
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        return jsonify({
            'original': "",
            'corrected': "",
            'feedback': f"FFmpeg error: {e.stderr.decode()}",
            'audio_path': "",
            'correctness': 0
        })

    size = os.path.getsize(wav_path)
    if size < 10000:
        return jsonify({
            'original': "",
            'corrected': "",    
            'feedback': "⚠️ Audio too short or silent. Please try again.",
            'audio_path': "",
            'correctness': 0
        })

    recording_dir = "recordings"
    os.makedirs(recording_dir, exist_ok=True)
    final_output_path = os.path.join(recording_dir, "output.wav")
    os.replace(wav_path, final_output_path)

    wf = wave.open(final_output_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    text = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            text += result.get("text", "") + " "
    final_result = json.loads(rec.FinalResult())
    text += final_result.get("text", "")
    wf.close()

    text = text.strip()
    if not text:
        return jsonify({
            'original': "",
            'corrected': "",
            'feedback': "⚠️ Could not recognize any speech. Try speaking clearly.",
            'audio_path': "",
            'correctness': 0
        })

    try:
        corrected_text, feedback, audio_filename, correctness = bot.process_audio(text)
        # Ensure correctness is int and in [0, 100]
        try:
            correctness = int(correctness)
        except Exception:
            correctness = 0
        correctness = max(0, min(100, correctness))
    except Exception as e:
        return jsonify({
            'original': text,
            'corrected': "",
            'feedback': f"Bot processing failed: {str(e)}",
            'audio_path': "",
            'correctness': 0
        })

    return jsonify({
        'original': text,
        'corrected': corrected_text,
        'feedback': feedback,
        'audio_path': '/' + audio_filename.replace("\\\\", "/"),
        'correctness': correctness
    })

if __name__ == '__main__':
    app.run(debug=True)
