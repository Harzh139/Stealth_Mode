from custom_speech_recognizer import CustomSpeechRecognizer
from text_analyzer import TextAnalyzer
from feedback_generator import FeedbackGenerator
from speech_synthesis import SpeechSynthesizer
from grammar_corrector import GrammarCorrector 
from speech_synthesis import get_synthesizer

class VoiceBot:
    def __init__(self):
        self.synthesizer = get_synthesizer()
        self.recognizer = CustomSpeechRecognizer()
        self.corrector = GrammarCorrector()
        self.analyzer = TextAnalyzer()
        self.generator = FeedbackGenerator()

    def process_audio(self, recognized_text):
        if not recognized_text.strip():
            return "", "⚠️ Could not recognize your speech. Please try again.", "", 0

        print(f"\n🗣️ You said: {recognized_text}")

        # Step 1: Grammar correction
        corrected_text = self.corrector.correct_grammar(recognized_text)
        print(f"\n✏️ Corrected Text: {corrected_text}")

        # Step 2: Generate feedback and correctness
        feedback, correctness = self.generator.generate(recognized_text, corrected_text)
        print(f"\n💬 Feedback:\n{feedback}")

        # Step 3: Extract summary for TTS
        if "AUDIO_SUMMARY::" in feedback:
            spoken_summary = feedback.split("AUDIO_SUMMARY::")[-1].strip()
            feedback = feedback.replace(f"AUDIO_SUMMARY::{spoken_summary}", "").strip()
        else:
            spoken_summary = corrected_text  # fallback

        # Step 4: Speak the summary
        audio_filename = self.synthesizer.speak(spoken_summary)

        return corrected_text, feedback, audio_filename, correctness
