# voice_bot.py

from custom_speech_recognizer import CustomSpeechRecognizer
from text_analyzer import TextAnalyzer
from feedback_generator import FeedbackGenerator
from speech_synthesis import SpeechSynthesizer
from grammar_corrector import GrammarCorrector 

def main():
    print("🎤 Say something to analyze your communication skills...")

    # Step 1: Record speech and convert to text
    recognizer = CustomSpeechRecognizer()
    recognized_text = recognizer.recognize_speech()
    print(f"\n You said: {recognized_text}")

    # Step 2: Correct grammar
    corrector = GrammarCorrector()
    corrected_text = corrector.correct_grammar(recognized_text)
    print(f"\n Corrected Text: {corrected_text}")

    # Step 3: Analyze the corrected text
    analyzer = TextAnalyzer()   
    analysis_result = analyzer.analyze_text(corrected_text)

    # Step 4: Generate feedback
    generator = FeedbackGenerator()
    feedback = generator.generate_feedback(analysis_result)
    print(f"\n Feedback:\n{feedback}")

    # Step 5: Speak out the corrected version
    synthesizer = SpeechSynthesizer()
    synthesizer.speak(corrected_text)

if __name__ == "__main__":
    main()
