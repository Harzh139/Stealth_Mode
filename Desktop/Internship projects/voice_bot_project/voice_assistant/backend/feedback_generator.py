import spacy
import re
from difflib import SequenceMatcher

class FeedbackGenerator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        print("🧠 Feedback Generator Initialized")

    def generate(self, original_text, corrected_text):
        corrected_text = self.clean_repeats(corrected_text)  # ✅ Fix repeated words

        analysis_result = self.analyze_text(corrected_text)
        grammar_issues = self.detect_issues(original_text, corrected_text)
        correctness = self.calculate_correctness(original_text, corrected_text)
        
        analysis_result["grammar_issues"] = grammar_issues
        analysis_result["correctness"] = correctness
        analysis_result["corrected_text"] = corrected_text
        return self.generate_feedback(analysis_result), correctness

    def analyze_text(self, text):
        doc = self.nlp(text)
        return {
            "num_sentences": len(list(doc.sents)),
            "num_tokens": len(doc),
            "num_nouns": len([token for token in doc if token.pos_ == "NOUN"]),
            "num_verbs": len([token for token in doc if token.pos_ == "VERB"]),
        }

    @staticmethod
    def remove_emojis(text):
        return re.sub(r'[^\w\s,.!?]', '', text)

    def normalize(self, text):
        # Lowercase, remove punctuation, and normalize whitespace
        return re.sub(r'[^\w\s]', '', text.lower()).strip()

    def clean_repeats(self, text):
        # Removes repeated consecutive words like "yesterday yesterday"
        return re.sub(r'\b(\w+)( \1\b)+', r'\1', text)

    def detect_issues(self, original, corrected):
        issues = []
        if self.normalize(original) == self.normalize(corrected):
            return issues

        original_doc = self.nlp(original)
        corrected_doc = self.nlp(corrected)

        for token in original_doc:
            if token.pos_ == "VERB":
                for corr_token in corrected_doc:
                    if corr_token.pos_ == "VERB" and token.text != corr_token.text:
                        issues.append(f"Tense issue: '{token.text}' might be better as '{corr_token.text}'")

        if len(original.split()) < 4:
            issues.append("Your sentence is too short to convey clear meaning.")
        elif len(original.split()) > 20:
            issues.append("Your sentence might be too long. Consider breaking it up.")

        issues.append(f"Suggestion: Try saying — \"{corrected}\"")
        return issues

    def calculate_correctness(self, original, corrected):
        original_norm = self.normalize(original)
        corrected_norm = self.normalize(corrected)
        if not corrected_norm:
            return 0
        # Use SequenceMatcher for better accuracy
        ratio = SequenceMatcher(None, original_norm, corrected_norm).ratio()
        return round(ratio * 100)

    def generate_feedback(self, analysis_result):
        feedback = []

        feedback.append(f"You spoke {analysis_result['num_sentences']} sentence(s) with {analysis_result['num_tokens']} words.")
        feedback.append(f"Nouns: {analysis_result['num_nouns']}, Verbs: {analysis_result['num_verbs']}")
        feedback.append(f"Overall grammar correctness: {analysis_result['correctness']}%")

        if analysis_result["grammar_issues"]:
            feedback.append("Grammar issues detected:")
            for issue in analysis_result["grammar_issues"]:
                feedback.append(f"- {issue}")
        else:
            feedback.append("Excellent! No major grammar issues found.")

        corrected_text = analysis_result.get("corrected_text", "")
        feedback.append("AUDIO_SUMMARY::" + self.generate_audio_summary(
            analysis_result["correctness"],
            analysis_result["grammar_issues"],
            corrected_text
        ))

        return "\n".join(feedback)

    def generate_audio_summary(self, correctness, issues, corrected_sentence=""):
        if correctness >= 90 and not issues:
            return f"Great job! Your sentence is almost perfect. Try saying: \"{corrected_sentence}\""
        elif correctness >= 70:
            return f"Your sentence is mostly correct, about {correctness} percent accurate. Try saying: \"{corrected_sentence}\""
        elif correctness >= 40:
            return f"Your sentence is partially correct, around {correctness} percent. Let's fix grammar and structure. Try saying: \"{corrected_sentence}\""
        else:
            return f"Your sentence has several issues and needs improvement. Try saying: \"{corrected_sentence}\""
