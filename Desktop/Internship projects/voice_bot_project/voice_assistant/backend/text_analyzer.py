import spacy

class TextAnalyzer:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def analyze_text(self, text):
        doc = self.nlp(text)
        return {
            "num_sentences": len(list(doc.sents)),
            "num_tokens": len(doc),
            "num_nouns": len([token for token in doc if token.pos_ == "NOUN"]),
            "num_verbs": len([token for token in doc if token.pos_ == "VERB"]),
            "grammar_issues": self.find_grammar_issues(doc)
        }

    def find_grammar_issues(self, doc):
        issues = []
        for sent in doc.sents:
            if len(sent) < 4:
                issues.append(f"Sentence too short: '{sent.text.strip()}'")
            if sent[-1].text not in [".", "!", "?"]:
                issues.append(f"Missing punctuation: '{sent.text.strip()}'")
        return issues