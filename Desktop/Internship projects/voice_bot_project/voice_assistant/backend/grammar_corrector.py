from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

class GrammarCorrector:
    def __init__(self):
        self.model_name = "vennify/t5-base-grammar-correction"
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)

    def correct_grammar(self, text):
        input_text = "grammar: " + text
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt", truncation=True)

        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_length=128, num_beams=4, early_stopping=True)

        corrected = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return corrected
