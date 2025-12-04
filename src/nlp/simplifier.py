import random

class PromptSimplifier:
    def optimize(self, text):
        """
        Mock logic to simulate an LLM rewriting a prompt to be more energy efficient.
        It replaces complex words with simpler ones and removes fluff.
        """
        if not text:
            return ""

        # Dictionary of "expensive" tokens vs "efficient" tokens
        # In a real app, this would be a T5 or GPT-2 model.
        replacements = {
            "utilize": "use",
            "facilitate": "help",
            "in order to": "to",
            "demonstrate": "show",
            "approximately": "about",
            "subsequently": "then",
            "please": "", # Politeness tokens add compute cost!
            "could you": "",
            "kindly": "",
            "basically": "",
            "actually": ""
        }
        
        improved_text = text.lower()
        for complex_word, simple_word in replacements.items():
            improved_text = improved_text.replace(complex_word, simple_word)
            
        # Mock truncation for efficiency
        words = improved_text.split()
        if len(words) > 20:
             # Keep first 80% of words for specific efficiency
            cutoff = int(len(words) * 0.8)
            words = words[:cutoff]
            
        return " ".join(words).strip().capitalize()