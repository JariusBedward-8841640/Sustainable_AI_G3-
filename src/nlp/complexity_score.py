class ComplexityAnalyzer:
    @staticmethod
    def get_token_count(text):
        """
        Simple whitespace tokenizer for demonstration.
        In production, replace with HuggingFace tokenizer.
        """
        if not text:
            return 0
        return len(text.split())