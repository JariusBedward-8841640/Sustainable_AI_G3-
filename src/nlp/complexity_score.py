class ComplexityAnalyzer:
    @staticmethod
    def get_token_count(text: str) -> int:
        if not text:
            return 0
        return len(text.split())

    @staticmethod
    def get_avg_word_length(text: str) -> float:
        if not text:
            return 0.0
        words = text.split()
        if not words:
            return 0.0
        return sum(len(w) for w in words) / len(words)
