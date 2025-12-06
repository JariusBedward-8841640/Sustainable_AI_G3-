"""
Semantic Similarity Module

This module provides semantic similarity computation between prompts using
sentence embeddings. It validates that optimized prompts maintain the
original meaning while being more efficient.

Features:
- Sentence embeddings using sentence-transformers or TF-IDF fallback
- Cosine similarity computation
- Semantic clustering for prompt alternatives
- Meaning preservation validation

Author: Sustainable AI Team
"""

import os
import sys
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
from dataclasses import dataclass
import warnings

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)


@dataclass
class SimilarityResult:
    """Data class for similarity computation results."""
    score: float
    interpretation: str
    details: Dict[str, float]


class SemanticSimilarity:
    """
    Semantic Similarity Calculator for prompt optimization validation.
    
    This class provides methods to:
    - Compute semantic similarity between text pairs
    - Validate meaning preservation in optimized prompts
    - Find semantically similar alternatives
    """
    
    # Similarity thresholds
    HIGH_SIMILARITY_THRESHOLD = 0.8
    MEDIUM_SIMILARITY_THRESHOLD = 0.6
    LOW_SIMILARITY_THRESHOLD = 0.4
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", use_gpu: bool = False):
        """
        Initialize the Semantic Similarity calculator.
        
        Args:
            model_name: Name of the sentence-transformer model to use
            use_gpu: Whether to use GPU for embeddings (if available)
        """
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.model = None
        self._tfidf_vectorizer = None
        self._use_fallback = False
        
        # Try to load sentence-transformers
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the sentence transformer model with fallback to TF-IDF."""
        try:
            from sentence_transformers import SentenceTransformer
            device = 'cuda' if self.use_gpu else 'cpu'
            self.model = SentenceTransformer(self.model_name, device=device)
            print(f"Loaded sentence-transformer model: {self.model_name}")
            self._use_fallback = False
        except ImportError:
            warnings.warn(
                "sentence-transformers not installed. Using TF-IDF fallback. "
                "Install with: pip install sentence-transformers"
            )
            self._use_fallback = True
            self._initialize_tfidf()
        except Exception as e:
            warnings.warn(f"Failed to load model: {e}. Using TF-IDF fallback.")
            self._use_fallback = True
            self._initialize_tfidf()
            
    def _initialize_tfidf(self):
        """Initialize TF-IDF vectorizer as fallback."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            print("Using TF-IDF fallback for similarity computation")
        except ImportError:
            warnings.warn("sklearn not installed. Similarity will use basic overlap.")
            self._tfidf_vectorizer = None
    
    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get embedding vector for a text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Numpy array representing the text embedding
        """
        if not text or not text.strip():
            return np.zeros(384)  # Default embedding size
            
        if self._use_fallback:
            return self._get_tfidf_embedding(text)
        else:
            return self.model.encode(text, convert_to_numpy=True)
    
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            2D numpy array of embeddings
        """
        if self._use_fallback:
            return np.array([self._get_tfidf_embedding(t) for t in texts])
        else:
            return self.model.encode(texts, convert_to_numpy=True)
    
    def _get_tfidf_embedding(self, text: str) -> np.ndarray:
        """Get TF-IDF embedding for text (fallback method)."""
        if self._tfidf_vectorizer is None:
            return self._get_word_overlap_vector(text)
            
        try:
            # Fit on single text and transform
            vector = self._tfidf_vectorizer.fit_transform([text]).toarray()[0]
            # Pad or truncate to standard size
            target_size = 384
            if len(vector) < target_size:
                vector = np.pad(vector, (0, target_size - len(vector)))
            else:
                vector = vector[:target_size]
            return vector
        except Exception:
            return self._get_word_overlap_vector(text)
    
    def _get_word_overlap_vector(self, text: str) -> np.ndarray:
        """Simple word-based vector (basic fallback)."""
        words = text.lower().split()
        # Create simple bag of words hash
        vector = np.zeros(384)
        for i, word in enumerate(words[:384]):
            vector[hash(word) % 384] += 1
        if np.linalg.norm(vector) > 0:
            vector = vector / np.linalg.norm(vector)
        return vector
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def compute_similarity(self, text1: str, text2: str) -> SimilarityResult:
        """
        Compute semantic similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            SimilarityResult with score and interpretation
        """
        if not text1 or not text2:
            return SimilarityResult(
                score=0.0,
                interpretation="Invalid input",
                details={"cosine": 0.0, "word_overlap": 0.0, "keyword_match": 0.0}
            )
        
        # Get embeddings
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        
        # Cosine similarity
        cosine_sim = self.cosine_similarity(emb1, emb2)
        
        # Word overlap similarity
        word_overlap = self._compute_word_overlap(text1, text2)
        
        # Keyword match score
        keyword_match = self._compute_keyword_match(text1, text2)
        
        # Combined score (weighted average)
        combined_score = (
            cosine_sim * 0.5 +
            word_overlap * 0.2 +
            keyword_match * 0.3
        )
        
        # Interpretation
        if combined_score >= self.HIGH_SIMILARITY_THRESHOLD:
            interpretation = "Highly similar - meaning well preserved"
        elif combined_score >= self.MEDIUM_SIMILARITY_THRESHOLD:
            interpretation = "Moderately similar - core meaning preserved"
        elif combined_score >= self.LOW_SIMILARITY_THRESHOLD:
            interpretation = "Somewhat similar - partial meaning preserved"
        else:
            interpretation = "Low similarity - meaning may differ"
            
        return SimilarityResult(
            score=round(combined_score, 4),
            interpretation=interpretation,
            details={
                "cosine": round(cosine_sim, 4),
                "word_overlap": round(word_overlap, 4),
                "keyword_match": round(keyword_match, 4)
            }
        )
    
    def _compute_word_overlap(self, text1: str, text2: str) -> float:
        """Compute word overlap (Jaccard similarity)."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / max(union, 1)
    
    def _compute_keyword_match(self, text1: str, text2: str) -> float:
        """Compute keyword preservation score."""
        # Stop words to ignore
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
            'up', 'about', 'into', 'through', 'during', 'before', 'after',
            'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
            'please', 'help', 'me', 'i', 'you', 'your', 'my', 'we', 'kindly'
        }
        
        # Extract keywords (non-stop words)
        keywords1 = [w.lower() for w in text1.split() if w.lower() not in stop_words and len(w) > 2]
        keywords2 = set(w.lower() for w in text2.split() if w.lower() not in stop_words and len(w) > 2)
        
        if not keywords1:
            return 1.0  # No keywords to preserve
            
        # Count preserved keywords
        preserved = sum(1 for kw in keywords1 if kw in keywords2)
        return preserved / len(keywords1)
    
    def validate_optimization(
        self, 
        original: str, 
        optimized: str,
        min_similarity: float = 0.5
    ) -> Tuple[bool, SimilarityResult]:
        """
        Validate that an optimized prompt preserves the original meaning.
        
        Args:
            original: Original prompt
            optimized: Optimized prompt
            min_similarity: Minimum acceptable similarity score
            
        Returns:
            Tuple of (is_valid, similarity_result)
        """
        result = self.compute_similarity(original, optimized)
        is_valid = result.score >= min_similarity
        
        return is_valid, result
    
    def find_similar_prompts(
        self, 
        query: str, 
        candidates: List[str],
        top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Find most similar prompts from a list of candidates.
        
        Args:
            query: Query prompt
            candidates: List of candidate prompts
            top_k: Number of top results to return
            
        Returns:
            List of (prompt, similarity_score) tuples
        """
        if not candidates:
            return []
            
        query_emb = self.get_embedding(query)
        candidate_embs = self.get_embeddings(candidates)
        
        # Compute similarities
        similarities = [
            self.cosine_similarity(query_emb, emb) 
            for emb in candidate_embs
        ]
        
        # Sort by similarity
        paired = list(zip(candidates, similarities))
        paired.sort(key=lambda x: x[1], reverse=True)
        
        return paired[:top_k]
    
    def cluster_prompts(
        self, 
        prompts: List[str], 
        n_clusters: int = 3
    ) -> Dict[int, List[str]]:
        """
        Cluster similar prompts together.
        
        Args:
            prompts: List of prompts to cluster
            n_clusters: Number of clusters
            
        Returns:
            Dictionary mapping cluster_id to list of prompts
        """
        if len(prompts) < n_clusters:
            return {0: prompts}
            
        try:
            from sklearn.cluster import KMeans
            
            embeddings = self.get_embeddings(prompts)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings)
            
            clusters = {}
            for prompt, label in zip(prompts, labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(prompt)
                
            return clusters
            
        except ImportError:
            warnings.warn("sklearn not available for clustering")
            return {0: prompts}


class EnhancedPromptValidator:
    """
    Enhanced validator combining semantic similarity with additional metrics.
    """
    
    def __init__(self):
        self.similarity = SemanticSimilarity()
        
    def validate_prompt_optimization(
        self,
        original: str,
        optimized: str
    ) -> Dict[str, Union[bool, float, str]]:
        """
        Comprehensive validation of prompt optimization.
        
        Args:
            original: Original prompt
            optimized: Optimized prompt
            
        Returns:
            Dictionary with validation results
        """
        # Semantic similarity
        is_valid, sim_result = self.similarity.validate_optimization(original, optimized)
        
        # Token analysis
        orig_tokens = len(original.split())
        opt_tokens = len(optimized.split())
        token_reduction = 1 - (opt_tokens / max(orig_tokens, 1))
        
        # Intent preservation check
        intent_preserved = self._check_intent_preservation(original, optimized)
        
        # Action word preservation
        action_preserved = self._check_action_preservation(original, optimized)
        
        # Overall quality score
        quality_score = (
            sim_result.score * 0.4 +
            (1 if intent_preserved else 0.5) * 0.3 +
            (1 if action_preserved else 0.5) * 0.3
        )
        
        return {
            "is_valid": is_valid and intent_preserved,
            "semantic_similarity": sim_result.score,
            "token_reduction_pct": round(token_reduction * 100, 1),
            "intent_preserved": intent_preserved,
            "action_preserved": action_preserved,
            "quality_score": round(quality_score, 3),
            "similarity_interpretation": sim_result.interpretation,
            "similarity_details": sim_result.details
        }
    
    def _check_intent_preservation(self, original: str, optimized: str) -> bool:
        """Check if the intent (question, command, etc.) is preserved."""
        orig_lower = original.lower()
        opt_lower = optimized.lower()
        
        # Question intent
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who', '?']
        orig_question = any(w in orig_lower for w in question_words)
        opt_question = any(w in opt_lower for w in question_words)
        
        if orig_question != opt_question:
            # Allow questions to become commands (e.g., "Can you explain?" -> "Explain")
            command_words = ['explain', 'describe', 'show', 'list', 'define', 'compare', 
                           'create', 'write', 'implement', 'fix', 'help']
            if any(w in opt_lower for w in command_words):
                return True
            return False
            
        return True
    
    def _check_action_preservation(self, original: str, optimized: str) -> bool:
        """Check if the main action/verb is preserved."""
        action_words = [
            'explain', 'describe', 'show', 'write', 'create', 'implement',
            'fix', 'debug', 'help', 'understand', 'learn', 'build',
            'design', 'analyze', 'compare', 'list', 'define', 'optimize',
            'review', 'test', 'deploy', 'configure', 'install', 'setup'
        ]
        
        orig_lower = original.lower()
        opt_lower = optimized.lower()
        
        # Find actions in original
        orig_actions = [a for a in action_words if a in orig_lower]
        
        if not orig_actions:
            return True  # No specific action to preserve
            
        # Check if any action is preserved (directly or semantically)
        for action in orig_actions:
            if action in opt_lower:
                return True
                
        # Check for semantic equivalents
        action_synonyms = {
            'explain': ['describe', 'clarify', 'define'],
            'write': ['create', 'implement', 'code'],
            'fix': ['debug', 'repair', 'solve'],
            'understand': ['learn', 'comprehend', 'grasp'],
            'show': ['demonstrate', 'display', 'present'],
            'compare': ['contrast', 'differentiate', 'vs']
        }
        
        for action in orig_actions:
            if action in action_synonyms:
                for synonym in action_synonyms[action]:
                    if synonym in opt_lower:
                        return True
                        
        return False


# --- Convenience functions ---

def compute_similarity(text1: str, text2: str) -> float:
    """Quick function to compute similarity between two texts."""
    sim = SemanticSimilarity()
    result = sim.compute_similarity(text1, text2)
    return result.score


def validate_optimization(original: str, optimized: str) -> Dict:
    """Quick validation of prompt optimization."""
    validator = EnhancedPromptValidator()
    return validator.validate_prompt_optimization(original, optimized)


if __name__ == "__main__":
    # Demo usage
    print("Semantic Similarity Module Demo")
    print("=" * 50)
    
    sim = SemanticSimilarity()
    validator = EnhancedPromptValidator()
    
    test_pairs = [
        (
            "Could you please help me understand what machine learning is?",
            "Explain machine learning."
        ),
        (
            "I was wondering if you might be able to write a Python function for me.",
            "Write Python function."
        ),
        (
            "What is recursion in programming?",
            "Explain programming recursion."
        )
    ]
    
    for original, optimized in test_pairs:
        print(f"\nOriginal: {original}")
        print(f"Optimized: {optimized}")
        
        result = sim.compute_similarity(original, optimized)
        print(f"Similarity: {result.score:.2%}")
        print(f"Interpretation: {result.interpretation}")
        
        validation = validator.validate_prompt_optimization(original, optimized)
        print(f"Valid: {validation['is_valid']}")
        print(f"Quality Score: {validation['quality_score']:.2%}")
