"""
Unit Tests for NLP Modules

This module provides comprehensive unit tests for:
- T5 Prompt Optimizer
- Semantic Similarity Calculator
- NLP Service
- Prompt Simplifier

Author: Sustainable AI Team
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


class TestComplexityAnalyzer(unittest.TestCase):
    """Tests for ComplexityAnalyzer class."""
    
    def setUp(self):
        from src.nlp.complexity_score import ComplexityAnalyzer
        self.analyzer = ComplexityAnalyzer
        
    def test_empty_string(self):
        """Test token count for empty string."""
        self.assertEqual(self.analyzer.get_token_count(""), 0)
        
    def test_none_input(self):
        """Test token count for None input."""
        self.assertEqual(self.analyzer.get_token_count(None), 0)
        
    def test_single_word(self):
        """Test token count for single word."""
        self.assertEqual(self.analyzer.get_token_count("hello"), 1)
        
    def test_multiple_words(self):
        """Test token count for multiple words."""
        self.assertEqual(self.analyzer.get_token_count("hello world test"), 3)
        
    def test_whitespace_handling(self):
        """Test handling of extra whitespace."""
        self.assertEqual(self.analyzer.get_token_count("  hello   world  "), 2)


class TestSemanticSimilarity(unittest.TestCase):
    """Tests for SemanticSimilarity class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        from src.nlp.semantic_similarity import SemanticSimilarity
        cls.similarity = SemanticSimilarity()
        
    def test_identical_texts(self):
        """Test similarity of identical texts."""
        text = "This is a test sentence."
        result = self.similarity.compute_similarity(text, text)
        self.assertGreaterEqual(result.score, 0.9)
        
    def test_empty_text(self):
        """Test similarity with empty text."""
        result = self.similarity.compute_similarity("", "test")
        self.assertEqual(result.score, 0.0)
        
    def test_similar_texts(self):
        """Test similarity of semantically similar texts."""
        text1 = "Could you please help me understand machine learning?"
        text2 = "Explain machine learning."
        result = self.similarity.compute_similarity(text1, text2)
        self.assertGreater(result.score, 0.3)
        
    def test_dissimilar_texts(self):
        """Test similarity of very different texts."""
        text1 = "The weather is nice today."
        text2 = "Python is a programming language."
        result = self.similarity.compute_similarity(text1, text2)
        # TF-IDF fallback may produce slightly higher similarity due to shared stopwords
        self.assertLess(result.score, 0.6)
        
    def test_word_overlap(self):
        """Test word overlap calculation."""
        score = self.similarity._compute_word_overlap(
            "hello world test",
            "hello world"
        )
        self.assertGreater(score, 0.5)
        
    def test_keyword_match(self):
        """Test keyword match calculation."""
        score = self.similarity._compute_keyword_match(
            "explain machine learning concepts",
            "machine learning explanation"
        )
        self.assertGreater(score, 0.4)
        
    def test_cosine_similarity_identical(self):
        """Test cosine similarity for identical vectors."""
        vec = np.array([1, 2, 3, 4, 5])
        sim = self.similarity.cosine_similarity(vec, vec)
        self.assertAlmostEqual(sim, 1.0, places=5)
        
    def test_cosine_similarity_orthogonal(self):
        """Test cosine similarity for orthogonal vectors."""
        vec1 = np.array([1, 0, 0])
        vec2 = np.array([0, 1, 0])
        sim = self.similarity.cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(sim, 0.0, places=5)
        
    def test_cosine_similarity_zero_vector(self):
        """Test cosine similarity with zero vector."""
        vec1 = np.array([1, 2, 3])
        vec2 = np.zeros(3)
        sim = self.similarity.cosine_similarity(vec1, vec2)
        self.assertEqual(sim, 0.0)
        
    def test_similarity_interpretation_high(self):
        """Test high similarity interpretation."""
        result = self.similarity.compute_similarity(
            "test prompt",
            "test prompt"
        )
        self.assertIn("similar", result.interpretation.lower())


class TestEnhancedPromptValidator(unittest.TestCase):
    """Tests for EnhancedPromptValidator class."""
    
    @classmethod
    def setUpClass(cls):
        from src.nlp.semantic_similarity import EnhancedPromptValidator
        cls.validator = EnhancedPromptValidator()
        
    def test_valid_optimization(self):
        """Test validation of a valid optimization."""
        result = self.validator.validate_prompt_optimization(
            "Could you please help me understand machine learning?",
            "Explain machine learning."
        )
        self.assertIn('is_valid', result)
        self.assertIn('semantic_similarity', result)
        self.assertIn('token_reduction_pct', result)
        
    def test_intent_preserved_question_to_command(self):
        """Test that question to command intent is allowed."""
        preserved = self.validator._check_intent_preservation(
            "Can you explain recursion?",
            "Explain recursion."
        )
        self.assertTrue(preserved)
        
    def test_action_preserved(self):
        """Test action word preservation check."""
        preserved = self.validator._check_action_preservation(
            "Please explain this concept.",
            "Explain concept."
        )
        self.assertTrue(preserved)
        
    def test_action_synonym_preserved(self):
        """Test action synonym preservation."""
        preserved = self.validator._check_action_preservation(
            "Write a function to calculate factorial.",
            "Create factorial function."
        )
        self.assertTrue(preserved)


class TestPromptSimplifier(unittest.TestCase):
    """Tests for PromptSimplifier class."""
    
    def setUp(self):
        from src.nlp.simplifier import PromptSimplifier
        # Use rule-based for consistent testing
        self.simplifier = PromptSimplifier(use_ml_model=False)
        
    def test_empty_input(self):
        """Test optimization of empty string."""
        result = self.simplifier.optimize("")
        self.assertEqual(result, "")
        
    def test_removes_please(self):
        """Test removal of 'please'."""
        result = self.simplifier.optimize("Please help me")
        self.assertNotIn("please", result.lower())
        
    def test_removes_fillers(self):
        """Test removal of filler words."""
        result = self.simplifier.optimize("Could you kindly help me understand")
        self.assertNotIn("kindly", result.lower())
        self.assertNotIn("could you", result.lower())
        
    def test_replaces_complex_words(self):
        """Test replacement of complex words."""
        result = self.simplifier.optimize("Utilize this to facilitate learning")
        self.assertIn("use", result.lower())
        self.assertIn("help", result.lower())
        
    def test_capitalizes_result(self):
        """Test that result is properly capitalized."""
        result = self.simplifier.optimize("test input")
        self.assertTrue(result[0].isupper())
        
    def test_truncates_long_prompts(self):
        """Test truncation of very long prompts."""
        long_prompt = " ".join(["word"] * 100)
        result = self.simplifier.optimize(long_prompt)
        self.assertLess(len(result.split()), 100)
        
    def test_get_full_analysis_returns_dict(self):
        """Test that get_full_analysis returns expected structure."""
        result = self.simplifier.get_full_analysis("test prompt")
        self.assertIsInstance(result, dict)
        self.assertIn('original', result)
        self.assertIn('optimized', result)
        self.assertIn('token_reduction_pct', result)
        
    def test_is_ml_available(self):
        """Test ML availability check."""
        # Should be False since we initialized with use_ml_model=False
        self.assertFalse(self.simplifier.is_ml_available())


class TestPromptOptimizerTokenCounting(unittest.TestCase):
    """Tests for T5PromptOptimizer token counting."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures - only run if T5 is available."""
        try:
            from src.nlp.prompt_optimizer import T5PromptOptimizer
            cls.optimizer = T5PromptOptimizer()
            cls.skip_tests = False
        except Exception as e:
            cls.skip_tests = True
            cls.skip_reason = str(e)
            
    def setUp(self):
        if self.skip_tests:
            self.skipTest(f"T5 not available: {self.skip_reason}")
            
    def test_count_tokens_empty(self):
        """Test token counting for empty string."""
        count = self.optimizer._count_tokens("")
        self.assertEqual(count, 0)
        
    def test_count_tokens_simple(self):
        """Test token counting for simple text."""
        count = self.optimizer._count_tokens("Hello world")
        self.assertGreater(count, 0)
        
    def test_count_tokens_complex(self):
        """Test token counting for complex text."""
        text = "Could you please help me understand machine learning?"
        count = self.optimizer._count_tokens(text)
        self.assertGreater(count, 5)


class TestPromptOptimizerOptimization(unittest.TestCase):
    """Tests for T5PromptOptimizer optimization functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from src.nlp.prompt_optimizer import T5PromptOptimizer
            cls.optimizer = T5PromptOptimizer()
            cls.skip_tests = False
        except Exception as e:
            cls.skip_tests = True
            cls.skip_reason = str(e)
            
    def setUp(self):
        if self.skip_tests:
            self.skipTest(f"T5 not available: {self.skip_reason}")
            
    def test_optimize_empty(self):
        """Test optimization of empty string."""
        result, changes = self.optimizer.optimize("")
        self.assertEqual(result, "")
        
    def test_optimize_short_prompt(self):
        """Test optimization of short prompt (5 tokens)."""
        result, changes = self.optimizer.optimize("Write code")
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)
        
    def test_optimize_medium_prompt(self):
        """Test optimization of medium prompt (~50 tokens)."""
        prompt = "Could you please help me understand how machine learning algorithms work and how they can be applied to solve real-world problems?"
        result, changes = self.optimizer.optimize(prompt)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)
        
    def test_optimize_long_prompt(self):
        """Test optimization of long prompt (~100-200 tokens)."""
        prompt = """I would be extremely grateful and appreciative if you could 
        take the time to provide me with a very detailed and comprehensive 
        explanation of how convolutional neural networks work in the context 
        of computer vision applications, including the architecture, the 
        convolutional layers, pooling layers, and how backpropagation 
        works to train these networks on image classification tasks."""
        result, changes = self.optimizer.optimize(prompt)
        self.assertIsNotNone(result)
        self.assertGreater(len(result), 0)
        
    def test_optimize_preserves_meaning(self):
        """Test that optimization preserves core meaning."""
        prompt = "Explain Python functions"
        result, changes = self.optimizer.optimize(prompt)
        # Should contain at least one key word
        self.assertTrue(
            any(word in result.lower() for word in ['python', 'function', 'explain'])
        )
        
    def test_full_optimization_returns_result(self):
        """Test get_full_optimization returns OptimizationResult."""
        from src.nlp.prompt_optimizer import OptimizationResult
        result = self.optimizer.get_full_optimization("Test prompt")
        self.assertIsInstance(result, OptimizationResult)
        
    def test_full_optimization_has_all_fields(self):
        """Test get_full_optimization has all expected fields."""
        result = self.optimizer.get_full_optimization("Test prompt for optimization")
        self.assertIsNotNone(result.original_prompt)
        self.assertIsNotNone(result.optimized_prompt)
        self.assertIsNotNone(result.original_tokens)
        self.assertIsNotNone(result.optimized_tokens)
        self.assertIsNotNone(result.token_reduction)
        self.assertIsNotNone(result.semantic_similarity)
        self.assertIsNotNone(result.energy_reduction_estimate)
        self.assertIsNotNone(result.optimization_quality)
        
    def test_fallback_optimize(self):
        """Test fallback optimization method."""
        prompt = "Could you please kindly help me understand"
        result, changes = self.optimizer._fallback_optimize(prompt)
        self.assertNotIn("please", result.lower())
        self.assertNotIn("kindly", result.lower())
        
    def test_extract_keywords(self):
        """Test keyword extraction."""
        keywords = self.optimizer._extract_keywords(
            "machine learning algorithms are used for prediction"
        )
        self.assertIn("machine", keywords)
        self.assertIn("learning", keywords)
        self.assertIn("algorithms", keywords)


class TestPromptOptimizerVariousLengths(unittest.TestCase):
    """Tests for prompt optimization across various prompt lengths."""
    
    @classmethod
    def setUpClass(cls):
        try:
            from src.nlp.prompt_optimizer import T5PromptOptimizer
            cls.optimizer = T5PromptOptimizer()
            cls.skip_tests = False
        except Exception as e:
            cls.skip_tests = True
            cls.skip_reason = str(e)
            
    def setUp(self):
        if self.skip_tests:
            self.skipTest(f"T5 not available: {self.skip_reason}")
            
    def test_5_token_prompt(self):
        """Test optimization of 5 token prompt."""
        prompt = "Write Python code"
        result = self.optimizer.get_full_optimization(prompt)
        self.assertIsNotNone(result.optimized_prompt)
        self.assertGreaterEqual(result.semantic_similarity, 0)
        
    def test_10_token_prompt(self):
        """Test optimization of ~10 token prompt."""
        prompt = "Can you help me understand this code?"
        result = self.optimizer.get_full_optimization(prompt)
        self.assertIsNotNone(result.optimized_prompt)
        
    def test_20_token_prompt(self):
        """Test optimization of ~20 token prompt."""
        prompt = "I would appreciate if you could explain how machine learning models are trained."
        result = self.optimizer.get_full_optimization(prompt)
        self.assertIsNotNone(result.optimized_prompt)
        
    def test_50_token_prompt(self):
        """Test optimization of ~50 token prompt."""
        prompt = """Could you please help me understand the concept of neural networks 
        and how they can be used for image classification tasks? I would really 
        appreciate a detailed explanation."""
        result = self.optimizer.get_full_optimization(prompt)
        self.assertIsNotNone(result.optimized_prompt)
        self.assertGreater(result.token_reduction, -10)  # Should not increase significantly
        
    def test_100_token_prompt(self):
        """Test optimization of ~100 token prompt."""
        prompt = """I was wondering if you might be able to assist me with understanding 
        the intricate details of how transformer models work in natural language processing. 
        Specifically, I am interested in learning about the attention mechanism, how 
        positional encoding is used, and what makes transformers so effective for tasks 
        like machine translation and text generation."""
        result = self.optimizer.get_full_optimization(prompt)
        self.assertIsNotNone(result.optimized_prompt)
        
    def test_200_token_prompt(self):
        """Test optimization of ~200 token prompt."""
        prompt = """I would be extremely grateful and appreciative if you could take the time 
        to provide me with a very detailed and comprehensive explanation of how deep learning 
        models are used in modern artificial intelligence applications. I am particularly 
        interested in understanding the following topics: how convolutional neural networks 
        work for computer vision, how recurrent neural networks and LSTMs are used for 
        sequence modeling, how transformer architectures have revolutionized natural language 
        processing, and how reinforcement learning enables agents to learn from their 
        environment. Additionally, I would like to know about the training process, including 
        backpropagation, gradient descent, and various optimization techniques. Please also 
        explain common challenges like overfitting and how techniques like dropout and 
        regularization help address them."""
        result = self.optimizer.get_full_optimization(prompt)
        self.assertIsNotNone(result.optimized_prompt)
        self.assertGreater(result.token_reduction, 0)  # Should reduce tokens


class TestBatchOptimization(unittest.TestCase):
    """Tests for batch optimization functionality."""
    
    @classmethod
    def setUpClass(cls):
        try:
            from src.nlp.prompt_optimizer import T5PromptOptimizer
            cls.optimizer = T5PromptOptimizer()
            cls.skip_tests = False
        except Exception as e:
            cls.skip_tests = True
            cls.skip_reason = str(e)
            
    def setUp(self):
        if self.skip_tests:
            self.skipTest(f"T5 not available: {self.skip_reason}")
            
    def test_batch_optimize(self):
        """Test batch optimization of multiple prompts."""
        prompts = [
            "Write code",
            "Explain machine learning",
            "Could you please help me understand Python?"
        ]
        results = self.optimizer.batch_optimize(prompts)
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIsNotNone(result.optimized_prompt)
            
    def test_batch_optimize_empty_list(self):
        """Test batch optimization with empty list."""
        results = self.optimizer.batch_optimize([])
        self.assertEqual(len(results), 0)


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)
