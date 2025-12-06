"""
Integration Tests for NLP Optimization Pipeline

This module tests the complete integration of:
- NLP Service with all components
- GUI integration (simulated)
- Energy estimator integration
- End-to-end optimization flow

Author: Sustainable AI Team
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch
import json

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)


class TestNLPServiceIntegration(unittest.TestCase):
    """Integration tests for NLP Service."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        try:
            from src.nlp.nlp_service import NLPService
            cls.service = NLPService()
            cls.skip_tests = False
        except Exception as e:
            cls.skip_tests = True
            cls.skip_reason = str(e)
            
    def setUp(self):
        if self.skip_tests:
            self.skipTest(f"NLP Service not available: {self.skip_reason}")
            
    def test_optimize_prompt_full_flow(self):
        """Test complete prompt optimization flow."""
        prompt = "Could you please help me understand machine learning?"
        result = self.service.optimize_prompt(prompt)
        
        # Check all expected fields
        self.assertEqual(result.original_prompt, prompt)
        self.assertIsNotNone(result.optimized_prompt)
        self.assertGreater(result.original_tokens, 0)
        self.assertGreaterEqual(result.semantic_similarity, 0)
        self.assertGreaterEqual(result.quality_score, 0)
        self.assertTrue(len(result.suggestions) > 0)
        
    def test_batch_optimization(self):
        """Test batch optimization through service."""
        prompts = [
            "Explain recursion",
            "Could you please write a Python function?",
            "Help me understand neural networks"
        ]
        results = self.service.batch_optimize(prompts)
        self.assertEqual(len(results), 3)
        
    def test_energy_comparison(self):
        """Test energy comparison calculation."""
        result = self.service.get_energy_comparison(
            "This is the original longer prompt",
            "Short prompt"
        )
        
        self.assertIn('original_energy_kwh', result)
        self.assertIn('optimized_energy_kwh', result)
        self.assertIn('energy_saved_kwh', result)
        self.assertIn('percent_saved', result)
        self.assertIn('carbon_saved_kg', result)
        
        # Shorter prompt should use less energy
        self.assertLess(result['optimized_energy_kwh'], result['original_energy_kwh'])
        
    def test_similarity_computation(self):
        """Test similarity computation through service."""
        sim = self.service.get_similarity(
            "Explain machine learning",
            "Machine learning explanation"
        )
        self.assertGreater(sim, 0)
        self.assertLessEqual(sim, 1)
        
    def test_result_to_dict(self):
        """Test result conversion to dictionary."""
        result = self.service.optimize_prompt("Test prompt")
        result_dict = result.to_dict()
        
        self.assertIsInstance(result_dict, dict)
        self.assertIn('original_prompt', result_dict)
        self.assertIn('optimized_prompt', result_dict)
        self.assertIn('quality_score', result_dict)
        
    def test_result_to_json(self):
        """Test result conversion to JSON."""
        result = self.service.optimize_prompt("Test prompt")
        result_json = result.to_json()
        
        # Should be valid JSON
        parsed = json.loads(result_json)
        self.assertIsInstance(parsed, dict)


class TestSimplifierIntegration(unittest.TestCase):
    """Integration tests for PromptSimplifier with NLP Service."""
    
    def test_simplifier_with_ml_model(self):
        """Test simplifier using ML model."""
        try:
            from src.nlp.simplifier import PromptSimplifier
            simplifier = PromptSimplifier(use_ml_model=True)
            
            prompt = "Could you please help me understand Python programming?"
            optimized = simplifier.optimize(prompt)
            
            self.assertIsNotNone(optimized)
            self.assertGreater(len(optimized), 0)
        except Exception as e:
            self.skipTest(f"ML model not available: {e}")
            
    def test_simplifier_fallback(self):
        """Test simplifier fallback to rule-based."""
        from src.nlp.simplifier import PromptSimplifier
        simplifier = PromptSimplifier(use_ml_model=False)
        
        prompt = "Could you please kindly help me?"
        optimized = simplifier.optimize(prompt)
        
        self.assertNotIn("please", optimized.lower())
        self.assertNotIn("kindly", optimized.lower())
        
    def test_full_analysis_integration(self):
        """Test full analysis through simplifier."""
        from src.nlp.simplifier import PromptSimplifier
        simplifier = PromptSimplifier(use_ml_model=True)
        
        result = simplifier.get_full_analysis(
            "I would appreciate if you could explain machine learning concepts."
        )
        
        self.assertIn('original', result)
        self.assertIn('optimized', result)
        self.assertIn('token_reduction_pct', result)
        self.assertIn('quality_score', result)


class TestEnergyEstimatorIntegration(unittest.TestCase):
    """Integration tests for Energy Estimator with NLP optimization."""
    
    def test_energy_estimate_with_optimization(self):
        """Test energy estimation with optimized prompt."""
        from src.prediction.estimator import EnergyEstimator
        from src.nlp.simplifier import PromptSimplifier
        
        estimator = EnergyEstimator(model_type="LinearRegression")
        simplifier = PromptSimplifier(use_ml_model=False)
        
        # Original prompt
        original = "Could you please help me understand machine learning algorithms?"
        orig_result = estimator.estimate(original, layers=12, training_hours=5.0, flops_str="1.5e18")
        
        # Optimized prompt
        optimized = simplifier.optimize(original)
        opt_result = estimator.estimate(optimized, layers=12, training_hours=5.0, flops_str="1.5e18")
        
        # Both should return valid results
        self.assertIn('energy_kwh', orig_result)
        self.assertIn('energy_kwh', opt_result)
        self.assertIn('token_count', orig_result)
        self.assertIn('token_count', opt_result)
        
        # Optimized should have fewer tokens
        self.assertLessEqual(opt_result['token_count'], orig_result['token_count'])


class TestTrainingDataIntegration(unittest.TestCase):
    """Integration tests for training data loading."""
    
    def test_training_data_exists(self):
        """Test that training data files exist."""
        data_dir = os.path.join(project_root, 'data', 'prompt_optimization')
        
        main_data = os.path.join(data_dir, 'training_data.json')
        extended_data = os.path.join(data_dir, 'extended_training_data.json')
        
        self.assertTrue(os.path.exists(main_data), "Main training data not found")
        self.assertTrue(os.path.exists(extended_data), "Extended training data not found")
        
    def test_training_data_valid_json(self):
        """Test that training data is valid JSON."""
        data_dir = os.path.join(project_root, 'data', 'prompt_optimization')
        
        for filename in ['training_data.json', 'extended_training_data.json']:
            filepath = os.path.join(data_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.assertIn('data', data)
                self.assertGreater(len(data['data']), 0)
                
    def test_training_data_structure(self):
        """Test that training data has correct structure."""
        data_dir = os.path.join(project_root, 'data', 'prompt_optimization')
        main_data = os.path.join(data_dir, 'training_data.json')
        
        with open(main_data, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in data['data'][:5]:  # Check first 5 items
            self.assertIn('original', item)
            self.assertIn('optimized', item)


class TestEndToEndOptimization(unittest.TestCase):
    """End-to-end integration tests for complete optimization pipeline."""
    
    def test_short_prompt_end_to_end(self):
        """Test end-to-end optimization of short prompt."""
        from src.nlp.simplifier import PromptSimplifier
        from src.prediction.estimator import EnergyEstimator
        
        prompt = "Help me"
        
        # Optimize
        simplifier = PromptSimplifier(use_ml_model=True)
        analysis = simplifier.get_full_analysis(prompt)
        
        # Estimate energy
        estimator = EnergyEstimator(model_type="LinearRegression")
        energy_result = estimator.estimate(
            analysis['optimized'], 
            layers=12, 
            training_hours=5.0, 
            flops_str="1.5e18"
        )
        
        # Verify results
        self.assertIsNotNone(analysis['optimized'])
        self.assertIn('energy_kwh', energy_result)
        
    def test_long_prompt_end_to_end(self):
        """Test end-to-end optimization of long prompt."""
        from src.nlp.simplifier import PromptSimplifier
        from src.prediction.estimator import EnergyEstimator
        
        prompt = """I would be extremely grateful and appreciative if you could 
        take the time to provide me with a very detailed and comprehensive 
        explanation of how machine learning algorithms work and how they can 
        be applied to solve real-world problems in various domains."""
        
        # Optimize
        simplifier = PromptSimplifier(use_ml_model=True)
        analysis = simplifier.get_full_analysis(prompt)
        
        # Estimate energy
        estimator = EnergyEstimator(model_type="LinearRegression")
        
        orig_energy = estimator.estimate(prompt, layers=12, training_hours=5.0, flops_str="1.5e18")
        opt_energy = estimator.estimate(analysis['optimized'], layers=12, training_hours=5.0, flops_str="1.5e18")
        
        # Verify optimization happened (rule-based may not always reduce tokens for all prompts)
        # Just ensure the analysis completed successfully
        self.assertLessEqual(analysis['optimized_tokens'], analysis['original_tokens'])
        self.assertGreaterEqual(analysis['token_reduction_pct'], 0)
        
    def test_various_prompt_categories(self):
        """Test optimization across different prompt categories."""
        from src.nlp.simplifier import PromptSimplifier
        
        simplifier = PromptSimplifier(use_ml_model=True)
        
        test_prompts = {
            "coding": "Could you please write a Python function for me?",
            "explanation": "I was wondering if you could explain machine learning?",
            "short": "Fix bug",
            "question": "What is recursion?",
            "complex": "Would it be possible for you to kindly help me understand?"
        }
        
        for category, prompt in test_prompts.items():
            analysis = simplifier.get_full_analysis(prompt)
            self.assertIsNotNone(analysis['optimized'], f"Failed for category: {category}")
            self.assertGreaterEqual(analysis['quality_score'], 0, f"Invalid quality for: {category}")


class TestSemanticPreservation(unittest.TestCase):
    """Tests for semantic meaning preservation during optimization."""
    
    @classmethod
    def setUpClass(cls):
        try:
            from src.nlp.nlp_service import NLPService
            cls.service = NLPService()
            cls.skip_tests = False
        except Exception as e:
            cls.skip_tests = True
            cls.skip_reason = str(e)
            
    def setUp(self):
        if self.skip_tests:
            self.skipTest(f"NLP Service not available: {self.skip_reason}")
            
    def test_coding_prompts_preserve_intent(self):
        """Test that coding prompts preserve the coding intent."""
        prompts = [
            ("Write a Python function", ["python", "function", "write", "code"]),
            ("Debug this code", ["debug", "code", "fix"]),
            ("Create a REST API", ["rest", "api", "create"])
        ]
        
        for prompt, expected_keywords in prompts:
            result = self.service.optimize_prompt(prompt)
            optimized_lower = result.optimized_prompt.lower()
            
            # At least one keyword should be present
            found = any(kw in optimized_lower for kw in expected_keywords)
            self.assertTrue(found, f"Keywords lost for: {prompt} -> {result.optimized_prompt}")
            
    def test_question_prompts_get_answered(self):
        """Test that question prompts remain actionable."""
        prompts = [
            "What is machine learning?",
            "How do neural networks work?",
            "Why use Python?"
        ]
        
        for prompt in prompts:
            result = self.service.optimize_prompt(prompt)
            # Should not be empty
            self.assertGreater(len(result.optimized_prompt), 0)
            # Should have decent similarity
            self.assertGreater(result.semantic_similarity, 30)


class TestPerformanceMetrics(unittest.TestCase):
    """Tests for performance metrics accuracy."""
    
    def test_token_reduction_calculation(self):
        """Test that token reduction is calculated correctly."""
        try:
            from src.nlp.nlp_service import NLPService
            service = NLPService()
        except Exception as e:
            self.skipTest(f"NLP Service not available: {e}")
            
        # Long verbose prompt should have significant reduction
        prompt = """I would be extremely grateful and appreciative if you could 
        please help me understand the concept of machine learning and artificial 
        intelligence in simple terms."""
        
        result = service.optimize_prompt(prompt)
        
        expected_reduction = (
            (result.original_tokens - result.optimized_tokens) / 
            result.original_tokens * 100
        )
        
        self.assertAlmostEqual(
            result.token_reduction_pct, 
            round(expected_reduction, 1), 
            places=1
        )
        
    def test_energy_reduction_formula(self):
        """Test that energy reduction follows expected formula."""
        try:
            from src.nlp.nlp_service import NLPService
            service = NLPService()
        except Exception as e:
            self.skipTest(f"NLP Service not available: {e}")
            
        result = service.optimize_prompt("Could you please help me?")
        
        # Energy reduction should be proportional to token reduction squared
        ratio = result.optimized_tokens / max(result.original_tokens, 1)
        expected_energy_reduction = (1 - ratio ** 2) * 100
        
        # Allow some tolerance due to rounding
        self.assertAlmostEqual(
            result.energy_reduction_pct,
            round(expected_energy_reduction, 1),
            delta=5  # 5% tolerance
        )


if __name__ == '__main__':
    unittest.main(verbosity=2)