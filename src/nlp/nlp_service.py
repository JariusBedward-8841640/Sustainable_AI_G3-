"""
NLP Service - Unified Interface for Prompt Optimization

This module provides a unified interface for all NLP operations including:
- Prompt optimization using T5 model
- Semantic similarity validation
- Energy reduction estimation
- Comprehensive prompt analysis

Author: Sustainable AI Team
"""

import os
import sys
from typing import Dict, Optional, List, Union
from dataclasses import dataclass, asdict
import json

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.nlp.prompt_optimizer import T5PromptOptimizer, OptimizationResult
from src.nlp.semantic_similarity import SemanticSimilarity, EnhancedPromptValidator
from src.nlp.complexity_score import ComplexityAnalyzer


@dataclass
class ComprehensiveOptimizationResult:
    """Complete optimization result with all metrics."""
    # Original prompt info
    original_prompt: str
    original_tokens: int
    original_complexity: str
    
    # Optimized prompt info
    optimized_prompt: str
    optimized_tokens: int
    
    # Reduction metrics
    token_reduction_pct: float
    energy_reduction_pct: float
    
    # Semantic validation
    semantic_similarity: float
    similarity_interpretation: str
    meaning_preserved: bool
    
    # Quality assessment
    optimization_quality: str
    quality_score: float
    
    # Suggestions
    suggestions: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


class NLPService:
    """
    Unified NLP Service for prompt optimization and analysis.
    
    This service combines:
    - T5-based prompt optimization
    - Semantic similarity validation
    - Complexity analysis
    - Energy estimation
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the NLP Service.
        
        Args:
            model_path: Optional path to a custom T5 model
        """
        print("Initializing NLP Service...")
        
        # Initialize components
        self.optimizer = T5PromptOptimizer(model_path=model_path)
        self.similarity = SemanticSimilarity()
        self.validator = EnhancedPromptValidator()
        self.complexity = ComplexityAnalyzer()
        
        # Store T5 status for external access
        self.t5_status = self.optimizer.get_status()
        
        if self.t5_status["t5_available"]:
            print(f"NLP Service initialized with T5 model ({self.t5_status['device']})")
        else:
            print(f"⚠️ NLP Service initialized (T5 unavailable - using rule-based fallback)")
        
    def get_optimizer_status(self) -> dict:
        """
        Get the current status of the T5 optimizer.
        
        Returns:
            Dictionary with t5_available, is_trained, device, and status_message
        """
        return self.optimizer.get_status()
        
    def optimize_prompt(self, prompt: str) -> ComprehensiveOptimizationResult:
        """
        Optimize a prompt with full analysis.
        
        Args:
            prompt: The prompt to optimize
            
        Returns:
            ComprehensiveOptimizationResult with all metrics
        """
        if not prompt or not prompt.strip():
            return self._empty_result(prompt)
        
        # Get optimization from T5
        opt_result = self.optimizer.get_full_optimization(prompt)
        
        # Validate semantic similarity
        validation = self.validator.validate_prompt_optimization(
            prompt, 
            opt_result.optimized_prompt
        )
        
        # Assess complexity
        original_complexity = self._assess_complexity(prompt)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(prompt, opt_result, validation)
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(opt_result, validation)
        
        return ComprehensiveOptimizationResult(
            original_prompt=prompt,
            original_tokens=opt_result.original_tokens,
            original_complexity=original_complexity,
            optimized_prompt=opt_result.optimized_prompt,
            optimized_tokens=opt_result.optimized_tokens,
            token_reduction_pct=opt_result.token_reduction,
            energy_reduction_pct=opt_result.energy_reduction_estimate,
            semantic_similarity=validation['semantic_similarity'] * 100,
            similarity_interpretation=validation['similarity_interpretation'],
            meaning_preserved=validation['is_valid'],
            optimization_quality=opt_result.optimization_quality,
            quality_score=quality_score,
            suggestions=suggestions
        )
    
    def _empty_result(self, prompt: str) -> ComprehensiveOptimizationResult:
        """Return empty result for invalid input."""
        return ComprehensiveOptimizationResult(
            original_prompt=prompt or "",
            original_tokens=0,
            original_complexity="None",
            optimized_prompt="",
            optimized_tokens=0,
            token_reduction_pct=0.0,
            energy_reduction_pct=0.0,
            semantic_similarity=0.0,
            similarity_interpretation="Invalid input",
            meaning_preserved=False,
            optimization_quality="N/A",
            quality_score=0.0,
            suggestions=["Please enter a valid prompt."]
        )
    
    def _assess_complexity(self, text: str) -> str:
        """Assess the complexity of a prompt."""
        tokens = self.complexity.get_token_count(text)
        word_count = len(text.split())
        avg_word_length = sum(len(w) for w in text.split()) / max(word_count, 1)
        
        # Check for complex patterns
        complex_indicators = [
            'could you please', 'would you mind', 'i was wondering',
            'would it be possible', 'i would appreciate',
            'in order to', 'the fact that', 'due to the fact'
        ]
        has_complex_patterns = any(p in text.lower() for p in complex_indicators)
        
        if tokens > 50 or has_complex_patterns:
            return "High"
        elif tokens > 20:
            return "Medium"
        elif tokens > 5:
            return "Low"
        else:
            return "Minimal"
    
    def _generate_suggestions(
        self, 
        original: str, 
        opt_result: OptimizationResult,
        validation: Dict
    ) -> List[str]:
        """Generate suggestions based on optimization results."""
        suggestions = []
        
        # Token-based suggestions
        if opt_result.token_reduction < 10:
            suggestions.append("✓ Prompt is already quite efficient.")
        elif opt_result.token_reduction > 50:
            suggestions.append(f"✓ Great reduction! {opt_result.token_reduction}% fewer tokens.")
            
        # Similarity-based suggestions
        if validation['semantic_similarity'] < 0.5:
            suggestions.append("⚠ Review optimized prompt - meaning may have changed.")
        elif validation['semantic_similarity'] > 0.8:
            suggestions.append("✓ Meaning well preserved in optimization.")
            
        # Energy suggestions
        if opt_result.energy_reduction_estimate > 30:
            suggestions.append(f"🌱 Estimated {opt_result.energy_reduction_estimate}% energy savings!")
            
        # Quality suggestions
        if validation['intent_preserved']:
            suggestions.append("✓ Intent preserved in optimized prompt.")
        else:
            suggestions.append("⚠ Consider if the optimized prompt captures your intent.")
            
        # General tips
        if not suggestions:
            suggestions.append("Consider being more specific in your prompt.")
            
        return suggestions
    
    def _calculate_quality_score(
        self, 
        opt_result: OptimizationResult,
        validation: Dict
    ) -> float:
        """Calculate overall optimization quality score (0-100)."""
        # Weighted components
        token_score = min(opt_result.token_reduction, 80) / 80 * 30  # Max 30 points
        similarity_score = validation['semantic_similarity'] * 40  # Max 40 points
        intent_score = 15 if validation['intent_preserved'] else 5  # Max 15 points
        action_score = 15 if validation['action_preserved'] else 5  # Max 15 points
        
        total = token_score + similarity_score + intent_score + action_score
        return round(min(100, total), 1)
    
    def batch_optimize(self, prompts: List[str]) -> List[ComprehensiveOptimizationResult]:
        """
        Optimize multiple prompts.
        
        Args:
            prompts: List of prompts to optimize
            
        Returns:
            List of ComprehensiveOptimizationResult objects
        """
        return [self.optimize_prompt(p) for p in prompts]
    
    def get_energy_comparison(
        self, 
        original: str, 
        optimized: str,
        base_energy_kwh: float = 0.01
    ) -> Dict[str, float]:
        """
        Get detailed energy comparison between original and optimized prompts.
        
        Args:
            original: Original prompt
            optimized: Optimized prompt
            base_energy_kwh: Base energy consumption per token
            
        Returns:
            Dictionary with energy metrics
        """
        orig_tokens = self.complexity.get_token_count(original)
        opt_tokens = self.complexity.get_token_count(optimized)
        
        # Energy model: E ∝ tokens² (quadratic due to attention mechanism)
        orig_energy = base_energy_kwh * (orig_tokens ** 1.5)
        opt_energy = base_energy_kwh * (opt_tokens ** 1.5)
        
        energy_saved = orig_energy - opt_energy
        percent_saved = (energy_saved / max(orig_energy, 0.001)) * 100
        
        # Carbon footprint (0.475 kg CO2 per kWh - US average)
        carbon_factor = 0.475
        orig_carbon = orig_energy * carbon_factor
        opt_carbon = opt_energy * carbon_factor
        
        return {
            "original_energy_kwh": round(orig_energy, 6),
            "optimized_energy_kwh": round(opt_energy, 6),
            "energy_saved_kwh": round(energy_saved, 6),
            "percent_saved": round(percent_saved, 1),
            "original_carbon_kg": round(orig_carbon, 6),
            "optimized_carbon_kg": round(opt_carbon, 6),
            "carbon_saved_kg": round(orig_carbon - opt_carbon, 6)
        }
    
    def train_optimizer(self, epochs: int = 10, batch_size: int = 8) -> Dict:
        """
        Train/fine-tune the T5 optimizer model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Training metrics dictionary
        """
        return self.optimizer.train(epochs=epochs, batch_size=batch_size)
    
    def is_optimizer_trained(self) -> bool:
        """Check if the optimizer model is trained."""
        return self.optimizer.is_trained
    
    def get_similarity(self, text1: str, text2: str) -> float:
        """
        Get similarity score between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score (0-1)
        """
        result = self.similarity.compute_similarity(text1, text2)
        return result.score


# --- Module-level convenience functions ---

_nlp_service: Optional[NLPService] = None

def get_nlp_service() -> NLPService:
    """Get singleton instance of NLP Service."""
    global _nlp_service
    if _nlp_service is None:
        _nlp_service = NLPService()
    return _nlp_service


def optimize(prompt: str) -> ComprehensiveOptimizationResult:
    """Quick function to optimize a prompt."""
    service = get_nlp_service()
    return service.optimize_prompt(prompt)


if __name__ == "__main__":
    # Demo
    print("NLP Service Demo")
    print("=" * 60)
    
    service = NLPService()
    
    test_prompts = [
        "Could you please help me understand what machine learning is and how it works?",
        "Write a Python function that calculates factorial.",
        "I was wondering if you might possibly be able to help me with understanding how neural networks work in the context of deep learning applications.",
        "Explain recursion"
    ]
    
    for prompt in test_prompts:
        print(f"\n{'=' * 60}")
        result = service.optimize_prompt(prompt)
        
        print(f"Original ({result.original_tokens} tokens): {result.original_prompt}")
        print(f"Optimized ({result.optimized_tokens} tokens): {result.optimized_prompt}")
        print(f"\nMetrics:")
        print(f"  Token Reduction: {result.token_reduction_pct}%")
        print(f"  Energy Reduction: {result.energy_reduction_pct}%")
        print(f"  Semantic Similarity: {result.semantic_similarity}%")
        print(f"  Quality Score: {result.quality_score}/100")
        print(f"  Meaning Preserved: {result.meaning_preserved}")
        print(f"\nSuggestions:")
        for s in result.suggestions:
            print(f"  {s}")
