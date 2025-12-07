"""
Training Script for T5 Prompt Optimizer

This script trains the T5 model on prompt optimization data and evaluates its performance.
Run this script to fine-tune the model before using the optimizer.

Usage:
    python train_optimizer.py [--epochs N] [--batch-size N] [--eval-only]

Author: Sustainable AI Team
"""

import os
import sys
import argparse
import json
from datetime import datetime

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.nlp.prompt_optimizer import T5PromptOptimizer
from src.nlp.semantic_similarity import SemanticSimilarity, EnhancedPromptValidator


def evaluate_model(optimizer: T5PromptOptimizer, test_prompts: list) -> dict:
    """
    Evaluate the optimizer on test prompts.
    
    Args:
        optimizer: The T5PromptOptimizer instance
        test_prompts: List of test prompt dictionaries
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("\n" + "=" * 60)
    print("Evaluating Model Performance")
    print("=" * 60)
    
    similarity_checker = SemanticSimilarity()
    validator = EnhancedPromptValidator()
    
    results = {
        "total_prompts": len(test_prompts),
        "avg_token_reduction": 0,
        "avg_semantic_similarity": 0,
        "avg_quality_score": 0,
        "meaning_preserved_count": 0,
        "by_category": {}
    }
    
    total_token_reduction = 0
    total_similarity = 0
    total_quality = 0
    
    for i, test in enumerate(test_prompts):
        original = test.get('original', '')
        expected = test.get('optimized', '')
        category = test.get('category', 'general')
        
        # Get optimization
        opt_result = optimizer.get_full_optimization(original)
        
        # Validate
        validation = validator.validate_prompt_optimization(original, opt_result.optimized_prompt)
        
        # Track metrics
        total_token_reduction += opt_result.token_reduction
        total_similarity += validation['semantic_similarity']
        total_quality += validation['quality_score']
        
        if validation['is_valid']:
            results['meaning_preserved_count'] += 1
        
        # Track by category
        if category not in results['by_category']:
            results['by_category'][category] = {
                'count': 0,
                'avg_reduction': 0,
                'avg_similarity': 0
            }
        results['by_category'][category]['count'] += 1
        
        # Print sample results (first 5)
        if i < 5:
            print(f"\nTest {i+1}:")
            print(f"  Original: {original[:80]}...")
            print(f"  Optimized: {opt_result.optimized_prompt}")
            print(f"  Expected: {expected}")
            print(f"  Token Reduction: {opt_result.token_reduction}%")
            print(f"  Similarity: {validation['semantic_similarity']:.2%}")
    
    # Calculate averages
    n = len(test_prompts)
    results['avg_token_reduction'] = round(total_token_reduction / n, 1)
    results['avg_semantic_similarity'] = round(total_similarity / n, 3)
    results['avg_quality_score'] = round(total_quality / n, 3)
    results['meaning_preservation_rate'] = round(results['meaning_preserved_count'] / n * 100, 1)
    
    return results


def load_test_data() -> list:
    """Load test data for evaluation."""
    data_dir = os.path.join(project_root, 'data', 'prompt_optimization')
    test_data = []
    
    # Load from training data (use last 20% as test)
    main_path = os.path.join(data_dir, 'training_data.json')
    if os.path.exists(main_path):
        with open(main_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if 'data' in data:
                all_data = data['data']
                split_idx = int(len(all_data) * 0.8)
                test_data.extend(all_data[split_idx:])
    
    return test_data


def print_evaluation_report(results: dict):
    """Print formatted evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    
    print(f"\nOverall Metrics:")
    print(f"  Total Test Prompts: {results['total_prompts']}")
    print(f"  Avg Token Reduction: {results['avg_token_reduction']}%")
    print(f"  Avg Semantic Similarity: {results['avg_semantic_similarity']:.1%}")
    print(f"  Avg Quality Score: {results['avg_quality_score']:.1%}")
    print(f"  Meaning Preservation Rate: {results['meaning_preservation_rate']}%")
    
    if results['by_category']:
        print(f"\nBy Category:")
        for cat, metrics in results['by_category'].items():
            print(f"  {cat}: {metrics['count']} prompts")
    
    # Quality assessment
    print("\n" + "-" * 40)
    if results['avg_semantic_similarity'] >= 0.7 and results['meaning_preservation_rate'] >= 80:
        print("✅ Model Quality: EXCELLENT")
        print("   Ready for production use.")
    elif results['avg_semantic_similarity'] >= 0.5 and results['meaning_preservation_rate'] >= 60:
        print("✓ Model Quality: GOOD")
        print("   Consider additional training for better results.")
    else:
        print("⚠ Model Quality: NEEDS IMPROVEMENT")
        print("   Additional training recommended.")


def main():
    parser = argparse.ArgumentParser(description='Train T5 Prompt Optimizer')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--eval-only', action='store_true', help='Only evaluate, skip training')
    parser.add_argument('--model-path', type=str, default=None, help='Path to load/save model')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("T5 Prompt Optimizer Training Script")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize optimizer
    optimizer = T5PromptOptimizer(model_path=args.model_path)
    
    if not args.eval_only:
        print(f"\nTraining Configuration:")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch Size: {args.batch_size}")
        print(f"  Learning Rate: {args.learning_rate}")
        
        # Train
        metrics = optimizer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate
        )
        
        print(f"\nTraining Metrics:")
        print(f"  Final Loss: {metrics.get('train_loss', 'N/A')}")
        print(f"  Eval Loss: {metrics.get('eval_loss', 'N/A')}")
    
    # Evaluate
    test_data = load_test_data()
    if test_data:
        results = evaluate_model(optimizer, test_data)
        print_evaluation_report(results)
        
        # Save evaluation results
        results_path = os.path.join(project_root, 'model', 'prompt_optimizer', 'eval_results.json')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nEvaluation results saved to: {results_path}")
    else:
        print("\nNo test data found for evaluation.")
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
