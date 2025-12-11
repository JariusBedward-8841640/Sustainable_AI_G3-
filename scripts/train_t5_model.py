#!/usr/bin/env python
"""
T5 Model Fine-Tuning Script for Prompt Optimization

This script fine-tunes the T5-small model on prompt optimization data and saves
the trained model to the model directory. The saved model can then be distributed
with the project so users don't need to train it themselves.

Usage:
    python scripts/train_t5_model.py [--epochs N] [--batch-size N]
    
After training, the model is saved to: model/prompt_optimizer/t5_finetuned/

The trained model will be automatically loaded when the application starts.
"""

import os
import sys
import argparse

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

from src.nlp.prompt_optimizer import T5PromptOptimizer


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune T5 model for prompt optimization'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=10,
        help='Number of training epochs (default: 10)'
    )
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=8,
        help='Training batch size (default: 8)'
    )
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=3e-4,
        help='Learning rate (default: 3e-4)'
    )
    parser.add_argument(
        '--warmup-steps', 
        type=int, 
        default=100,
        help='Number of warmup steps (default: 100)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for trained model (default: model/prompt_optimizer/t5_finetuned)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("T5 Prompt Optimizer Fine-Tuning")
    print("=" * 60)
    print(f"\nConfiguration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.learning_rate}")
    print(f"  Warmup Steps: {args.warmup_steps}")
    
    # Initialize optimizer (loads base model)
    print("\nInitializing T5 optimizer...")
    optimizer = T5PromptOptimizer()
    
    if not optimizer.t5_available:
        print("\n❌ T5 model not available. Please install dependencies:")
        print("   pip install torch transformers sentencepiece")
        sys.exit(1)
    
    # Verify training data exists
    training_data_path = os.path.join(project_root, 'model', 'prompt_optimizer', 'training_data.json')
    if not os.path.exists(training_data_path):
        print(f"\n❌ Training data not found at: {training_data_path}")
        sys.exit(1)
    
    print(f"\n✅ Training data found at: {training_data_path}")
    
    # Train the model
    print("\n" + "=" * 60)
    print("Starting Fine-Tuning...")
    print("=" * 60)
    
    try:
        metrics = optimizer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            warmup_steps=args.warmup_steps,
            save_path=args.output_dir
        )
        
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print("=" * 60)
        print(f"\nMetrics:")
        print(f"  Training Loss: {metrics['train_loss']:.4f}")
        print(f"  Eval Loss: {metrics['eval_loss']:.4f}")
        print(f"  Samples Used: {metrics['samples']}")
        
        model_dir = args.output_dir or os.path.join(project_root, 'model', 'prompt_optimizer', 't5_finetuned')
        print(f"\nModel saved to: {model_dir}")
        print("\nThe model will be automatically loaded when the app starts.")
        print("To share this model, include the 't5_finetuned' directory in your distribution.")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def test_trained_model():
    """Test the trained model after training."""
    print("\n" + "=" * 60)
    print("Testing Trained Model")
    print("=" * 60)
    
    # Re-initialize to load the trained model
    optimizer = T5PromptOptimizer()
    
    test_prompts = [
        "Could you please help me understand machine learning?",
        "I would really appreciate it if you could summarize this text.",
        "Would you kindly provide me with some Python code examples?",
    ]
    
    print("\nTest Results:")
    for prompt in test_prompts:
        optimized, changes = optimizer.optimize(prompt)
        print(f"\n  Original: {prompt}")
        print(f"  Optimized: {optimized}")
        print(f"  Changes: {changes}")


if __name__ == '__main__':
    main()
    
    # Optionally test the model
    print("\n" + "-" * 60)
    response = input("\nWould you like to test the trained model? (y/n): ").strip().lower()
    if response == 'y':
        test_trained_model()
