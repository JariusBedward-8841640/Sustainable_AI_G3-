"""
T5-Based Prompt Optimizer Module

This module provides a fine-tuned T5 model for optimizing prompts to be more
energy-efficient while maintaining semantic equivalence. It reduces token count
and complexity without losing meaning.

Features:
- Fine-tuned T5-small model for prompt optimization
- Semantic similarity validation using sentence transformers
- Energy estimation based on token reduction
- Professional-grade prompt optimization for 5-200 tokens

Author: Sustainable AI Team
"""

import os
import sys
import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Transformers imports
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from torch.utils.data import Dataset


@dataclass
class OptimizationResult:
    """Data class for prompt optimization results."""
    original_prompt: str
    optimized_prompt: str
    original_tokens: int
    optimized_tokens: int
    token_reduction: float
    semantic_similarity: float
    energy_reduction_estimate: float
    optimization_quality: str
    

class PromptDataset(Dataset):
    """PyTorch Dataset for prompt optimization training data."""
    
    def __init__(
        self, 
        data: List[Dict], 
        tokenizer: T5Tokenizer, 
        max_input_length: int = 256,
        max_target_length: int = 128
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Prepare input with task prefix
        input_text = f"optimize prompt: {item['original']}"
        target_text = item['optimized']
        
        # Tokenize
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        labels = target_encoding['input_ids'].squeeze()
        # Replace padding token id with -100 for loss computation
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': labels
        }


class T5PromptOptimizer:
    """
    T5-based Prompt Optimizer for energy-efficient prompt rewriting.
    
    This class provides methods to:
    - Load/train a T5 model for prompt optimization
    - Optimize prompts while preserving semantic meaning
    - Estimate energy savings from optimized prompts
    """
    
    MODEL_NAME = "t5-small"
    TASK_PREFIX = "optimize prompt: "
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the T5 Prompt Optimizer.
        
        Args:
            model_path: Path to a fine-tuned model. If None, loads base model.
            device: Device to use ('cuda' or 'cpu'). Auto-detected if None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default model directory
        self.default_model_dir = os.path.join(project_root, 'model', 'prompt_optimizer', 't5_finetuned')
        
        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.MODEL_NAME, legacy=False)
        
        # Load model (either fine-tuned or base)
        if model_path and os.path.exists(model_path):
            print(f"Loading fine-tuned model from {model_path}")
            self.model = T5ForConditionalGeneration.from_pretrained(model_path)
            self.is_trained = True
        elif os.path.exists(self.default_model_dir):
            print(f"Loading fine-tuned model from default path: {self.default_model_dir}")
            self.model = T5ForConditionalGeneration.from_pretrained(self.default_model_dir)
            self.is_trained = True
        else:
            print(f"Loading base {self.MODEL_NAME} model (not fine-tuned)")
            self.model = T5ForConditionalGeneration.from_pretrained(self.MODEL_NAME)
            self.is_trained = False
            
        self.model.to(self.device)
        self.model.eval()
        
        # Training metrics storage
        self.training_metrics = {}
        
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using T5 tokenizer."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))
    
    def _load_training_data(self) -> List[Dict]:
        """Load all training data from JSON files."""
        data = []
        data_dir = os.path.join(project_root, 'data', 'prompt_optimization')
        
        # Load main training data
        main_data_path = os.path.join(data_dir, 'training_data.json')
        if os.path.exists(main_data_path):
            with open(main_data_path, 'r', encoding='utf-8') as f:
                main_data = json.load(f)
                if 'data' in main_data:
                    data.extend(main_data['data'])
                    
        # Load extended training data
        ext_data_path = os.path.join(data_dir, 'extended_training_data.json')
        if os.path.exists(ext_data_path):
            with open(ext_data_path, 'r', encoding='utf-8') as f:
                ext_data = json.load(f)
                if 'data' in ext_data:
                    data.extend(ext_data['data'])
                    
        print(f"Loaded {len(data)} training examples")
        return data
    
    def train(
        self, 
        epochs: int = 10,
        batch_size: int = 8,
        learning_rate: float = 3e-4,
        warmup_steps: int = 100,
        save_path: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Fine-tune the T5 model on prompt optimization data.
        
        Args:
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            warmup_steps: Number of warmup steps
            save_path: Path to save the trained model
            
        Returns:
            Dictionary containing training metrics
        """
        print("=" * 50)
        print("Starting T5 Prompt Optimizer Training")
        print("=" * 50)
        
        # Load training data
        train_data = self._load_training_data()
        
        if len(train_data) == 0:
            raise ValueError("No training data found!")
        
        # Create train/validation split
        split_idx = int(len(train_data) * 0.9)
        train_split = train_data[:split_idx]
        val_split = train_data[split_idx:]
        
        print(f"Training samples: {len(train_split)}")
        print(f"Validation samples: {len(val_split)}")
        
        # Create datasets
        train_dataset = PromptDataset(train_split, self.tokenizer)
        val_dataset = PromptDataset(val_split, self.tokenizer)
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Training arguments
        output_dir = save_path or self.default_model_dir
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            save_total_limit=2,
            report_to="none",
            fp16=torch.cuda.is_available(),
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )
        
        # Train
        print("\nTraining started...")
        train_result = trainer.train()
        
        # Evaluate
        eval_result = trainer.evaluate()
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Store metrics
        self.training_metrics = {
            'train_loss': train_result.training_loss,
            'eval_loss': eval_result.get('eval_loss', 0),
            'epochs': epochs,
            'samples': len(train_data)
        }
        
        self.is_trained = True
        print("\nTraining complete!")
        print(f"Model saved to: {output_dir}")
        print(f"Final training loss: {self.training_metrics['train_loss']:.4f}")
        print(f"Final eval loss: {self.training_metrics['eval_loss']:.4f}")
        
        return self.training_metrics
    
    def optimize(
        self, 
        prompt: str,
        max_length: int = 128,
        num_beams: int = 4,
        temperature: float = 0.7,
        do_sample: bool = False
    ) -> str:
        """
        Optimize a prompt to be more energy-efficient.
        
        Args:
            prompt: The original prompt to optimize
            max_length: Maximum length of generated output
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Optimized prompt string
        """
        if not prompt or not prompt.strip():
            return prompt
            
        # Prepare input
        input_text = f"{self.TASK_PREFIX}{prompt}"
        
        # Tokenize
        input_ids = self.tokenizer.encode(
            input_text, 
            return_tensors='pt',
            max_length=256,
            truncation=True
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                early_stopping=True,
                no_repeat_ngram_size=2
            )
        
        # Decode
        optimized = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Validate - if optimization is longer or nonsensical, return original
        if len(optimized.strip()) == 0:
            return prompt
            
        orig_tokens = self._count_tokens(prompt)
        opt_tokens = self._count_tokens(optimized)
        
        # If no significant reduction for longer prompts, apply fallback
        if orig_tokens > 10 and opt_tokens >= orig_tokens * 0.95:
            optimized = self._fallback_optimize(prompt)
            
        return optimized.strip()
    
    def _fallback_optimize(self, prompt: str) -> str:
        """
        Fallback optimization using rule-based approach.
        Used when T5 doesn't provide significant optimization.
        """
        # Common filler phrases to remove
        fillers = [
            "could you please", "would you please", "can you please",
            "i was wondering if you could", "would it be possible to",
            "i would appreciate if you could", "please help me",
            "i need your help with", "i am looking for help with",
            "kindly", "please", "thank you", "thanks",
            "i would like you to", "could you possibly",
            "would you mind", "i was hoping you could",
            "it would be great if you could", "i really need",
            "basically", "actually", "literally", "honestly",
            "in order to", "the fact that", "due to the fact that"
        ]
        
        result = prompt.lower()
        for filler in fillers:
            result = result.replace(filler, "")
        
        # Clean up extra spaces
        result = " ".join(result.split())
        
        # Capitalize first letter
        if result:
            result = result[0].upper() + result[1:] if len(result) > 1 else result.upper()
            
        return result
    
    def get_full_optimization(self, prompt: str) -> OptimizationResult:
        """
        Get complete optimization result with all metrics.
        
        Args:
            prompt: Original prompt to optimize
            
        Returns:
            OptimizationResult with all optimization metrics
        """
        original_tokens = self._count_tokens(prompt)
        
        # Optimize
        optimized = self.optimize(prompt)
        optimized_tokens = self._count_tokens(optimized)
        
        # Calculate metrics
        token_reduction = 1 - (optimized_tokens / max(original_tokens, 1))
        token_reduction = max(0, token_reduction)  # Ensure non-negative
        
        # Estimate energy reduction (proportional to token reduction)
        # Based on research: energy ~ O(n^2) for transformer attention
        energy_reduction = 1 - ((optimized_tokens / max(original_tokens, 1)) ** 2)
        energy_reduction = max(0, min(1, energy_reduction))
        
        # Semantic similarity (placeholder - will be enhanced by similarity module)
        semantic_similarity = self._estimate_semantic_similarity(prompt, optimized)
        
        # Quality rating
        if token_reduction > 0.5 and semantic_similarity > 0.7:
            quality = "Excellent"
        elif token_reduction > 0.3 and semantic_similarity > 0.6:
            quality = "Good"
        elif token_reduction > 0.1 and semantic_similarity > 0.5:
            quality = "Moderate"
        else:
            quality = "Minimal"
            
        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            token_reduction=round(token_reduction * 100, 1),
            semantic_similarity=round(semantic_similarity * 100, 1),
            energy_reduction_estimate=round(energy_reduction * 100, 1),
            optimization_quality=quality
        )
    
    def _estimate_semantic_similarity(self, original: str, optimized: str) -> float:
        """
        Estimate semantic similarity between original and optimized prompts.
        This is a basic implementation - will be enhanced by the similarity module.
        """
        if not original or not optimized:
            return 0.0
            
        # Simple overlap-based similarity
        orig_words = set(original.lower().split())
        opt_words = set(optimized.lower().split())
        
        if not orig_words or not opt_words:
            return 0.0
            
        # Jaccard similarity
        intersection = len(orig_words.intersection(opt_words))
        union = len(orig_words.union(opt_words))
        
        jaccard = intersection / max(union, 1)
        
        # Boost score for keyword preservation
        important_keywords = self._extract_keywords(original)
        preserved_keywords = sum(1 for kw in important_keywords if kw in optimized.lower())
        keyword_score = preserved_keywords / max(len(important_keywords), 1)
        
        # Combined score
        return min(1.0, (jaccard * 0.4) + (keyword_score * 0.6))
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text."""
        # Common stop words to ignore
        stop_words = {
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'need', 'dare', 'ought', 'used', 'to', 'of', 'in',
            'for', 'on', 'with', 'at', 'by', 'from', 'up', 'about',
            'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'between', 'under', 'again', 'further', 'then',
            'once', 'here', 'there', 'when', 'where', 'why', 'how',
            'all', 'each', 'few', 'more', 'most', 'other', 'some',
            'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
            'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
            'because', 'as', 'until', 'while', 'although', 'though',
            'please', 'help', 'me', 'i', 'you', 'your', 'my', 'we',
            'kindly', 'could', 'would', 'please'
        }
        
        words = text.lower().split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:10]  # Return top 10 keywords
    
    def batch_optimize(self, prompts: List[str]) -> List[OptimizationResult]:
        """
        Optimize multiple prompts in batch.
        
        Args:
            prompts: List of prompts to optimize
            
        Returns:
            List of OptimizationResult objects
        """
        return [self.get_full_optimization(p) for p in prompts]
    
    def save(self, path: Optional[str] = None):
        """Save the model and tokenizer."""
        save_path = path or self.default_model_dir
        os.makedirs(save_path, exist_ok=True)
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        print(f"Model saved to {save_path}")
        
    def load(self, path: str):
        """Load a saved model."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}")
        self.model = T5ForConditionalGeneration.from_pretrained(path)
        self.tokenizer = T5Tokenizer.from_pretrained(path, legacy=False)
        self.model.to(self.device)
        self.model.eval()
        self.is_trained = True
        print(f"Model loaded from {path}")


# --- Convenience function for quick optimization ---
def optimize_prompt(prompt: str) -> str:
    """
    Quick function to optimize a single prompt.
    
    Args:
        prompt: The prompt to optimize
        
    Returns:
        Optimized prompt string
    """
    optimizer = T5PromptOptimizer()
    return optimizer.optimize(prompt)


if __name__ == "__main__":
    # Demo usage
    print("T5 Prompt Optimizer Demo")
    print("=" * 50)
    
    optimizer = T5PromptOptimizer()
    
    test_prompts = [
        "Could you please help me understand what machine learning is and how it works?",
        "I was wondering if you might be able to assist me with writing a Python function.",
        "Write code",
        "Please explain the concept of recursion in simple terms that I can understand."
    ]
    
    for prompt in test_prompts:
        result = optimizer.get_full_optimization(prompt)
        print(f"\nOriginal ({result.original_tokens} tokens): {result.original_prompt}")
        print(f"Optimized ({result.optimized_tokens} tokens): {result.optimized_prompt}")
        print(f"Token Reduction: {result.token_reduction}%")
        print(f"Energy Reduction: {result.energy_reduction_estimate}%")
        print(f"Quality: {result.optimization_quality}")
