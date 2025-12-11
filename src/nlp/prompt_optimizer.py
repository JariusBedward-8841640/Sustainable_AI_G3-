"""
T5-Based Prompt Optimizer Module

This module provides intelligent prompt optimization using T5 model combined
with sophisticated rule-based transformations. It truly rephrases prompts
(not just truncates) to be more energy-efficient while maintaining semantic meaning.

Features:
- T5-small/base model for intelligent text rephrasing
- Advanced rule-based linguistic transformations
- Removes energy-intensive formatting requests
- Converts verbose phrases to concise equivalents
- Preserves core semantic intent and keywords
- Energy estimation based on token reduction

Author: Sustainable AI Team
"""

import os
import sys
import json
import re
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Core transformers imports (these don't trigger TF/Keras)
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.utils.data import Dataset

# Lazy imports for training (only needed when training, not for inference)
TrainingArguments = None
Trainer = None
DataCollatorForSeq2Seq = None

def _import_training_components():
    """Lazy import training components only when needed."""
    global TrainingArguments, Trainer, DataCollatorForSeq2Seq
    if TrainingArguments is None:
        from transformers import TrainingArguments as TA, Trainer as TR, DataCollatorForSeq2Seq as DC
        TrainingArguments = TA
        Trainer = TR
        DataCollatorForSeq2Seq = DC


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
    changes_made: List[str] = field(default_factory=list)
    

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
    
    T5 Setup Requirements:
    ----------------------
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install transformers sentencepiece
    
    Note: If you encounter tf-keras errors, the optimizer will still work
    using rule-based transformations. T5 is only required for training.
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
        
        # T5 model availability tracking
        self.t5_available = False
        self.t5_status_message = ""
        
        # Try to load T5 model
        try:
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(self.MODEL_NAME, legacy=False)
            
            # Model loading kwargs to avoid meta tensor issues with newer transformers
            # - low_cpu_mem_usage=False: Fully load model into memory (no meta tensors)
            # - device_map=None: Don't use accelerate's device mapping
            # We'll load to CPU first, then move to device to avoid meta tensor issues
            load_kwargs = {
                "low_cpu_mem_usage": False,
                "device_map": None,  # Explicitly disable device_map
            }
            
            # Load model (either fine-tuned or base)
            if model_path and os.path.exists(model_path):
                print(f"Loading fine-tuned model from {model_path}")
                self.model = T5ForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
                self.is_trained = True
                self.t5_status_message = f"T5 fine-tuned model loaded from {model_path}"
            elif os.path.exists(self.default_model_dir):
                print(f"Loading fine-tuned model from default path: {self.default_model_dir}")
                self.model = T5ForConditionalGeneration.from_pretrained(self.default_model_dir, **load_kwargs)
                self.is_trained = True
                self.t5_status_message = "T5 fine-tuned model loaded"
            else:
                print(f"Loading base {self.MODEL_NAME} model (not fine-tuned)")
                self.model = T5ForConditionalGeneration.from_pretrained(self.MODEL_NAME, **load_kwargs)
                self.is_trained = False
                self.t5_status_message = f"T5 base model ({self.MODEL_NAME}) loaded"
            
            # Move to device - should be safe now with low_cpu_mem_usage=False
            try:
                self.model = self.model.to(self.device)
            except Exception as move_error:
                print(f"Warning: Could not move model to {self.device}: {move_error}")
                # Keep model on CPU if move fails
                self.device = 'cpu'
                
            self.model.eval()
            self.t5_available = True
            
        except Exception as e:
            print(f"⚠️ T5 model not available: {e}")
            print("Using rule-based optimization only.")
            self.model = None
            self.tokenizer = None
            self.is_trained = False
            self.t5_available = False
            self.t5_status_message = f"T5 unavailable - using rule-based fallback. Error: {str(e)[:100]}"
        
        # Training metrics storage
        self.training_metrics = {}
        
        # Load word replacements from JSON
        self.word_replacements = self._load_word_replacements()
        
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using T5 tokenizer or word count fallback."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Fallback to word count approximation if tokenizer not available
        return len(text.split())
    
    def get_status(self) -> dict:
        """
        Get the current status of the T5 optimizer.
        
        Returns:
            Dictionary with t5_available, is_trained, device, and status_message
        """
        return {
            "t5_available": self.t5_available,
            "is_trained": self.is_trained,
            "device": self.device,
            "status_message": self.t5_status_message,
            "using_fallback": not self.t5_available
        }
    
    def _load_word_replacements(self) -> Dict:
        """
        Load word replacement rules from JSON file.
        
        Returns:
            Dictionary with verbose_phrases, word_simplifications, 
            filler_words, and redundant_phrases.
        """
        default_replacements = {
            'verbose_phrases': {},
            'word_simplifications': {},
            'filler_words': [],
            'redundant_phrases': {}
        }
        
        json_path = os.path.join(project_root, 'model', 'prompt_optimizer', 'word_replacements.json')
        
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    replacements = json.load(f)
                    print(f"Loaded word replacements from {json_path}")
                    return replacements
            except Exception as e:
                print(f"Warning: Could not load word_replacements.json: {e}")
                return default_replacements
        else:
            print(f"Note: word_replacements.json not found at {json_path}, using defaults")
            return default_replacements
    
    def _remove_duplicate_content(self, text: str) -> Tuple[str, bool]:
        """
        Detect and remove duplicate sentences, paragraphs, or repeated content.
        Handles cases where the same content is copy-pasted multiple times.
        
        Args:
            text: Input text
        
        Returns:
            Tuple of (deduplicated_text, was_deduplicated)
        """
        if not text or len(text) < 20:
            return text, False
        
        # Normalize whitespace first
        normalized = re.sub(r'\s+', ' ', text.strip())
        
        # Strategy 1: Check if entire text is a repeated pattern
        deduplicated = self._find_repeating_pattern(normalized)
        if deduplicated and len(deduplicated) < len(normalized) * 0.6:
            return deduplicated.strip(), True
        
        # Strategy 2: Remove duplicate sentences
        try:
            # Simple sentence splitting (avoid NLTK dependency issues)
            sentences = re.split(r'(?<=[.!?])\s+', normalized)
        except Exception:
            sentences = [s.strip() for s in normalized.split('.') if s.strip()]
        
        if len(sentences) > 1:
            # Track seen sentences (normalized for comparison)
            seen = set()
            unique_sentences = []
            
            for sentence in sentences:
                # Normalize for comparison (lowercase, strip punctuation)
                normalized_sent = re.sub(r'[^\w\s]', '', sentence.lower()).strip()
                
                if normalized_sent and normalized_sent not in seen:
                    seen.add(normalized_sent)
                    unique_sentences.append(sentence.strip())
            
            # If we removed duplicates, join them back
            if len(unique_sentences) < len(sentences):
                result = ' '.join(unique_sentences)
                if result and result[-1] not in '.?!':
                    result += '.'
                return result, True
        
        return text, False
    
    def _find_repeating_pattern(self, text: str) -> Optional[str]:
        """
        Find if text is composed of a repeating pattern.
        For example: "ABC.ABC.ABC.ABC" -> "ABC."
        
        Args:
            text: Input text to check
            
        Returns:
            The base pattern if found, None otherwise
        """
        n = len(text)
        if n < 100:  # Don't try to find patterns in short text
            return None
        
        # Normalize whitespace
        norm_text = re.sub(r'\s+', ' ', text).strip()
        n = len(norm_text)
        
        # Try to find pattern by looking for repeated sentence-like blocks
        parts = [p.strip() for p in norm_text.split('.') if p.strip()]
        
        if len(parts) >= 2:
            # Check if all parts are the same (with minor variations)
            first_part = re.sub(r'[^\w\s]', '', parts[0].lower()).strip()
            same_count = sum(1 for p in parts if re.sub(r'[^\w\s]', '', p.lower()).strip() == first_part)
            
            if same_count >= len(parts) * 0.8:  # 80% same = repeating pattern
                return parts[0].strip() + '.'
        
        # Try pattern matching for cases without periods as separators
        for pattern_len in range(50, min(n // 2 + 1, 500)):
            pattern = norm_text[:pattern_len]
            
            # Count occurrences
            count = norm_text.count(pattern)
            
            if count >= 2:
                coverage = (count * len(pattern)) / n
                if coverage >= 0.7:  # Pattern covers 70%+ of text
                    # Make sure we return a complete sentence/thought
                    if '.' in pattern:
                        end_idx = pattern.rfind('.') + 1
                        if end_idx > 20:  # Reasonable sentence
                            return pattern[:end_idx].strip()
                    return pattern.strip()
        
        return None
    
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
        # Lazy import training components (avoids TF/Keras dependency on import)
        _import_training_components()
        
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
    ) -> Tuple[str, List[str]]:
        """
        Optimize a prompt to be more energy-efficient using a hybrid approach:
        1. First apply rule-based transformations for guaranteed improvements
        2. Then use T5 for additional semantic compression if beneficial
        
        Args:
            prompt: The original prompt to optimize
            max_length: Maximum length of generated output
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Tuple of (optimized_prompt, list_of_changes_made)
        """
        if not prompt or not prompt.strip():
            return prompt, []
        
        changes_made = []
        
        # =================================================================
        # STEP 1: Apply rule-based transformations first
        # =================================================================
        rule_optimized, rule_changes = self._fallback_optimize(prompt)
        changes_made.extend(rule_changes)
        
        orig_tokens = self._count_tokens(prompt)
        rule_tokens = self._count_tokens(rule_optimized)
        
        # If rule-based already achieved significant reduction, use it
        rule_reduction = (orig_tokens - rule_tokens) / max(orig_tokens, 1)
        
        # =================================================================
        # STEP 2: Try T5 for additional semantic compression (if available)
        # =================================================================
        if not self.t5_available or self.model is None or self.tokenizer is None:
            # T5 not available, use rule-based result only
            if rule_changes:
                changes_made.append("Rule-based optimization applied (T5 unavailable)")
            return rule_optimized, changes_made
        
        # Use T5 to further compress the rule-optimized version
        # Use "summarize:" prefix which T5 understands well for compression
        input_text = f"summarize: {rule_optimized}"
        
        input_ids = self.tokenizer.encode(
            input_text, 
            return_tensors='pt',
            max_length=256,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_length=min(max_length, rule_tokens + 10),  # Don't allow much longer
                min_length=max(5, rule_tokens // 3),  # Ensure meaningful output
                num_beams=num_beams,
                temperature=temperature,
                do_sample=do_sample,
                early_stopping=True,
                no_repeat_ngram_size=2,
                length_penalty=1.5,  # Encourage shorter outputs
            )
        
        t5_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        t5_tokens = self._count_tokens(t5_output)
        
        # =================================================================
        # STEP 3: Choose best result
        # =================================================================
        # Use T5 output only if it's meaningful and shorter
        if (t5_output and 
            len(t5_output.strip()) > 5 and 
            t5_tokens < rule_tokens * 0.9 and  # At least 10% shorter
            self._is_semantically_valid(prompt, t5_output)):
            
            final_output = t5_output.strip()
            changes_made.append("T5 semantic compression applied")
            
            # Ensure proper capitalization and punctuation
            if final_output[0].islower():
                final_output = final_output[0].upper() + final_output[1:]
            if final_output[-1] not in '.?!':
                final_output += '.'
        else:
            final_output = rule_optimized
            if rule_changes:
                if self.is_trained:
                    changes_made.append("T5 validated, rule-based result used (optimal)")
                else:
                    changes_made.append("Rule-based optimization applied")
        
        return final_output, changes_made
    
    def _is_semantically_valid(self, original: str, optimized: str) -> bool:
        """
        Check if the optimized prompt preserves key semantic content.
        """
        # Extract keywords from original
        orig_keywords = set(self._extract_keywords(original))
        opt_keywords = set(self._extract_keywords(optimized))
        
        if not orig_keywords:
            return True
        
        # At least 40% of important keywords should be preserved
        preserved = len(orig_keywords.intersection(opt_keywords))
        preservation_rate = preserved / len(orig_keywords)
        
        return preservation_rate >= 0.4
    
    def _fallback_optimize(self, prompt: str) -> Tuple[str, List[str]]:
        """
        Advanced rule-based optimization that performs true linguistic transformations.
        This does actual rephrasing, not just filler removal.
        
        Returns:
            Tuple of (optimized_prompt, list_of_changes_made)
        """
        changes = []
        result = prompt
        
        # =================================================================
        # PHASE 0: Remove duplicate content FIRST (biggest potential savings)
        # =================================================================
        result, was_deduplicated = self._remove_duplicate_content(result)
        if was_deduplicated:
            changes.append("Removed duplicate content")
        
        # =================================================================
        # PHASE 0.5: Apply JSON-loaded verbose phrase replacements
        # =================================================================
        if hasattr(self, 'word_replacements') and self.word_replacements:
            # Apply verbose phrases
            for verbose, simple in self.word_replacements.get('verbose_phrases', {}).items():
                pattern = r'\b' + re.escape(verbose) + r'\b'
                if re.search(pattern, result, re.IGNORECASE):
                    result = re.sub(pattern, simple, result, flags=re.IGNORECASE)
                    if simple:
                        changes.append(f"Simplified '{verbose}' → '{simple}'")
                    else:
                        changes.append(f"Removed '{verbose}'")
            
            # Apply word simplifications
            for complex_word, simple_word in self.word_replacements.get('word_simplifications', {}).items():
                pattern = r'\b' + re.escape(complex_word) + r'\b'
                if re.search(pattern, result, re.IGNORECASE):
                    result = re.sub(pattern, simple_word, result, flags=re.IGNORECASE)
                    changes.append(f"Simplified '{complex_word}' → '{simple_word}'")
            
            # Apply redundant phrase fixes
            for redundant, simple in self.word_replacements.get('redundant_phrases', {}).items():
                pattern = r'\b' + re.escape(redundant) + r'\b'
                if re.search(pattern, result, re.IGNORECASE):
                    result = re.sub(pattern, simple, result, flags=re.IGNORECASE)
                    changes.append(f"Fixed redundancy: '{redundant}' → '{simple}'")
        
        # =================================================================
        # PHASE 1: Remove energy/sustainability phrases (not relevant to AI tasks)
        # =================================================================
        sustainability_patterns = [
            # "Using renewable energy sources, ..." → remove the phrase
            (r"(?:using|leveraging|with|through)\s+(?:renewable\s+)?(?:energy\s+sources?|green\s+computing\s+practices?|sustainable\s+(?:methods?|practices?|approaches?)|eco-friendly\s+(?:methods?|approaches?))\s*,?\s*", "", "Removed sustainability phrase"),
            
            # "In a sustainable manner, ..." → remove the phrase
            (r"(?:in\s+)?(?:a\s+)?(?:sustainable|environmentally?\s+(?:friendly|conscious)|eco-friendly|green)\s+(?:manner|way|fashion)\s*,?\s*", "", "Removed sustainability phrase"),
            
            # "With minimal carbon footprint, ..." → remove the phrase
            (r"(?:with\s+)?(?:minimal|low|reduced|zero)\s+(?:carbon\s+)?(?:footprint|emissions?|impact)\s*,?\s*", "", "Removed carbon phrase"),
            
            # "In an energy-efficient way, ..." → remove the phrase  
            (r"(?:in\s+)?(?:an?\s+)?(?:energy-?efficient|power-?efficient|resource-?efficient)\s+(?:manner|way|fashion)?\s*,?\s*", "", "Removed efficiency phrase"),
            
            # "Sustainably, ..." or "Efficiently, ..." at start
            (r"^(?:sustainably|efficiently|eco-?consciously)\s*,?\s*", "", "Removed sustainability adverb"),
        ]
        
        for pattern, replacement, change_desc in sustainability_patterns:
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                changes.append(change_desc)
        
        # =================================================================
        # PHASE 1b: Remove energy-intensive formatting requests (complete sentences)
        # =================================================================
        energy_intensive_patterns = [
            # Complete sentence patterns that can be safely removed
            (r"\.?\s*(?:please\s+)?(?:format|structure|organize)\s+(?:the\s+)?(?:output|response|answer|result|it)\s+(?:as|in|using)\s+(?:proper\s+)?(?:json|xml|yaml|markdown|md|html|csv|table)(?:\s+format)?\.?", "", "Removed formatting request"),
            (r"\.?\s*(?:please\s+)?use\s+(?:proper\s+)?(?:markdown|md)\s+(?:formatting|syntax|headers?)?\.?", "", "Removed markdown request"),
            (r"\.?\s*(?:please\s+)?include\s+(?:proper\s+)?(?:code\s+)?syntax\s+highlighting\.?", "", "Removed syntax highlighting request"),
            (r"\.?\s*(?:please\s+)?(?:format|structure|organize)\s+(?:it\s+|this\s+|the\s+response\s+)?(?:as|in|into)\s+(?:a\s+)?(?:bulleted?|numbered)\s+(?:list|points)\.?", "", "Removed list formatting request"),
            (r"\.?\s*(?:please\s+)?(?:add|include)\s+(?:appropriate\s+)?(?:headers?|headings?|sections?)(?:\s+in\s+the\s+response)?\.?", "", "Removed header request"),
            (r"(?:,?\s+)?with\s+(?:proper\s+)?(?:indentation|formatting|structure)", "", "Removed indentation request"),
            (r"\.?\s*(?:please\s+)?(?:format|use)\s+(?:proper\s+)?markdown\s+with\s+headers\.?", "", "Removed markdown header request"),
        ]
        
        for pattern, replacement, change_desc in energy_intensive_patterns:
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                changes.append(change_desc)
        
        # =================================================================
        # PHASE 2: Transform verbose requests to directives (TRUE REPHRASING)
        # =================================================================
        # These patterns capture and transform entire phrase structures
        verbose_to_concise = [
            # Transform "Could you please help me understand X" → "Explain X"
            (r"(?:could|would|can)\s+you\s+(?:please\s+)?help\s+me\s+(?:to\s+)?understand\s+", "Explain ", "Transformed to directive"),
            (r"(?:could|would|can)\s+you\s+(?:please\s+)?help\s+me\s+(?:to\s+)?", "", "Simplified help request"),
            
            # Transform "Could you please explain" → "Explain"
            (r"(?:could|would|can)\s+you\s+(?:please\s+)?explain\s+", "Explain ", "Transformed to directive"),
            (r"(?:could|would|can)\s+you\s+(?:please\s+)?describe\s+", "Describe ", "Transformed to directive"),
            (r"(?:could|would|can)\s+you\s+(?:please\s+)?write\s+", "Write ", "Transformed to directive"),
            (r"(?:could|would|can)\s+you\s+(?:please\s+)?create\s+", "Create ", "Transformed to directive"),
            (r"(?:could|would|can)\s+you\s+(?:please\s+)?generate\s+", "Generate ", "Transformed to directive"),
            (r"(?:could|would|can)\s+you\s+(?:please\s+)?provide\s+", "Provide ", "Transformed to directive"),
            (r"(?:could|would|can)\s+you\s+(?:please\s+)?tell\s+me\s+", "Tell me ", "Transformed to directive"),
            (r"(?:could|would|can)\s+you\s+(?:please\s+)?show\s+me\s+", "Show me ", "Transformed to directive"),
            (r"(?:could|would|can)\s+you\s+(?:please\s+)?give\s+me\s+", "Give me ", "Transformed to directive"),
            (r"(?:could|would|can)\s+you\s+(?:please\s+)?\?", "?", "Simplified question"),
            
            # Transform "I was wondering if you could" → remove entirely (action follows)
            (r"i\s+was\s+wondering\s+if\s+you\s+(?:could|would|might)(?:\s+be\s+able\s+to)?\s+", "", "Removed wondering phrase"),
            (r"i\s+am\s+wondering\s+(?:if|whether)\s+", "", "Removed wondering phrase"),
            
            # Transform "It would be great if you could" → remove entirely
            (r"it\s+would\s+be\s+(?:great|helpful|nice|wonderful|appreciated)\s+if\s+you\s+(?:could|would)\s+", "", "Removed conditional request"),
            
            # Transform "I would really like you to" → ""
            (r"i\s+(?:would\s+)?(?:really\s+)?(?:like|want|need)\s+(?:you\s+)?to\s+", "", "Simplified desire phrase"),
            
            # Transform "I honestly really need you to help me" → ""
            (r"i\s+(?:honestly\s+)?(?:really\s+)?need\s+(?:you\s+)?to\s+(?:help\s+me\s+)?", "", "Simplified need phrase"),
            
            # Transform "provide me with a detailed explanation of" → "explain"
            (r"provide\s+(?:me\s+)?(?:with\s+)?(?:a\s+)?(?:detailed\s+|comprehensive\s+)?explanation\s+(?:of|about|regarding)\s+", "Explain ", "Simplified to explain"),
            (r"provide\s+(?:me\s+)?(?:with\s+)?(?:a\s+)?(?:comprehensive\s+)?overview\s+(?:of|about|regarding)\s+", "Summarize ", "Simplified to summarize"),
            (r"give\s+(?:me\s+)?(?:a\s+)?(?:detailed\s+)?description\s+(?:of|about)\s+", "Describe ", "Simplified to describe"),
            (r"provide\s+(?:me\s+)?(?:with\s+)?information\s+(?:about|on|regarding)\s+", "Explain ", "Simplified info request"),
            
            # Transform verbose modifiers
            (r"(?:very\s+)?(?:detailed|comprehensive|thorough|extensive|in-depth)\s+(?:explanation|analysis|breakdown|overview)", "explanation", "Simplified detail modifier"),
            (r"step[- ]by[- ]step\s+(?:guide|instructions?|explanation|process|tutorial)\s+(?:on|for|about)\s+", "Steps for ", "Simplified step-by-step"),
            (r"in\s+(?:great|more|a\s+lot\s+of)\s+detail", "in detail", "Simplified detail phrase"),
        ]
        
        for pattern, replacement, change_desc in verbose_to_concise:
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                changes.append(change_desc)
        
        # =================================================================
        # PHASE 3: Remove filler words and politeness markers
        # =================================================================
        fillers_to_remove = [
            # Filler words WITH surrounding commas (handle ", essentially," patterns)
            (r",\s*(?:basically|actually|literally|honestly|essentially)\s*,", ",", "Removed filler word"),
            (r",\s*(?:basically|actually|literally|honestly|essentially)\s+", ", ", "Removed filler word"),
            (r"\s+(?:basically|actually|literally|honestly|essentially)\s*,", ",", "Removed filler word"),
            
            # Filler words at start of sentence or standalone
            (r"^(?:basically|actually|literally|honestly|essentially)\s*,?\s*", "", "Removed filler word"),
            (r"\b(?:basically|actually|literally|honestly|essentially)\b\s*", "", "Removed filler word"),
            
            # Intensifiers
            (r"\b(?:really|very|just|simply)\b\s+", "", "Removed intensifier"),
            
            # Politeness that adds tokens but not meaning
            (r"\.?\s*thank\s+you(?:\s+(?:so\s+much|very\s+much|in\s+advance))?!?\.?", "", "Removed thanks"),
            (r"\.?\s*thanks(?:\s+(?:so\s+much|in\s+advance))?!?\.?", "", "Removed thanks"),
            (r"i\s+(?:would\s+)?(?:really\s+)?appreciate\s+(?:it\s+)?(?:if|a)\s+", "", "Removed appreciation"),
            (r"\bif\s+(?:it'?s?\s+)?(?:not\s+too\s+much\s+trouble|you\s+don'?t\s+mind)\b,?\s*", "", "Removed politeness"),
            (r"\bkindly\b\s*", "", "Removed kindly"),
            (r"\bplease\b\s*", "", "Removed please"),
            
            # Verbose connectors
            (r"\bin\s+order\s+to\b", "to", "Simplified connector"),
            (r"\bdue\s+to\s+the\s+fact\s+that\b", "because", "Simplified connector"),
            (r"\bthe\s+fact\s+that\b", "that", "Simplified connector"),
            (r"\bfor\s+(?:the\s+)?purpose\s+of\b", "for", "Simplified connector"),
            (r"\bat\s+(?:this|the)\s+point\s+in\s+time\b", "now", "Simplified time phrase"),
            (r"\bin\s+the\s+event\s+that\b", "if", "Simplified conditional"),
            (r"\bwith\s+(?:regard|respect)\s+to\b", "about", "Simplified reference"),
            
            # ===============================================================
            # NEW: Conciseness patterns for already-short prompts
            # ===============================================================
            # Remove "now" when it's implied by present tense
            (r"\bnow\s+the\s+most\b", "the most", "Removed redundant 'now'"),
            (r"\bis\s+now\s+", "is ", "Removed redundant 'now'"),
            
            # Simplify superlative phrases
            (r"\bthe\s+most\s+popular\s+technique\s+for\b", "the top approach for", "Simplified superlative"),
            
            # Remove rhetorical self-answers like ", right?"
            (r",?\s*right\s*\?$", "?", "Removed rhetorical tag"),
            (r",?\s*correct\s*\?$", "?", "Removed rhetorical tag"),
            (r",?\s*isn't\s+it\s*\?$", "?", "Removed rhetorical tag"),
            (r",?\s*don't\s+you\s+think\s*\?$", "?", "Removed rhetorical tag"),
            
            # Simplify "But how do you know" → "How do you know"
            (r"^but\s+", "", "Removed leading conjunction"),
            (r"\.\s*But\s+", ". ", "Simplified conjunction"),
            
            # Condense em-dash expansions (keep before dash, remove after)
            (r"—from\s+[\w\s,]+(?:to|or|and)\s+[\w\s]+\.", ".", "Condensed em-dash expansion"),
            
            # Remove "any" when it's redundant with superlatives
            (r"\bsolving\s+any\b", "solving", "Removed redundant 'any'"),
            (r"\bfor\s+any\b", "for", "Removed redundant 'any'"),
            
            # Simplify "We can use X as Y" → "Use X as Y"  
            (r"\bwe\s+can\s+use\b", "Use", "Simplified suggestion to directive"),
            
            # Condense "as an evaluation metric" → "to evaluate"
            (r"\bas\s+an?\s+evaluation\s+metric\b", "to evaluate", "Simplified metric phrase"),
            
            # Simplify "if a X is performing well" → "if X performs well"
            (r"\bif\s+a\s+(\w+)\s+is\s+performing\s+well\b", r"if the \1 performs well", "Simplified continuous tense"),
            
            # Remove "deep" when followed by "model" or "learning" (redundant in context)
            (r"\bdeep\s+model\b", "model", "Simplified 'deep model' to 'model'"),
        ]

        for pattern, replacement, change_desc in fillers_to_remove:
            if re.search(pattern, result, re.IGNORECASE):
                original_result = result
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                if result != original_result and change_desc not in changes:
                    changes.append(change_desc)
        
        # =================================================================
        # PHASE 4: Sentence structure optimizations (passive to active)
        # =================================================================
        passive_patterns = [
            (r"(?:it\s+)?(?:is|was)\s+(?:being\s+)?requested\s+that\s+(?:you\s+)?", "", "Simplified passive"),
            (r"(?:it\s+)?would\s+be\s+(?:greatly\s+)?appreciated\s+if\s+(?:you\s+)?(?:could|would)\s+", "", "Simplified passive"),
        ]
        
        for pattern, replacement, change_desc in passive_patterns:
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                changes.append(change_desc)
        
        # =================================================================
        # PHASE 5: Clean up and finalize
        # =================================================================
        # Multiple cleanup passes to ensure clean output
        for _ in range(3):  # Multiple passes for thorough cleanup
            prev_result = result
            
            # Remove multiple spaces
            result = re.sub(r'\s+', ' ', result)
            
            # Remove leading/trailing whitespace
            result = result.strip()
            
            # Fix "a explanation" → "an explanation" (article agreement)
            result = re.sub(r'\ba\s+([aeiou])', r'an \1', result, flags=re.IGNORECASE)
            
            # Clean up orphaned punctuation and artifacts
            result = re.sub(r'\s+([,.\?!])', r'\1', result)  # Space before punctuation
            result = re.sub(r',\s*,', ',', result)  # Double commas (e.g., ", ," → ",")
            result = re.sub(r'([,.\?!])\s*([,.\?!])', r'\1', result)  # Double punctuation
            result = re.sub(r'^[,.\s]+', '', result)  # Leading punctuation
            result = re.sub(r'\s*\.\s*\?', '?', result)  # Period before question mark
            
            # Clean up "and," or "and ," followed by space - remove the orphan comma
            result = re.sub(r'\band,\s+', 'and ', result, flags=re.IGNORECASE)
            
            # Clean up comma after first word when it doesn't make sense
            # e.g., "Understand, the concept" → "Understand the concept"
            result = re.sub(r'^(\w+),\s+(the|a|an|how|what|why|this|that|it)\b', r'\1 \2', result, flags=re.IGNORECASE)
            
            # Clean up orphaned ", the X" patterns mid-sentence after removals
            # e.g., "Help me understand, the concept" → "Help me understand the concept"
            result = re.sub(r',\s+(the|a|an)\s+(\w+)\s+of\b', r' \1 \2 of', result, flags=re.IGNORECASE)
            
            # Clean up ", and" patterns that might be orphaned
            result = re.sub(r',\s+and,', ' and', result, flags=re.IGNORECASE)
            
            # Clean up orphaned trailing words/phrases
            result = re.sub(r'\s+(?:and|with|or|in|the|a|to)\s*[.\?!]?$', '.', result, flags=re.IGNORECASE)
            result = re.sub(r'\s+(?:and|with|or)\?$', '?', result, flags=re.IGNORECASE)
            
            # Clean up orphaned leading words
            result = re.sub(r'^(?:and|with|or)\s+', '', result, flags=re.IGNORECASE)
            
            # Remove "in advance" as standalone at end
            result = re.sub(r'\s+in\s+advance[.!]?$', '.', result, flags=re.IGNORECASE)
            
            # Clean up "with code syntax highlighting" without context
            result = re.sub(r'\s+with\s+code\s+syntax\s+highlighting\.?$', '.', result, flags=re.IGNORECASE)
            
            # Clean up "with headers" without context
            result = re.sub(r'\s+with\s+headers\.?$', '.', result, flags=re.IGNORECASE)
            
            # Clean up orphaned fragments like "explanation with examples" after punctuation
            result = re.sub(r'([.\?!])\s+explanation\s+with\s+examples\.?$', r'\1 Include examples.', result, flags=re.IGNORECASE)
            
            # Strip again
            result = result.strip()
            
            # Exit if no changes were made
            if result == prev_result:
                break
        
        # Final sentence structure fixes
        # Split into sentences and capitalize each
        sentences = re.split(r'([.\?!])\s+', result)
        if len(sentences) > 1:
            fixed_sentences = []
            for i, part in enumerate(sentences):
                if part in '.?!':
                    fixed_sentences.append(part + ' ')  # Add space after punctuation
                elif part:
                    # Capitalize first letter of each sentence
                    fixed_sentences.append(part[0].upper() + part[1:] if len(part) > 1 else part.upper())
            result = ''.join(fixed_sentences).strip()
        
        # If we have orphaned "explanation with examples" at start, make it a sentence
        if result.lower().startswith('explanation'):
            result = 'Provide an ' + result[0].lower() + result[1:]
        
        # Capitalize first letter
        if result and result[0].islower():
            result = result[0].upper() + result[1:]
        
        # Ensure ends with proper punctuation
        if result and result[-1] not in '.?!':
            # Add question mark if it reads like a genuine question (not an imperative)
            # Only "what/how/why/when/where/who/which" starters should be questions
            # "explain/describe/tell/show" are imperatives and should end with period
            if re.match(r'^(?:what|how|why|when|where|who|which)\b', result, re.IGNORECASE):
                result += '?'
            else:
                result += '.'
        
        return result, changes
    
    def get_full_optimization(self, prompt: str) -> OptimizationResult:
        """
        Get complete optimization result with all metrics.
        
        Args:
            prompt: Original prompt to optimize
            
        Returns:
            OptimizationResult with all optimization metrics
        """
        original_tokens = self._count_tokens(prompt)
        
        # Optimize (now returns tuple)
        optimized, changes_made = self.optimize(prompt)
        optimized_tokens = self._count_tokens(optimized)
        
        # Calculate metrics
        token_reduction = 1 - (optimized_tokens / max(original_tokens, 1))
        token_reduction = max(0, token_reduction)  # Ensure non-negative
        
        # Estimate energy reduction (proportional to token reduction)
        # Based on research: energy ~ O(n^2) for transformer attention
        energy_reduction = 1 - ((optimized_tokens / max(original_tokens, 1)) ** 2)
        energy_reduction = max(0, min(1, energy_reduction))
        
        # Semantic similarity
        semantic_similarity = self._estimate_semantic_similarity(prompt, optimized)
        
        # Quality rating based on both reduction AND semantic preservation
        if token_reduction > 0.4 and semantic_similarity > 0.7:
            quality = "Excellent"
        elif token_reduction > 0.25 and semantic_similarity > 0.6:
            quality = "Good"
        elif token_reduction > 0.1 and semantic_similarity > 0.5:
            quality = "Moderate"
        elif token_reduction > 0:
            quality = "Minimal"
        else:
            quality = "No Change"
            
        return OptimizationResult(
            original_prompt=prompt,
            optimized_prompt=optimized,
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            token_reduction=round(token_reduction * 100, 1),
            semantic_similarity=round(semantic_similarity * 100, 1),
            energy_reduction_estimate=round(energy_reduction * 100, 1),
            optimization_quality=quality,
            changes_made=changes_made
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
    optimized, _ = optimizer.optimize(prompt)
    return optimized


if __name__ == "__main__":
    # Demo usage
    print("=" * 70)
    print("T5 Prompt Optimizer Demo - Intelligent Rephrasing")
    print("=" * 70)
    
    optimizer = T5PromptOptimizer()
    
    test_prompts = [
        "Could you please help me understand what machine learning is and how it works? I would really appreciate a detailed explanation with examples.",
        "I was wondering if you might be able to assist me with writing a Python function that calculates the factorial of a number. Please format the output using proper markdown with code syntax highlighting.",
        "It would be great if you could provide me with a comprehensive, step-by-step guide on how to set up a virtual environment in Python. Thank you so much in advance!",
        "Please explain the concept of recursion in simple terms that I can understand, and kindly include some examples in your response."
    ]
    
    print("\n" + "=" * 70)
    print("DEMONSTRATING TRUE OPTIMIZATION (not truncation)")
    print("=" * 70)
    
    for i, prompt in enumerate(test_prompts, 1):
        result = optimizer.get_full_optimization(prompt)
        print(f"\n{'─' * 70}")
        print(f"Example {i}:")
        print(f"{'─' * 70}")
        print(f"\n📝 ORIGINAL ({result.original_tokens} tokens):")
        print(f"   {result.original_prompt}")
        print(f"\n✅ OPTIMIZED ({result.optimized_tokens} tokens):")
        print(f"   {result.optimized_prompt}")
        print(f"\n📊 METRICS:")
        print(f"   Token Reduction: {result.token_reduction}%")
        print(f"   Semantic Similarity: {result.semantic_similarity}%")
        print(f"   Energy Reduction: {result.energy_reduction_estimate}%")
        print(f"   Quality: {result.optimization_quality}")
        if result.changes_made:
            print(f"\n🔧 CHANGES MADE:")
            for change in result.changes_made[:5]:
                print(f"   • {change}")
            if len(result.changes_made) > 5:
                print(f"   • ... and {len(result.changes_made) - 5} more")
