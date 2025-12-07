# Sustainable AI - API Reference

## Overview

This document provides detailed API documentation for all modules in the Sustainable AI system.

---

## NLP Module

### PromptSimplifier

**Location:** `src/nlp/simplifier.py`

The main interface for prompt optimization, providing both ML-based and rule-based optimization.

#### Constructor

```python
PromptSimplifier(use_ml_model: bool = True)
```

| Parameter    | Type | Default | Description                |
| ------------ | ---- | ------- | -------------------------- |
| use_ml_model | bool | True    | Whether to use T5 ML model |

#### Methods

##### optimize(text: str) -> str

Optimize a prompt to be more energy efficient.

```python
simplifier = PromptSimplifier()
optimized = simplifier.optimize("Could you please help me understand Python?")
# Returns: "Explain Python."
```

| Parameter   | Type | Description             |
| ----------- | ---- | ----------------------- |
| text        | str  | The prompt to optimize  |
| **Returns** | str  | Optimized prompt string |

##### get_full_analysis(text: str) -> Dict

Get comprehensive optimization analysis.

```python
analysis = simplifier.get_full_analysis("Could you please explain machine learning?")
```

**Returns Dictionary:**

| Key                       | Type      | Description                   |
| ------------------------- | --------- | ----------------------------- |
| original                  | str       | Original prompt               |
| optimized                 | str       | Optimized prompt              |
| original_tokens           | int       | Original token count          |
| optimized_tokens          | int       | Optimized token count         |
| token_reduction_pct       | float     | Percentage reduction          |
| energy_reduction_pct      | float     | Estimated energy savings      |
| semantic_similarity       | float     | Meaning preservation (0-100)  |
| similarity_interpretation | str       | Human-readable interpretation |
| meaning_preserved         | bool      | Whether meaning is preserved  |
| quality_score             | float     | Overall quality (0-100)       |
| optimization_quality      | str       | Quality category              |
| suggestions               | List[str] | Improvement suggestions       |

##### is_ml_available() -> bool

Check if ML model is available.

```python
if simplifier.is_ml_available():
    print("Using T5 model")
else:
    print("Using rule-based fallback")
```

---

### T5PromptOptimizer

**Location:** `src/nlp/prompt_optimizer.py`

Core T5-based prompt optimization engine.

#### Constructor

```python
T5PromptOptimizer(
    model_name: str = "t5-small",
    model_path: Optional[str] = None
)
```

| Parameter  | Type | Default    | Description              |
| ---------- | ---- | ---------- | ------------------------ |
| model_name | str  | "t5-small" | Hugging Face model name  |
| model_path | str  | None       | Path to fine-tuned model |

#### Methods

##### optimize(prompt: str, max_length: int = 128) -> str

Optimize a single prompt.

```python
optimizer = T5PromptOptimizer()
result = optimizer.optimize("Could you please explain recursion?")
```

##### get_full_optimization(prompt: str) -> OptimizationResult

Get full optimization with metrics.

```python
result = optimizer.get_full_optimization("Please help me understand AI")
print(f"Optimized: {result.optimized_prompt}")
print(f"Token reduction: {result.token_reduction}%")
```

**OptimizationResult Attributes:**

| Attribute                 | Type  | Description              |
| ------------------------- | ----- | ------------------------ |
| original_prompt           | str   | Original input           |
| optimized_prompt          | str   | Optimized output         |
| original_tokens           | int   | Original token count     |
| optimized_tokens          | int   | Optimized token count    |
| token_reduction           | float | Percentage reduction     |
| energy_reduction_estimate | float | Estimated energy savings |
| optimization_quality      | str   | Quality category         |

##### batch_optimize(prompts: List[str]) -> List[str]

Optimize multiple prompts.

```python
prompts = ["Explain AI", "What is ML?", "Help me code"]
optimized = optimizer.batch_optimize(prompts)
```

##### train(epochs: int = 10, batch_size: int = 8, learning_rate: float = 3e-4) -> Dict

Train the optimizer on custom data.

```python
metrics = optimizer.train(epochs=5)
print(f"Final loss: {metrics['train_loss']}")
```

---

### SemanticSimilarity

**Location:** `src/nlp/semantic_similarity.py`

Validates meaning preservation between original and optimized prompts.

#### Constructor

```python
SemanticSimilarity(model_name: str = "all-MiniLM-L6-v2")
```

#### Methods

##### compute_similarity(text1: str, text2: str) -> SimilarityResult

Compute similarity between two texts.

```python
similarity = SemanticSimilarity()
result = similarity.compute_similarity(
    "Explain machine learning",
    "What is machine learning?"
)
print(f"Score: {result.score}")  # 0.0 to 1.0
print(f"Interpretation: {result.interpretation}")
```

**SimilarityResult Attributes:**

| Attribute      | Type  | Description                   |
| -------------- | ----- | ----------------------------- |
| score          | float | Similarity score (0-1)        |
| interpretation | str   | Human-readable interpretation |

##### validate_optimization(original: str, optimized: str) -> Dict

Validate an optimization result.

```python
validation = similarity.validate_optimization(
    "Could you please help me?",
    "Help me."
)
if validation['is_valid']:
    print("Optimization is valid")
```

---

### EnhancedPromptValidator

**Location:** `src/nlp/semantic_similarity.py`

Advanced validation with intent and action preservation.

#### Methods

##### validate_prompt_optimization(original: str, optimized: str) -> Dict

```python
validator = EnhancedPromptValidator()
result = validator.validate_prompt_optimization(
    "Could you please write a Python function?",
    "Write Python function."
)
```

**Returns:**

| Key                       | Type  | Description             |
| ------------------------- | ----- | ----------------------- |
| semantic_similarity       | float | Similarity score (0-1)  |
| similarity_interpretation | str   | Interpretation text     |
| is_valid                  | bool  | Overall validity        |
| intent_preserved          | bool  | Core intent maintained  |
| action_preserved          | bool  | Action verbs maintained |
| quality_score             | float | Combined quality (0-1)  |

---

### NLPService

**Location:** `src/nlp/nlp_service.py`

Unified interface combining all NLP components.

#### Constructor

```python
NLPService()
```

#### Methods

##### optimize_prompt(prompt: str) -> ComprehensiveOptimizationResult

Full optimization with all metrics.

```python
service = NLPService()
result = service.optimize_prompt("Please help me understand neural networks")

print(f"Original: {result.original_prompt}")
print(f"Optimized: {result.optimized_prompt}")
print(f"Quality Score: {result.quality_score}")
```

**ComprehensiveOptimizationResult Attributes:**

| Attribute                 | Type      | Description             |
| ------------------------- | --------- | ----------------------- |
| original_prompt           | str       | Original input          |
| original_tokens           | int       | Token count             |
| original_complexity       | str       | Complexity level        |
| optimized_prompt          | str       | Optimized output        |
| optimized_tokens          | int       | Token count             |
| token_reduction_pct       | float     | Percentage reduction    |
| energy_reduction_pct      | float     | Energy savings estimate |
| semantic_similarity       | float     | Similarity (0-100)      |
| similarity_interpretation | str       | Interpretation          |
| meaning_preserved         | bool      | Is meaning preserved    |
| optimization_quality      | str       | Quality category        |
| quality_score             | float     | Overall score (0-100)   |
| suggestions               | List[str] | Suggestions             |

##### batch_optimize(prompts: List[str]) -> List[ComprehensiveOptimizationResult]

Optimize multiple prompts.

##### get_energy_comparison(original: str, optimized: str) -> Dict

Compare energy between two prompts.

```python
comparison = service.get_energy_comparison(
    "This is a long verbose prompt",
    "Short prompt"
)
print(f"Energy saved: {comparison['energy_saved_kwh']} kWh")
```

##### get_similarity(text1: str, text2: str) -> float

Get similarity score between two texts.

---

## Prediction Module

### EnergyEstimator

**Location:** `src/prediction/estimator.py`

Predicts energy consumption and carbon footprint.

#### Constructor

```python
EnergyEstimator(model_type: str = "LinearRegression")
```

| Parameter  | Type | Options                            | Description     |
| ---------- | ---- | ---------------------------------- | --------------- |
| model_type | str  | "LinearRegression", "RandomForest" | ML model to use |

#### Methods

##### estimate(prompt: str, layers: int, training_hours: float, flops_str: str) -> Dict

Estimate energy for a prompt.

```python
estimator = EnergyEstimator(model_type="LinearRegression")
result = estimator.estimate(
    prompt="Explain machine learning",
    layers=12,
    training_hours=5.0,
    flops_str="1.5e18"
)

print(f"Energy: {result['energy_kwh']} kWh")
print(f"Carbon: {result['carbon_kg']} kgCO₂")
print(f"Tokens: {result['token_count']}")
```

**Returns:**

| Key         | Type  | Description        |
| ----------- | ----- | ------------------ |
| energy_kwh  | float | Energy consumption |
| carbon_kg   | float | Carbon footprint   |
| token_count | int   | Prompt token count |

##### get_training_plot(layers: int) -> Figure

Get matplotlib figure of training performance.

```python
fig = estimator.get_training_plot(layers=12)
plt.show()
```

---

## Utilities Module

### DataLogger

**Location:** `src/utils/data_logger.py`

Logging and reporting for analysis sessions.

#### Constructor

```python
DataLogger(
    db_type: str = "sqlite",
    db_path: Optional[str] = None,
    pg_config: Optional[Dict] = None
)
```

| Parameter | Type | Options                        | Description          |
| --------- | ---- | ------------------------------ | -------------------- |
| db_type   | str  | "sqlite", "postgresql", "json" | Database backend     |
| db_path   | str  | -                              | Custom database path |
| pg_config | Dict | -                              | PostgreSQL config    |

#### Methods

##### log_analysis(...) -> str

Log an analysis session.

```python
logger = DataLogger()
session_id = logger.log_analysis(
    prompt_text="Test prompt",
    original_tokens=10,
    optimized_tokens=7,
    token_reduction_pct=30.0,
    energy_kwh=1.5,
    carbon_kg=0.7,
    optimized_energy_kwh=1.2,
    optimized_carbon_kg=0.56,
    semantic_similarity=85.0,
    quality_score=75.0,
    model_type="LinearRegression",
    layers=12,
    training_hours=5.0,
    flops="1.5e18",
    store_full_prompt=False  # Privacy protection
)
```

**Parameters:**

| Parameter            | Type  | Description          |
| -------------------- | ----- | -------------------- |
| prompt_text          | str   | Original prompt      |
| original_tokens      | int   | Token count          |
| optimized_tokens     | int   | Optimized count      |
| token_reduction_pct  | float | Reduction percentage |
| energy_kwh           | float | Original energy      |
| carbon_kg            | float | Original carbon      |
| optimized_energy_kwh | float | Optimized energy     |
| optimized_carbon_kg  | float | Optimized carbon     |
| semantic_similarity  | float | Similarity score     |
| quality_score        | float | Quality score        |
| model_type           | str   | ML model used        |
| layers               | int   | Number of layers     |
| training_hours       | float | Training time        |
| flops                | str   | FLOPs value          |
| store_full_prompt    | bool  | Store full text?     |

##### get_session_log(session_id: str) -> Optional[Dict]

Retrieve a specific session.

##### get_recent_logs(limit: int = 100) -> List[Dict]

Get recent analysis logs.

##### get_statistics(start_date: str = None, end_date: str = None) -> Dict

Get aggregate statistics.

```python
stats = logger.get_statistics()
print(f"Total analyses: {stats['total_analyses']}")
print(f"Energy saved: {stats['total_energy_saved_kwh']} kWh")
```

##### generate_report(format: str = "text") -> str

Generate summary report.

```python
# Text format
print(logger.generate_report(format="text"))

# Markdown format
markdown = logger.generate_report(format="markdown")

# JSON format
json_data = logger.generate_report(format="json")
```

##### export_logs(filepath: str, format: str = "csv") -> bool

Export logs to file.

```python
logger.export_logs("analysis_export.csv", format="csv")
logger.export_logs("analysis_export.json", format="json")
```

---

## Data Classes

### AnalysisLog

**Location:** `src/utils/data_logger.py`

```python
@dataclass
class AnalysisLog:
    session_id: str
    timestamp: str
    prompt_text: str
    prompt_hash: str
    original_tokens: int
    optimized_tokens: int
    token_reduction_pct: float
    energy_kwh: float
    carbon_kg: float
    optimized_energy_kwh: float
    optimized_carbon_kg: float
    energy_saved_pct: float
    semantic_similarity: float
    quality_score: float
    model_type: str
    layers: int
    training_hours: float
    flops: str
    recommendation_chosen: Optional[str]
```

---

## Error Handling

### Common Exceptions

```python
try:
    result = simplifier.optimize(prompt)
except Exception as e:
    # Falls back to rule-based optimization automatically
    print(f"ML failed, using fallback: {e}")
```

### Fallback Behavior

| Component    | Primary               | Fallback   |
| ------------ | --------------------- | ---------- |
| Optimization | T5 Model              | Rule-based |
| Similarity   | Sentence-Transformers | TF-IDF     |
| Database     | SQLite                | JSON file  |

---

## Configuration

### Environment Variables

| Variable                  | Description   | Default                    |
| ------------------------- | ------------- | -------------------------- |
| SUSTAINABLE_AI_DB_PATH    | Database path | data/logs/analysis_logs.db |
| SUSTAINABLE_AI_MODEL_PATH | Model path    | model/prompt_optimizer/    |

### Model Paths

```python
# Default paths
PROJECT_ROOT/
├── model/
│   ├── energy_predictor/
│   │   └── energy_model_*.pkl
│   └── prompt_optimizer/
│       └── checkpoint/
└── data/
    ├── prompt_optimization/
    │   ├── training_data.json
    │   └── extended_training_data.json
    └── logs/
        └── analysis_logs.db
```
