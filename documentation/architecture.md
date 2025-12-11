# Sustainable AI - System Architecture

## Overview

The Sustainable AI system is designed to predict and optimize energy consumption for Large Language Model (LLM) prompts. The architecture follows a modular design pattern with clear separation of concerns.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INTERFACE                          │
│                    (Streamlit Web Application)                  │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        CORE SERVICES                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │    NLP      │  │   Energy    │  │      Anomaly            │ │
│  │   Module    │  │  Predictor  │  │     Detection           │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA & STORAGE                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │   SQLite    │  │   Models    │  │    Training Data        │ │
│  │   Logger    │  │  (.pkl)     │  │     (JSON)              │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
Sustainable_AI_G3-/
├── src/
│   ├── gui/                    # User Interface
│   │   └── app.py             # Streamlit application
│   ├── nlp/                    # NLP Processing
│   │   ├── simplifier.py      # Prompt optimization interface
│   │   ├── prompt_optimizer.py # T5-based optimizer
│   │   ├── semantic_similarity.py # Similarity validation
│   │   ├── nlp_service.py     # Unified NLP service
│   │   ├── complexity_score.py # Token/complexity analysis
│   │   └── train_optimizer.py # Model training script
│   ├── prediction/             # Energy Prediction
│   │   └── estimator.py       # Energy estimation models
│   ├── anomaly/                # Anomaly Detection
│   │   └── detector.py        # Isolation Forest detector
│   ├── optimization/           # Architecture Optimization
│   │   └── ...
│   └── utils/                  # Utilities
│       └── data_logger.py     # Data logging system
├── data/
│   ├── prompt_optimization/    # Training data for T5
│   └── logs/                   # Analysis session logs
├── model/
│   ├── energy_predictor/       # Trained energy models
│   └── prompt_optimizer/       # T5 model checkpoints
├── tests/                      # Unit and integration tests
├── documentation/              # Project documentation
└── reports/                    # Generated reports
```

## Component Details

### 1. User Interface (GUI)

**File:** `src/gui/app.py`

**Technology:** Streamlit with Plotly visualizations

**Features:**

- Prompt input and analysis
- Energy prediction display
- Optimization results visualization
- Side-by-side comparison charts
- Interactive parameter controls

**Key Components:**

- Session state management
- Toast notifications
- Responsive layout with columns
- Interactive Plotly charts (bar, gauge, pie)

### 2. NLP Module

#### 2.1 Prompt Simplifier

**File:** `src/nlp/simplifier.py`

**Purpose:** High-level interface for prompt optimization

**Features:**

- ML-based optimization (T5 model)
- Rule-based fallback
- Full analysis reporting
- Backward compatibility

#### 2.2 T5 Prompt Optimizer

**File:** `src/nlp/prompt_optimizer.py`

**Purpose:** Core T5-based prompt optimization engine with advanced rule-based transformations

**Model:** T5-small (fine-tunable, 60M parameters, CPU-optimized)

**5-Phase Optimization Pipeline:**

```
┌─────────────────────────────────────────────────────────────────┐
│                    OPTIMIZATION PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│ Phase 0   │ Duplicate Content Detection                         │
│           │ - Removes copy-pasted/repeated text                 │
│           │ - Finds repeating patterns (80%+ coverage)          │
│           │ - Deduplicates identical sentences                  │
├───────────┼─────────────────────────────────────────────────────┤
│ Phase 0.5 │ JSON-Loaded Word Replacements                       │
│           │ - 103 verbose phrases → concise equivalents         │
│           │ - 30 word simplifications (utilize → use)           │
│           │ - 30 redundant phrase fixes                         │
│           │ - Configurable via word_replacements.json           │
├───────────┼─────────────────────────────────────────────────────┤
│ Phase 1   │ Energy-Intensive Feature Removal                    │
│           │ - Formatting requests (JSON, Markdown, HTML)        │
│           │ - Syntax highlighting requests                      │
│           │ - List/header formatting requests                   │
├───────────┼─────────────────────────────────────────────────────┤
│ Phase 2   │ Verbose-to-Concise Transformations                  │
│           │ - "Could you please help me" → ""                   │
│           │ - "I was wondering if" → ""                         │
│           │ - Rhetorical tag removal (", right?")               │
│           │ - Em-dash expansion condensation                    │
├───────────┼─────────────────────────────────────────────────────┤
│ Phase 3   │ Filler & Politeness Removal                         │
│           │ - Filler words (basically, actually, literally)     │
│           │ - Intensifiers (really, very, just)                 │
│           │ - Politeness markers (please, kindly, thanks)       │
├───────────┼─────────────────────────────────────────────────────┤
│ Phase 4   │ Passive to Active Voice Conversion                  │
│           │ - "It was requested that you" → ""                  │
│           │ - "It would be appreciated if" → ""                 │
├───────────┼─────────────────────────────────────────────────────┤
│ Phase 5   │ Multi-Pass Cleanup                                  │
│           │ - Article agreement (a → an before vowels)          │
│           │ - Sentence capitalization                           │
│           │ - Orphaned fragment removal                         │
│           │ - Punctuation normalization                         │
└───────────┴─────────────────────────────────────────────────────┘
```

**Configuration Files:**

| File | Purpose |
|------|---------|
| `model/prompt_optimizer/word_replacements.json` | Verbose phrases, word simplifications, filler words, redundant phrases |

**Features:**

- Tokenization and optimization
- Batch processing
- Duplicate content detection
- JSON-configurable word replacements
- Fallback optimization
- Training capabilities

**Optimization Quality Metrics:**

- Token reduction percentage
- Energy reduction estimate
- Semantic similarity preservation
- Changes made (detailed list)

#### 2.3 Semantic Similarity

**File:** `src/nlp/semantic_similarity.py`

**Purpose:** Validate meaning preservation

**Models:**

- Primary: Sentence-Transformers (all-MiniLM-L6-v2)
- Fallback: TF-IDF with cosine similarity

**Features:**

- Similarity scoring
- Meaning preservation validation
- Intent and action preservation checks

#### 2.4 NLP Service

**File:** `src/nlp/nlp_service.py`

**Purpose:** Unified interface combining all NLP components

**Quality Score Calculation:**

```
quality_score = token_score + similarity_score + intent_score + action_score

Where:
- token_score: min(token_reduction%, 80) / 80 × 30  (max 30 points)
- similarity_score: semantic_similarity × 40        (max 40 points)
- intent_score: 15 if preserved, else 5            (max 15 points)
- action_score: 15 if preserved, else 5            (max 15 points)
```

### 3. Energy Prediction Engine

**File:** `src/prediction/estimator.py`

**Purpose:** Predict energy consumption (kWh) and carbon footprint

**Models:**

- LinearRegression (default)
- RandomForest

**Features:**

- Token-based feature extraction
- Synthetic data training
- Energy and carbon estimation
- Training visualization

**Formula:**

```
energy_kwh = f(tokens, layers, training_hours, flops)
carbon_kg = energy_kwh × 0.475 (CO₂ per kWh factor)
```

### 4. Anomaly Detection

**File:** `src/anomaly/detector.py`

**Purpose:** Flag unusual resource demands

**Algorithm:** Isolation Forest / One-Class SVM

**Features:**

- Pattern recognition
- Outlier detection
- Usage anomaly flagging

### 5. Data Logging System

**File:** `src/utils/data_logger.py`

**Purpose:** Store analysis sessions for transparency and benchmarking

**Database Options:**

- SQLite (default, lightweight)
- PostgreSQL (scalable)
- JSON (fallback)

**Logged Data:**
| Field | Description |
|-------|-------------|
| session_id | Unique identifier |
| timestamp | Analysis time |
| prompt_hash | Privacy-safe prompt hash |
| original_tokens | Original token count |
| optimized_tokens | Optimized token count |
| energy_kwh | Original energy estimate |
| carbon_kg | Original carbon footprint |
| optimized_energy_kwh | Optimized energy |
| semantic_similarity | Meaning preservation score |
| quality_score | Overall optimization quality |
| model_type | ML model used |
| recommendation_chosen | User's selection |

**Features:**

- Daily statistics aggregation
- Report generation (text, JSON, markdown)
- CSV/JSON export
- Privacy-safe logging (hash prompts)

## Data Flow

```
1. User Input
   └─► Prompt + Parameters (layers, training_hours, flops)

2. NLP Processing
   ├─► Token counting (ComplexityAnalyzer)
   ├─► Prompt optimization (T5PromptOptimizer)
   └─► Semantic validation (SemanticSimilarity)

3. Energy Prediction
   ├─► Feature extraction
   ├─► Model inference
   └─► Carbon calculation

4. Anomaly Detection
   └─► Flag unusual patterns

5. Results Display
   ├─► Original vs Optimized metrics
   ├─► Visualizations
   └─► Recommendations

6. Data Logging
   └─► Session data to SQLite
```

## Technology Stack

| Component       | Technology                              |
| --------------- | --------------------------------------- |
| Frontend        | Streamlit                               |
| Visualization   | Plotly, Matplotlib                      |
| ML Framework    | PyTorch, Transformers                   |
| NLP Models      | T5-small, Sentence-Transformers         |
| ML Models       | scikit-learn (RF, LR, Isolation Forest) |
| Database        | SQLite, PostgreSQL                      |
| Data Processing | Pandas, NumPy                           |

## API Reference

### PromptSimplifier

```python
from src.nlp.simplifier import PromptSimplifier

simplifier = PromptSimplifier(use_ml_model=True)

# Simple optimization
optimized = simplifier.optimize("Your prompt here")

# Full analysis
analysis = simplifier.get_full_analysis("Your prompt here")
# Returns: {
#   'original': str,
#   'optimized': str,
#   'original_tokens': int,
#   'optimized_tokens': int,
#   'token_reduction_pct': float,
#   'energy_reduction_pct': float,
#   'semantic_similarity': float,
#   'quality_score': float,
#   'suggestions': list
# }
```

### EnergyEstimator

```python
from src.prediction.estimator import EnergyEstimator

estimator = EnergyEstimator(model_type="LinearRegression")

results = estimator.estimate(
    prompt="Your prompt",
    layers=12,
    training_hours=5.0,
    flops_str="1.5e18"
)
# Returns: {
#   'energy_kwh': float,
#   'carbon_kg': float,
#   'token_count': int
# }
```

### DataLogger

```python
from src.utils.data_logger import DataLogger

logger = DataLogger(db_type="sqlite")

session_id = logger.log_analysis(
    prompt_text="...",
    original_tokens=50,
    optimized_tokens=35,
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
    flops="1.5e18"
)

# Generate report
report = logger.generate_report(format="markdown")
```

## Security Considerations

1. **Privacy Protection:**

   - Prompts can be logged as hashes only
   - No PII storage by default

2. **Data Integrity:**

   - Session IDs are SHA-256 hashed
   - Database indices for fast queries

3. **Fallback Mechanisms:**
   - Rule-based optimization if ML fails
   - TF-IDF similarity if Transformers unavailable
   - JSON logging if database fails

## Performance Considerations

1. **Model Loading:**

   - Models loaded once at startup
   - Lazy initialization for optional components

2. **Optimization:**

   - Batch processing supported
   - Caching of embeddings

3. **Database:**
   - SQLite for single-instance deployment
   - PostgreSQL for multi-instance scaling

## Future Enhancements

1. **Real-time Training:** Online learning from user feedback
2. **Multi-language Support:** Extend to non-English prompts
3. **API Endpoint:** REST API for integration
4. **Dashboard:** Admin dashboard for statistics
5. **Model Selection:** User-selectable optimization models
