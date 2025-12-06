# Sustainable AI - Project Summary

## Executive Summary

**Project:** Sustainable AI - Transparency and Energy-Efficient Prompt Engineering with Machine Learning

**Course:** CSCN8010 - Machine Learning

**Team Members:**

- Jarius Bedward (8841640)
- Mostafa Allahmoradi (9087818)
- Oluwafemi Lawal (8967308)
- Jatinder Pal Singh (9083762)

---

## Problem Statement

The rapid expansion of AI data centers is creating a significant environmental impact:

- **1,240+ new data centers** built in 2025 alone
- **EU regulations** require energy usage reporting by August 2026
- Limited transparency in AI energy consumption
- Users have no visibility into the computational cost of their prompts

---

## Solution

A Machine Learning framework that:

1. **Predicts** energy consumption of LLM prompts
2. **Optimizes** prompts for energy efficiency
3. **Validates** that optimizations preserve meaning
4. **Detects** anomalies in resource usage
5. **Logs** all analysis for transparency

---

## Key Features Implemented

### 1. User Interface (GUI) ✅

- **Technology:** Streamlit with Plotly visualizations
- **Features:**
  - Prompt input and configuration
  - Real-time energy analysis
  - Interactive visualizations (bar charts, gauges, pie charts)
  - Side-by-side comparison of original vs optimized

### 2. NLP Module ✅

- **Prompt Optimization:**

  - T5-small transformer model for intelligent rewriting
  - Rule-based fallback for reliability
  - 195+ training examples across multiple categories

- **Semantic Similarity:**

  - Sentence-Transformers (all-MiniLM-L6-v2)
  - TF-IDF fallback
  - Intent and action preservation validation

- **Quality Score Calculation:**
  - Token reduction (30%)
  - Semantic similarity (40%)
  - Intent preservation (15%)
  - Action preservation (15%)

### 3. Energy Prediction Engine ✅

- **Models:** LinearRegression, RandomForest
- **Features:**
  - Synthetic training data generation
  - Multi-feature prediction (tokens, layers, FLOPs, training time)
  - Carbon footprint calculation (0.475 kgCO₂/kWh)

### 4. Anomaly Detection ✅

- **Algorithm:** Isolation Forest
- **Purpose:** Flag unusual resource demands
- **Features:** Pattern recognition, outlier detection

### 5. Data Logging & Reporting ✅

- **Database:** SQLite (with PostgreSQL support)
- **Logged Data:**
  - Prompt hash (privacy-safe)
  - Token counts (original/optimized)
  - Energy estimates
  - Carbon footprint
  - Quality scores
  - User selections
- **Features:**
  - Daily statistics aggregation
  - Report generation (text, JSON, markdown)
  - CSV/JSON export

---

## Technical Architecture

```
┌─────────────────────────────────────────┐
│           Streamlit GUI                 │
│    (User Input & Visualization)         │
└─────────────────┬───────────────────────┘
                  │
    ┌─────────────┴─────────────┐
    │                           │
    ▼                           ▼
┌─────────┐              ┌──────────────┐
│   NLP   │              │    Energy    │
│ Module  │              │  Predictor   │
└────┬────┘              └──────┬───────┘
     │                          │
     ▼                          │
┌─────────────┐                 │
│  Semantic   │                 │
│ Similarity  │                 │
└─────────────┘                 │
                                │
    ┌───────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│          Data Logger (SQLite)           │
│   Sessions, Statistics, Reports         │
└─────────────────────────────────────────┘
```

---

## Technology Stack

| Component     | Technology                           |
| ------------- | ------------------------------------ |
| Frontend      | Streamlit 1.28+                      |
| Visualization | Plotly, Matplotlib                   |
| ML Framework  | PyTorch, scikit-learn                |
| NLP Models    | Hugging Face Transformers (T5-small) |
| Embeddings    | Sentence-Transformers                |
| Database      | SQLite (PostgreSQL optional)         |
| Testing       | pytest, unittest                     |

---

## File Structure

```
Sustainable_AI_G3-/
├── src/
│   ├── gui/app.py              # Streamlit application
│   ├── nlp/
│   │   ├── simplifier.py       # Prompt optimization interface
│   │   ├── prompt_optimizer.py # T5-based optimizer
│   │   ├── semantic_similarity.py # Validation
│   │   ├── nlp_service.py      # Unified service
│   │   ├── complexity_score.py # Token analysis
│   │   └── train_optimizer.py  # Training script
│   ├── prediction/estimator.py # Energy prediction
│   ├── anomaly/detector.py     # Anomaly detection
│   └── utils/data_logger.py    # Logging system
├── data/
│   ├── prompt_optimization/    # Training data (195+ examples)
│   └── logs/                   # Session logs
├── model/                      # Trained models
├── tests/
│   ├── test_nlp.py            # Unit tests (47 tests)
│   └── test_integration.py    # Integration tests (20 tests)
├── documentation/
│   ├── architecture.md        # System architecture
│   ├── user_manual.md         # User guide
│   ├── api_reference.md       # API documentation
│   └── project_summary.md     # This document
└── requirements.txt           # Dependencies
```

---

## Test Coverage

### Unit Tests (test_nlp.py)

- ComplexityAnalyzer: 5 tests ✅
- SemanticSimilarity: 10 tests ✅
- EnhancedPromptValidator: 4 tests ✅
- PromptSimplifier: 8 tests ✅
- T5PromptOptimizer: 20 tests (skipped without training)

### Integration Tests (test_integration.py)

- NLPServiceIntegration: 6 tests
- SimplifierIntegration: 3 tests ✅
- EnergyEstimatorIntegration: 1 test ✅
- TrainingDataIntegration: 3 tests ✅
- EndToEndOptimization: 3 tests ✅
- SemanticPreservation: 2 tests
- PerformanceMetrics: 2 tests

**Result:** 37 passed, 30 skipped (model not trained)

---

## Sample Results

### Input

```
Prompt: "Could you please help me understand the concept of
machine learning and artificial intelligence in simple terms?"

Layers: 12
Training Time: 5.0 hours
FLOPs: 1.5e18
```

### Output

```
Original:
- Energy: 1.29 kWh
- Carbon: 0.61 kgCO₂
- Tokens: 61

Optimized:
- Prompt: "Explain machine learning and AI simply."
- Energy: 0.89 kWh
- Carbon: 0.42 kgCO₂
- Tokens: 38

Savings:
- Token Reduction: 37.7%
- Energy Saved: 31.0%
- Semantic Similarity: 87%
- Quality Score: 78/100
```

---

## How to Run

```bash
# 1. Clone repository
git clone <repo-url>
cd Sustainable_AI_G3-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run application
cd src/gui
streamlit run app.py

# 4. Run tests
cd ../..
python -m pytest tests/ -v
```

---

## Future Enhancements

1. **Real-time Model Training:** Online learning from user feedback
2. **Multi-language Support:** Non-English prompt optimization
3. **REST API:** Enable integration with other systems
4. **Admin Dashboard:** Aggregate statistics visualization
5. **Model Selection:** User-selectable optimization models
6. **Cloud Deployment:** AWS/GCP/Azure hosting

---

## Conclusion

The Sustainable AI project successfully demonstrates a proof-of-concept for linking NLP inputs directly to physical energy estimates. The system:

- ✅ Predicts energy consumption for any text prompt
- ✅ Generates optimized alternatives preserving meaning
- ✅ Validates semantic similarity
- ✅ Detects anomalous resource usage
- ✅ Logs all analysis for transparency
- ✅ Provides intuitive visualizations

By making AI energy consumption visible and actionable, this project contributes to the broader goal of sustainable AI development.

---

## References

1. Hugging Face Transformers Documentation
2. Sentence-Transformers Library
3. Streamlit Documentation
4. scikit-learn User Guide
5. EU AI Act - Energy Transparency Requirements
