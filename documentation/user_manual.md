# Sustainable AI - User Manual

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Using the Application](#using-the-application)
4. [Understanding Results](#understanding-results)
5. [Advanced Features](#advanced-features)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

---

## Introduction

### What is Sustainable AI?

Sustainable AI is a machine learning-powered tool that helps you understand and reduce the environmental impact of your AI prompts. It predicts energy consumption and suggests optimized alternatives that maintain your original intent while using fewer computational resources.

### Why Does This Matter?

- **Environmental Impact:** AI data centers consume significant energy
- **EU Regulations:** Energy usage reporting required by August 2026
- **Cost Savings:** Optimized prompts can reduce computational costs
- **Transparency:** Understand the true cost of your AI interactions

### Key Features

| Feature                | Description                             |
| ---------------------- | --------------------------------------- |
| 🔋 Energy Prediction   | Estimate kWh consumption for any prompt |
| 🌱 Carbon Footprint    | Calculate CO₂ emissions                 |
| ✨ Prompt Optimization | Get energy-efficient alternatives       |
| 📊 Visualizations      | Interactive charts and graphs           |
| 🔍 Anomaly Detection   | Flag unusual resource demands           |

---

## Getting Started

### System Requirements

- Python 3.9 or higher
- 4GB RAM minimum (8GB recommended)
- Modern web browser (Chrome, Firefox, Edge)
- Internet connection (for initial model download)

### Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd Sustainable_AI_G3-
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application:**

   ```bash
   cd src/gui
   streamlit run app.py
   ```

4. **Access the Interface:**
   - Open your browser to `http://localhost:8501`

### First-Time Setup

On first run, the system will:

- Initialize the SQLite database for logging
- Load energy prediction models
- Prepare NLP optimization components

This may take 30-60 seconds on first launch.

---

## Using the Application

### Main Interface Overview

```
┌────────────────────────────────────────────────────────────┐
│  🌱 Sustainable AI - Energy Predictor                      │
├──────────────────┬─────────────────────────────────────────┤
│   SIDEBAR        │          MAIN AREA                      │
│                  │                                         │
│  Model Config:   │   Enter Prompt Context:                │
│  • Layers        │   ┌─────────────────────────────────┐  │
│  • Training Time │   │ Your prompt here...             │  │
│  • FLOPs         │   └─────────────────────────────────┘  │
│  • Model Type    │                                         │
│                  │   [🚀 Analyze Consumption]              │
│                  │                                         │
│                  │   📊 Analysis Report                    │
│                  │   🧠 Prompt Optimization Analysis       │
│                  │                                         │
└──────────────────┴─────────────────────────────────────────┘
```

### Step-by-Step Guide

#### Step 1: Configure Model Parameters (Sidebar)

| Parameter            | Description               | Default          |
| -------------------- | ------------------------- | ---------------- |
| **Number of Layers** | LLM architecture depth    | 12               |
| **Training Time**    | Expected training hours   | 5.0              |
| **FLOPs**            | Floating point operations | 1.5e18           |
| **Model Type**       | Prediction algorithm      | LinearRegression |

**Tip:** If unsure, use the defaults. They're calibrated for typical LLM usage.

#### Step 2: Enter Your Prompt

Type or paste your prompt into the text area. The system works with prompts of any length, from short commands to detailed instructions.

**Examples:**

- Short: `"Explain recursion"`
- Medium: `"Could you please help me understand machine learning?"`
- Long: `"I would be extremely grateful if you could provide a comprehensive explanation of the differences between supervised and unsupervised learning, including practical examples."`

#### Step 3: Analyze

Click the **🚀 Analyze Consumption** button to:

1. Calculate energy consumption for your original prompt
2. Generate an optimized version
3. Compare the results

#### Step 4: Review Results

The results appear in multiple sections:

- **Analysis Report** - Energy and token metrics
- **Prompt Optimization Analysis** - Detailed optimization results
- **Visualizations** - Interactive charts

---

## Understanding Results

### Analysis Report Section

```
📊 Analysis Report

| Metric           | Original | Optimized |
|------------------|----------|-----------|
| Predicted Energy | 1.29 kWh | 1.02 kWh  |
| Carbon Footprint | 0.61 kgCO₂ | 0.48 kgCO₂ |
| Tokens          | 61       | 42        |
```

### Key Metrics Explained

#### Energy Metrics

| Metric                       | What It Means                    | Good Value      |
| ---------------------------- | -------------------------------- | --------------- |
| **Predicted Energy (kWh)**   | Kilowatt-hours to process prompt | Lower is better |
| **Carbon Footprint (kgCO₂)** | CO₂ emissions estimate           | Lower is better |
| **Energy Saved (%)**         | Reduction from optimization      | >10% is good    |

#### Optimization Metrics

| Metric                      | What It Means                            | Good Value |
| --------------------------- | ---------------------------------------- | ---------- |
| **Token Reduction (%)**     | How much shorter the optimized prompt is | 20-50%     |
| **Semantic Similarity (%)** | How well meaning is preserved            | >70%       |
| **Quality Score**           | Overall optimization rating              | >60/100    |

### Quality Score Breakdown

The Quality Score (0-100) is calculated from:

| Component           | Points | Description                |
| ------------------- | ------ | -------------------------- |
| Token Reduction     | 0-30   | Based on % tokens removed  |
| Semantic Similarity | 0-40   | How well meaning preserved |
| Intent Preserved    | 5-15   | Core question/task intact  |
| Action Preserved    | 5-15   | Verbs/commands maintained  |

**Interpretation:**

- **80-100:** Excellent - Use optimized prompt confidently
- **60-79:** Good - Review optimized prompt
- **40-59:** Fair - Consider manual adjustments
- **0-39:** Review needed - Original may be better

### When No Optimization Occurs

If your prompt is already efficient:

- Semantic Similarity = 100%
- Quality Score = 100 (short prompts) or 50 (no changes possible)
- Message: "Prompt already optimized - no changes needed"

**This is a good thing!** It means your prompt is already well-crafted.

---

## Advanced Features

### Full Architecture Optimization

After initial analysis, click **✨ Full Architecture Optimization** to:

- Apply architecture-level optimizations
- Reduce layer count and training time
- See comprehensive savings

### Visualizations

#### Token Count Comparison

Bar chart showing original vs. optimized token counts.

#### Quality Gauge

Speedometer-style gauge showing optimization quality (0-100).

#### Energy Distribution Pie

Shows energy saved vs. energy used after optimization.

#### Energy & Carbon Bar Chart

Side-by-side comparison of original and optimized energy/carbon.

### Data Export

Analysis data is automatically logged. Access via:

```python
from src.utils.data_logger import DataLogger
logger = DataLogger()
report = logger.generate_report(format="markdown")
```

---

## Troubleshooting

### Common Issues

#### "ML model not available"

**Cause:** Keras/TensorFlow compatibility issue

**Solution:**

```bash
pip install tf-keras
```

Or use rule-based optimization (automatic fallback).

#### Slow First Load

**Cause:** Model initialization

**Solution:** Wait 30-60 seconds on first run. Subsequent runs are faster.

#### High Memory Usage

**Cause:** T5 model loaded in memory

**Solution:** Use rule-based mode for lower memory:

```python
simplifier = PromptSimplifier(use_ml_model=False)
```

#### Database Errors

**Cause:** SQLite file permissions

**Solution:** Ensure write access to `data/logs/` directory.

### Error Messages

| Message                  | Meaning                 | Action                                 |
| ------------------------ | ----------------------- | -------------------------------------- |
| "Please enter a prompt"  | Input field empty       | Type a prompt                          |
| "ML model not available" | T5 model failed to load | Uses rule-based fallback automatically |
| "Logging failed"         | Database write error    | Check file permissions                 |

---

## FAQ

### General Questions

**Q: Is my data stored?**

A: By default, only a hash of your prompt is stored (not the full text). This protects privacy while allowing usage statistics.

**Q: How accurate are the energy predictions?**

A: The predictions are estimates based on synthetic training data. They provide relative comparisons and trends, not absolute values.

**Q: Can I use this for commercial purposes?**

A: This is a proof-of-concept academic project. Contact the team for commercial licensing.

### Technical Questions

**Q: What models are used?**

A:

- Energy Prediction: LinearRegression or RandomForest
- Prompt Optimization: T5-small (Hugging Face Transformers)
- Similarity: Sentence-Transformers (all-MiniLM-L6-v2)

**Q: Can I train on my own data?**

A: Yes! Use the training script:

```bash
cd src/nlp
python train_optimizer.py --epochs 10
```

**Q: What browsers are supported?**

A: Any modern browser. Tested on Chrome, Firefox, and Edge.

### Optimization Questions

**Q: Why is my prompt not being optimized?**

A: Short, efficient prompts may already be optimal. Check:

- Token count < 10
- No filler words present
- Direct command structure

**Q: The optimized prompt changed my meaning. What do I do?**

A: Check the Semantic Similarity score:

- Below 60%: Consider using original prompt
- 60-80%: Review and manually adjust
- Above 80%: Meaning well preserved

**Q: Can I customize the optimization rules?**

A: Yes, modify `src/nlp/simplifier.py` replacements dictionary.

---

## Support

### Contact

For questions or issues, contact the project team:

- Jarius Bedward - 8841640
- Mostafa Allahmoradi - 9087818
- Oluwafemi Lawal - 8967308
- Jatinder Pal Singh - 9083762

### Reporting Bugs

1. Check the troubleshooting section
2. Search existing issues
3. Submit detailed bug report with:
   - Steps to reproduce
   - Error messages
   - System information

---

## Quick Reference Card

### Keyboard Shortcuts

| Action        | Shortcut             |
| ------------- | -------------------- |
| Submit prompt | Enter (in text area) |
| Refresh page  | F5 or Ctrl+R         |

### Metric Thresholds

| Metric              | Poor | Good   | Excellent |
| ------------------- | ---- | ------ | --------- |
| Token Reduction     | <10% | 10-30% | >30%      |
| Semantic Similarity | <60% | 60-80% | >80%      |
| Quality Score       | <40  | 40-70  | >70       |
| Energy Saved        | <5%  | 5-20%  | >20%      |

### Model Defaults

```
Layers: 12
Training Time: 5.0 hours
FLOPs: 1.5e18
Model Type: LinearRegression
```
