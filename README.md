# Sustainable AI - Transparency and Energy-Efficient Prompt/Context Engineering with Machine Learning (CSCN8010)

## Project Members:

    1. Jarius Bedward - 8841640
    2. Mostafa Allahmoradi - 9087818
    3. Oluwafemi Lawal - 8967308
    4. Jatinder Pal Singh: - 9083762

## Project Summary:

This project addresses the critical environmental impact of the rapid expansion of AI data centers. With EU regulations requiring energy usage reporting by August 2026 and over 1,240 new data centers built in 2025 alone, there is an urgent need for transparency. This project builds a Machine Learning framework to predict the energy consumption of Large Language Model (LLM) prompts and recommends semantically equivalent, energy-efficient alternatives. It combines supervised learning for energy estimation and unsupervised learning for anomaly detection to optimize dynamic workloads.

## Project Setup:

## **Key Features:**

- **User Interface (GUI):**
  - Built with Streamlit/Flask to accept user prompts and model parameters (Layers, Training Time, FLOPs).
  - Visualizes energy costs and recommended improvements side-by-side.
- **NLP Module:**
  - Parses input prompts to extract token counts and linguistic complexity scores.
  - Uses sentence embeddings (Sentence-Transformers/OpenAI) to understand semantic context.
- **Energy Prediction Engine:**
  - A Supervised Learning model (Random Forest/Neural Network) predicts energy consumption (kWh).
  - Trains on features like prompt length, model depth, and FLOPs/hour.
- **Anomaly Detection:**
  - An Unsupervised Learning module (Isolation Forest/One-Class SVM) flags prompts with unusually high resource demands.
  - Identifies outliers in usage patterns for transparency.
- **Prompt Optimization:**
  - A recommendation engine suggests alternative prompts that yield similar outputs but require less computational power.
  - Uses fine-tuned models (T5/GPT-2) for paraphrasing and simplification.

## Requirements:

    - pip install -r requirements.txt

## 🤖 T5 Model Setup

The prompt optimizer uses a **fine-tuned T5-small transformer model** for intelligent optimization. The system includes both a fine-tuned model and rule-based fallback.

### Option 1: Use Pre-trained Model (Recommended)

If the fine-tuned model is included in the distribution (`model/prompt_optimizer/t5_finetuned/`), it will load automatically. No additional setup needed!

### Option 2: Train Your Own Model

If you need to train the model (or retrain with custom data):

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Install T5 training dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentencepiece
pip install tf-keras accelerate

# 3. Run the training script (takes ~25 minutes on CPU)
python scripts/train_t5_model.py --epochs 10 --batch-size 4

# The model will be saved to: model/prompt_optimizer/t5_finetuned/
```

**Training Options:**
- `--epochs N` - Number of training epochs (default: 10)
- `--batch-size N` - Batch size (default: 8, use 4 for low memory)
- `--learning-rate N` - Learning rate (default: 0.0003)

**Training Data:**
The training data is in `model/prompt_optimizer/training_data.json`. You can add more examples following the format:
```json
{
  "original": "Could you please help me...",
  "optimized": "Help me..."
}
```

### Option 3: Install Base T5 (No Fine-tuning)

If you just want to use the base T5 model without fine-tuning:

```bash
# For CPU-only:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers sentencepiece

# For GPU support (if you have CUDA):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers sentencepiece
```

### Verify T5 Installation:

```bash
python -c "from transformers import T5Tokenizer, T5ForConditionalGeneration; print('T5 ready!')"
```

### Distributing to Others

To share the trained model with teammates:
1. Train the model using `python scripts/train_t5_model.py`
2. Include the `model/prompt_optimizer/t5_finetuned/` directory when sharing
3. The model will load automatically on their machines

**Note:** The application works without T5, using rule-based optimization. The sidebar shows the current mode (T5 Fine-tuned, T5 Base, or Rule-Based).

## 🎯 How to Run:

1. Clone this repo (git clone <repo-url> cd <repo-folder>)
2. Install Required Dependencies: "pip install -r requirements.txt"
3. Navigate to the source directory: `cd src/gui`
4. Run the application: `streamlit run app.py` (or `python app.py`)
5. Input a prompt

## Code Explanation/Workflow:

1. **User Input & Configuration**

   - The user submits a text prompt via the GUI.
   - User provides LLM architecture details: Number of Layers, Known Training Time, and Expected FLOPs/hour.

2. **NLP Preprocessing**

   - The system calculates the token count and complexity score of the input text.
   - Vector embeddings are generated to capture the semantic meaning for the optimization engine.

3. **Energy Prediction (Supervised)**

   - The extracted features are passed to the Energy Prediction Model.
   - The model estimates the specific energy cost (kWh) for processing that prompt.

4. **Anomaly Detection (Unsupervised)**

   - The input metrics are cross-referenced with normal usage patterns using the Anomaly Detection Module.
   - Outliers are flagged (e.g., if a prompt requires excessive computation relative to its length).

5. **Optimization & Recommendation**

   - The Prompt Optimizer searches for or generates a more efficient version of the prompt.
   - It targets a lower token count or complexity while maintaining the original intent.

6. **Output & Logging**
   - The estimated energy and the optimized prompt are displayed to the user.
   - Data is

### Final Conclusion:

    - This project demonstrates a proof-of-concept for "Sustainable AI" by linking NLP inputs directly to physical energy estimates.
    - The application successfully gets the energy usage from a prompt, predicts how much energy it will take and comapre to the actual and then will provide the user with teh option for an optimzied version of the prompt to promote sustainbility.
        

### 🤝 Contributing

This is a Final Project Protocol developed for CSCN8010. If any questions arise do not hesitate to contact the project member.

### References:
