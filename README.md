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
    - The application successfully...
    - By integrating anomaly detection and prompt optimization...

### 🤝 Contributing

This is a Final Project Protocol developed for CSCN8010. If any questions arise do not hesitate to contact the project member.

### References:
