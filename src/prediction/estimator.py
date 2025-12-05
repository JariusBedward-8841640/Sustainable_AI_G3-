import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn.preprocessing as pp
from src.nlp.complexity_score import ComplexityAnalyzer
import os
import sys

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from model.model_trainer import ModelTrainer

class EnergyEstimator:
    def __init__(self):
        # --- MOCK DATA & TRAINING ---
        # self.data = pd.DataFrame({
        #     'num_layers': [2, 4, 6, 8, 10, 12, 100],
        #     'training_hours': [1, 2, 3, 4, 5, 24, 48],
        #     'flops_per_hour': [10, 20, 40, 60, 90, 100, 200], 
        #     'energy_kwh': [0.5, 1.3, 2.8, 4.5, 6.9, 35.5, 80.2]
        # })

        self.trainer = ModelTrainer()
        
        # Join it with the filepath and the filename to get the absolute path
        features_file_path = os.path.join(project_root, "data", "processed", 'features_df.csv')
        energy_file_path = os.path.join(project_root, "data", "synthetic", 'energy_dataset.csv')
        
        print(f"Reading file from: {energy_file_path}") # Debug print
        
        # Read the file
        self.data = pd.read_csv(energy_file_path)

        # Define Features
        FEATURES = ['num_layers', 'training_hours', 'flops_per_hour']
        TARGET = 'energy_kwh'

        # Train
        self.trainer.train(self.data, FEATURES, TARGET)

    def estimate(self, prompt_text, layers, training_hours, flops_str):
        # 1. Parse Inputs
        token_count = ComplexityAnalyzer.get_token_count(prompt_text)
        
        try:
            flops_val = float(flops_str)
            flops_score = (flops_val / 1e18) * 100 
            if flops_score < 10: flops_score = 10
        except ValueError:
            flops_score = 10.0

        input_data = pd.DataFrame({
            'num_layers': [layers],
            'training_hours': [training_hours],
            'flops_per_hour': [flops_score]
        })

        predicted_energy = self.trainer.predict(input_data)[0]
        predicted_energy = max(0.1, predicted_energy)

        suggestion = "✅ Optimized."
        if predicted_energy > 50:
            suggestion = "⚠️ High Consumption. Consider reducing layers."
        elif token_count > 1000:
            suggestion = "⚠️ Context too long. Summarize input."

        return {
            "energy_kwh": round(predicted_energy, 4),
            "carbon_kg": round(predicted_energy * 0.475, 4),
            "suggestion": suggestion,
            "token_count": token_count
        }

    def get_training_plot(self):
        """
        Generates the matplotlib figure for Actual vs Predicted Energy.
        """

        fig = self.trainer.visualize_results()
        
        return fig