import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from src.nlp.complexity_score import ComplexityAnalyzer

class EnergyEstimator:
    def __init__(self):
        # --- MOCK DATA & TRAINING ---
        self.data = pd.DataFrame({
            'num_layers': [2, 4, 6, 8, 10, 12, 100],
            'training_hours': [1, 2, 3, 4, 5, 24, 48],
            'flops_per_hour': [10, 20, 40, 60, 90, 100, 200], 
            'energy_kwh': [0.5, 1.3, 2.8, 4.5, 6.9, 35.5, 80.2]
        })

        self.X = self.data[['num_layers', 'training_hours', 'flops_per_hour']]
        self.y = self.data['energy_kwh']

        self.model = LinearRegression()
        self.model.fit(self.X, self.y)

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

        predicted_energy = self.model.predict(input_data)[0]
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
        This matches the mockup code requirement.
        """
        # Get predictions for the training data to show fit
        predictions = self.model.predict(self.X)

        # Create Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(self.data['num_layers'], self.y, 'o-', label='Actual Data', color='blue')
        ax.plot(self.data['num_layers'], predictions, 'x--', label='Linear Reg. Prediction', color='orange')
        
        ax.set_xlabel('Number of Layers')
        ax.set_ylabel('Energy Consumption (kWh)')
        ax.set_title('Energy Prediction Model Accuracy (Mock Data)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        return fig