import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import sklearn.preprocessing as pp
from src.nlp.complexity_score import ComplexityAnalyzer
import os
import sys

import joblib  # For saving models

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class ModelTrainer:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler_X = pp.MinMaxScaler()
        self.scaler_y = pp.MinMaxScaler()

        self.X_scaled = None
        self.y_scaled = None
        self.y_original = None
        self.is_trained = False

    def preprocess(self, df, feature_cols, target_col):
        """
        Separates features and targets, and scales them.
        """
        X = df[feature_cols]
        y = df[target_col]

        # Store original y for plotting later
        self.y_original = y.values

        # Fit and transform
        self.X_scaled = self.scaler_X.fit_transform(X)
        self.y_scaled = self.scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()
        
        return self.X_scaled, self.y_scaled

    def train(self, df, feature_cols, target_col):
        """
        Orchestrates the training process.
        """
        print("Preprocessing data...")
        X, y = self.preprocess(df, feature_cols, target_col)

        print("Training model...")
        self.model.fit(X, y)
        self.is_trained = True
        print("Training complete.")

    def predict(self, X_new):
        """
        Predicts energy consumption for new data.
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Run .train() first.")
        
        X_new_scaled = self.scaler_X.transform(X_new)
        y_pred_scaled = self.model.predict(X_new_scaled)
        
        # Inverse transform to get actual kWh back (human readable)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        return y_pred

    def visualize_results(self):
        """
        Generates the matplotlib figure.
        Checks if model is trained first to prevent errors.
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet. Run .train() first.")
        
        # Get predictions for the training data to show fit
        # predictions = self.model.predict(self.X)

        # # Create Plot
        # fig, ax = plt.subplots(figsize=(8, 4))
        # ax.plot(self.data['num_layers'], self.y, 'o-', label='Actual Data', color='blue')
        # ax.plot(self.data['num_layers'], predictions, 'x--', label='Linear Reg. Prediction', color='orange')
        
        # ax.set_xlabel('Number of Layers')
        # ax.set_ylabel('Energy Consumption (kWh)')
        # ax.set_title('Energy Prediction Model Accuracy (Synthetic Data)')
        # ax.legend()
        # ax.grid(True, linestyle='--', alpha=0.7)

        # Get predictions on the training set
        predictions_scaled = self.model.predict(self.X_scaled)
        
        # Inverse transform to get actual kWh back (human readable)
        predictions_kwh = self.scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1)).flatten()

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plotting
        ax.scatter(range(len(self.y_original)), self.y_original, color='#1f77b4', label='Actual Energy', alpha=0.6)
        ax.scatter(range(len(predictions_kwh)), predictions_kwh, color='#ff7f0e', label='Predicted Energy', alpha=0.6, marker='x')

        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Energy Consumption (kWh)')
        ax.set_title('Actual vs Predicted Energy (Training Set)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

        return fig
    
    def save_model(self, filepath):
        if self.is_trained:
            joblib.dump(self.model, filepath)
            print(f"Model saved to {filepath}")