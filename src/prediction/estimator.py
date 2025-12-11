import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from typing import List, Dict, Union

import sklearn.preprocessing as pp
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import os
import sys
import joblib

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.nlp.complexity_score import ComplexityAnalyzer

class EnergyEstimator:
    def __init__(self, model_type: str = "RandomForest", test_size: float = 0.2, anomaly_contamination: float = 0.05):

        self.test_size = test_size

        # Transformers
        self.scaler_X = pp.MinMaxScaler()
        self.scaler_y = pp.MinMaxScaler()

        # Model Selection
        if model_type == "RandomForest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == "LinearRegression":
            self.model = LinearRegression()
        else:
            raise ValueError("model_type must be 'RandomForest' or 'LinearRegression'")
        
        self.model_type = model_type
        
        # Join it with the filepath and the filename to get the absolute path
        energy_file_path = os.path.join(project_root, "data", "synthetic", 'energy_dataset.csv')
        # Read the data
        self.data = pd.read_csv(energy_file_path)

        # Define Features
        self.FEATURES = ['num_layers', 'training_hours', 'flops_per_hour', 'token_count', 'avg_word_length']
        self.TARGET = 'energy_kwh'

        self.anomaly_contamination = anomaly_contamination
        # Anomaly Detection Model
        self.anomaly_model = IsolationForest(contamination=anomaly_contamination, random_state=42)
        
        # State storage
        self.metrics = {}
        self.is_trained = False
        
        # Initialize prediction state variables
        self.latest_prediction = None

        # Define Model Directory and Dynamic Filename
        model_dir = os.path.join(project_root, "model", 'energy_predictor')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        # Filename now includes the model type
        self.model_path = os.path.join(model_dir, f'energy_model_{self.model_type}.pkl')
        
        # Smart Load/Train Logic
        if os.path.exists(self.model_path):
            self.load_model(self.model_path)
        else:
            print(f"No existing model found for {self.model_type}. Training new model...")
            self.train()
            self.save_model(self.model_path)

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
            'flops_per_hour': [flops_score],
            'token_count': [token_count],
            'avg_word_length': [ComplexityAnalyzer.get_avg_word_length(prompt_text)]
        })

        # Get raw prediction
        param = input_data.iloc[0].to_dict()
        print(f"--------------- Input Parameters for Prediction: {param} ---------------")
        predicted_energy = self.predict_energy(param)
        print(f"-----------------Predicted Energy (kWh): {predicted_energy}-------------------")
        predicted_energy = max(0.1, predicted_energy)
        
        # Save prediction to instance variables for plotting
        self.latest_prediction = predicted_energy

        suggestion = "✅ Optimized."
        if predicted_energy > 50:
            suggestion = "⚠️ High Consumption. Consider reducing layers."
        elif token_count > 1000:
            suggestion = "⚠️ Context too long. Summarize input."

        return {
            "energy_kwh": round(predicted_energy, 4),
            "carbon_kg": round(predicted_energy * 0.475, 4),
            "suggestion": suggestion,
            "token_count": token_count,
            "predicted_val": predicted_energy 
        }
    
    def preprocess_and_split(self):
        
        missing_cols = [c for c in self.FEATURES if c not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        X = self.data[self.FEATURES].values
        y = self.data[self.TARGET].values.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)

        # Scale Data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)

        X_test_scaled = self.scaler_X.transform(X_test)
        y_test_scaled = self.scaler_y.transform(y_test)

        return (X_train_scaled, y_train_scaled), (X_test_scaled, y_test_scaled)

    def train(self):
        print(f"--- Starting Training ({self.model_type}) ---")
        
        (X_train, y_train), (X_test, y_test) = self.preprocess_and_split()

        # Flatten y_train for sklearn
        # ravel is used to convert a multi-dimensional array into a one-dimensional (1D) array.
        self.model.fit(X_train, y_train.ravel())  
        self.anomaly_model.fit(X_train)
        
        self.is_trained = True
        self._evaluate(X_test, y_test)
        
        print(f"Training complete. R2 Score: {self.metrics['r2']:.4f}")

    def _evaluate(self, X_test_scaled, y_test_scaled):
        preds_scaled = self.model.predict(X_test_scaled)
        y_test_real = self.scaler_y.inverse_transform(y_test_scaled).flatten()
        preds_real = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        self.metrics = {
            "mse": mean_squared_error(y_test_real, preds_real),
            "r2": r2_score(y_test_real, preds_real)
        }

        self.y_test_real = y_test_real
        self.preds_real = preds_real

    def _prepare_input(self, input_data: dict) -> np.array:
        try:
            raw_values = [input_data[f] for f in self.FEATURES]
        except KeyError as e:
            raise ValueError(f"Input is missing feature: {e}")
        return np.array([raw_values])

    def predict_energy(self, input_data: dict) -> float:
        if not self.is_trained:
            raise ValueError("Model not trained (in predict_energy function).")
        if not input_data:
            raise ValueError("Input data is empty (in predict_energy function).")
        input_array = self._prepare_input(input_data)
        X_scaled = self.scaler_X.transform(input_array)
        pred_scaled = self.model.predict(X_scaled)
        
        return self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]

    def detect_anomaly(self, input_data: dict) -> Dict[str, Union[bool, str]]:
        if not self.is_trained:
            raise ValueError("Model not trained (in detect_anomaly function).")
        if not input_data:
            raise ValueError("Input data is empty (in detect_anomaly function).")
            
        input_array = self._prepare_input(input_data)
        X_scaled = self.scaler_X.transform(input_array)
        
        prediction = self.anomaly_model.predict(X_scaled)[0]
        return {
            "is_anomaly": True if prediction == -1 else False, 
            "status": "ANOMALY DETECTED" if prediction == -1 else "Normal Operation"
        }
    
    def save_model(self, file_path: str = None):
        """Saves the entire state."""
        if not self.is_trained:
            raise ValueError("Cannot save an untrained model.")
        
        # Use the instance path if none provided
        path_to_save = file_path if file_path else self.model_path

        artifact = {
            "model": self.model,
            "anomaly_model": self.anomaly_model,
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "FEATURES": self.FEATURES,
            "metrics": self.metrics,
            "model_type": self.model_type
        }
        joblib.dump(artifact, path_to_save)
        print(f"Model saved to {path_to_save}")

    def load_model(self, filename: str):
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No file found at {filename}")
            
        artifact = joblib.load(filename)
        
        self.model = artifact["model"]
        self.anomaly_model = artifact["anomaly_model"]
        self.scaler_X = artifact["scaler_X"]
        self.scaler_y = artifact["scaler_y"]
        self.FEATURES = artifact["FEATURES"]
        self.metrics = artifact["metrics"]
        self.model_type = artifact["model_type"]
        self.is_trained = True
        print(f"Model loaded from {filename}")    

    def get_training_plot(self, layers: int = None):
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        if layers is None:
            raise ValueError("Number of layers must be provided for plotting.")

        # Retrieve or reconstruct actual+predicted values
        if not hasattr(self, "y_test_real"):
            (X_scaled, y_scaled), _ = self.preprocess_and_split()
            preds_scaled = self.model.predict(X_scaled)

            y_real = self.scaler_y.inverse_transform(y_scaled).flatten()
            preds_real = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        else:
            y_real = self.y_test_real
            preds_real = self.preds_real

        # Slice for plotting
        num_points = round(layers * 1.5)
        limit = min(num_points, len(y_real))
        y_real = y_real[:limit]
        preds_real = preds_real[:limit]

        x = np.arange(1, len(y_real) + 1)

        # Create Plot
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(x, y_real, label="Actual", color="#1f77b4", marker="o")
        ax.plot(x, preds_real, label="Predicted", color="#ff7f0e",
                linestyle="--", marker="x")

        # --------------------------------------------------------
        #              HIGHLIGHT THE LATEST PREDICTION
        # --------------------------------------------------------
        if self.latest_prediction is not None:
            lx = layers                    # number of layers
            ly = self.latest_prediction    # predicted kWh

            # Convert "layers" to correct x-location.
            # If your x-axis is layers, use lx directly.
            # If your x-axis is sample index, then:
            x_pred = lx

            # Vertical line
            ax.vlines(
                x_pred, ymin=0, ymax=ly,
                colors="cyan", linewidth=2, zorder=5
            )

            # Horizontal line
            ax.hlines(
                ly, xmin=1, xmax=x_pred,
                colors="cyan", linewidth=2, zorder=5
            )

            # Marker (diamond)
            ax.scatter(
                x_pred, ly,
                s=120, marker="D",
                facecolor="cyan",
                edgecolor="black",
                linewidth=1,
                zorder=5
            )

            # Text bubble
            ax.annotate(
                "Prediction",
                xy=(x_pred, ly),
                xytext=(x_pred + 0.8, ly + 0.8),
                bbox=dict(boxstyle="round,pad=0.4", fc="cyan"),
                arrowprops=dict(arrowstyle="->"),
                zorder=11
            )

        # Axis settings
        ax.set_title(f"Energy Prediction vs Actual ({self.model_type})")
        ax.set_xlabel("Number of Layers")
        ax.set_ylabel("Energy Consumption (kWh)")
        ax.grid(True, alpha=0.4)
        ax.legend()

        return fig
