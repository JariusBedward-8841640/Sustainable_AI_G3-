import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, IsolationForest
import sklearn.preprocessing as pp
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from src.nlp.complexity_score import ComplexityAnalyzer
import matplotlib.pyplot as plt
from typing import List, Dict, Union
import os
import sys
import joblib

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

class EnergyEstimator:
    def __init__(self, model_type: str = "RandomForest", test_size: float = 0.2, anomaly_contamination: float = 0.05):

        self.model_type = model_type
        self.test_size = test_size
        self.anomaly_contamination = anomaly_contamination
        
        # Join it with the filepath and the filename to get the absolute path
        features_file_path = os.path.join(project_root, "data", "processed", 'features_df.csv')
        energy_file_path = os.path.join(project_root, "data", "synthetic", 'energy_dataset.csv')
        
        # Read the file
        self.data = pd.read_csv(energy_file_path)

        # Define Features
        FEATURES = ['num_layers', 'training_hours', 'flops_per_hour']
        TARGET = 'energy_kwh'

        # --- Model Selection ---
        if self.model_type == "RandomForest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == "LinearRegression":
            self.model = LinearRegression()
        else:
            raise ValueError("model_type must be 'RandomForest' or 'LinearRegression'")
            
        # --- Anomaly Detection Model (Isolation Forest) ---
        # This learns the "shape" of normal data to detect outliers.
        self.anomaly_model = IsolationForest(contamination=anomaly_contamination, random_state=42)
        
        # Transformers
        self.scaler_X = pp.MinMaxScaler()
        self.scaler_y = pp.MinMaxScaler()

        # State storage
        self.metrics = {}
        self.is_trained = False
        self.feature_names = []
            
        MODEL_FILE_PATH = os.path.join(project_root, "model", 'energy_predictor', 'energy_model_pkg.pkl')

        # Smart Load/Train Logic
        if os.path.exists(MODEL_FILE_PATH):
            self.load_model(MODEL_FILE_PATH)
        else:
            print("No existing model found. Training new model...")
            # Load your dataframe here
            self.train(self.data, FEATURES, TARGET)
            self.save_model()

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

        predicted_energy = self.predict_energy(input_data.iloc[0].to_dict())
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
    
    def preprocess_and_split(self, df: pd.DataFrame, feature_cols: List[str], target_col: str):
        self.feature_names = feature_cols
        
        missing_cols = [c for c in feature_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        X = df[feature_cols].values
        y = df[target_col].values.reshape(-1, 1)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )

        # Scale Data
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        y_train_scaled = self.scaler_y.fit_transform(y_train)

        X_test_scaled = self.scaler_X.transform(X_test)
        y_test_scaled = self.scaler_y.transform(y_test)

        return (X_train_scaled, y_train_scaled), (X_test_scaled, y_test_scaled)

    def train(self, df: pd.DataFrame, feature_cols: List[str], target_col: str):
        """
        Trains both the Energy Regressor and the Anomaly Detector.
        """
        print(f"--- Starting Training ({self.model_type} + IsolationForest) ---")
        
        (X_train, y_train), (X_test, y_test) = self.preprocess_and_split(df, feature_cols, target_col)

        # Train Energy Predictor
        self.model.fit(X_train, y_train.ravel())

        # Train Anomaly Detector (Unsupervised - learns from X_train features only). It learns what "Normal" inputs look like.
        self.anomaly_model.fit(X_train)
        
        self.is_trained = True

        # Evaluate Regressor
        self._evaluate(X_test, y_test) # This function stores test results in metrics dictionary
        
        print(f"Training complete. R2 Score: {self.metrics['r2']:.4f}")
        print("Anomaly Detection Module: Active")

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
        """Helper to safely format input dictionary to 2D array."""
        try:
            raw_values = [input_data[f] for f in self.feature_names]
        except KeyError as e:
            raise ValueError(f"Input is missing feature: {e}")

        return np.array([raw_values])

    def predict_energy(self, input_data: dict) -> float:
        """Predicts energy cost (kWh)."""
        if not self.is_trained:
            raise ValueError("Model not trained.")
        
        input_array = self._prepare_input(input_data)
        X_scaled = self.scaler_X.transform(input_array)
        pred_scaled = self.model.predict(X_scaled)
        
        return self.scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()[0]

    def detect_anomaly(self, input_data: dict) -> Dict[str, Union[bool, str]]:
        """
        Checks if the input data is an 'Anomaly' (outlier) compared to training data.
        Returns: Dictionary with status and description.
        """
        if not self.is_trained:
            raise ValueError("Model not trained.")
            
        input_array = self._prepare_input(input_data)
        X_scaled = self.scaler_X.transform(input_array)
        
        prediction = self.anomaly_model.predict(X_scaled)[0]
        
        return {
            "is_anomaly": True if prediction == -1 else False, 
            "status": "ANOMALY DETECTED" if prediction == -1 else "Normal Operation"
        }
    
    # --- Persistence Methods ---
    def save_model(self, file_path: str = "model/energy_predictor/energy_model_pkg.pkl"):
        """Saves the entire state including scalers and feature names."""
        if not self.is_trained:
            raise ValueError("Cannot save an untrained model.")
            
        artifact = {
            "model": self.model,
            "anomaly_model": self.anomaly_model,
            "scaler_X": self.scaler_X,
            "scaler_y": self.scaler_y,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "model_type": self.model_type
        }
        joblib.dump(artifact, file_path)
        print(f"Model saved to {file_path}")

    def load_model(self, filename: str = "energy_model_pkg.pkl"):
        """Loads the model and restores state."""
        if not os.path.exists(filename):
            raise FileNotFoundError(f"No file found at {filename}")
            
        artifact = joblib.load(filename)
        
        self.model = artifact["model"]
        self.anomaly_model = artifact["anomaly_model"]
        self.scaler_X = artifact["scaler_X"]
        self.scaler_y = artifact["scaler_y"]
        self.feature_names = artifact["feature_names"]
        self.metrics = artifact["metrics"]
        self.model_type = artifact["model_type"]
        self.is_trained = True
        print(f"Model loaded from {filename}")    

    def get_training_plot(self):
        """
        Generates plots. 
        If model was loaded from disk, you MUST provide 'df_new' and 'target_col' to see plots.
        """
        target_col='energy_kwh'
        df = self.data
        if not self.is_trained:
            raise ValueError("Model not trained.")

        # --- Scenario A: Model just loaded (No test data in memory) ---
        if not hasattr(self, 'y_test_real'):
            if df is not None and target_col is not None:
                print("Generating plots using provided new data...")
                # Generate predictions on the new data provided
                (X_new, y_new), _ = self.preprocess_and_split(df, self.feature_names, target_col)
                preds_scaled = self.model.predict(X_new)
                
                # Inverse transform to get real units (kWh)
                y_real = self.scaler_y.inverse_transform(y_new).flatten()
                preds_real = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
                
                # Calculate metrics for title
                r2 = r2_score(y_real, preds_real)
            else:
                print("⚠️ Cannot visualize: Model was loaded from disk and no new data was provided.")
                print("Fix: Call visualize_performance(df=my_dataframe, target_col='Energy')")
                return

        # --- Scenario B: Model just trained (Test data exists in memory) ---
        else:
            y_real = self.y_test_real
            preds_real = self.preds_real
            r2 = self.metrics['r2']

        # --- Plotting Logic (Same as before) ---
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Actual vs Predicted
        ax1.scatter(y_real, preds_real, alpha=0.6, color='#2c3e50')
        min_val, max_val = min(y_real), max(y_real)
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
        ax1.set_xlabel('Actual Energy (kWh)')
        ax1.set_ylabel('Predicted Energy (kWh)')
        ax1.set_title(f'Accuracy: {self.model_type} (R2: {r2:.2f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Feature Importance
        if self.model_type == "RandomForest":
            importances = self.model.feature_importances_
            indices = np.argsort(importances)
            ax2.barh(range(len(indices)), importances[indices], color='#27ae60')
            ax2.set_yticks(range(len(indices)))
            ax2.set_yticklabels([self.feature_names[i] for i in indices])
            ax2.set_title('Feature Importance')
            
        elif self.model_type == "LinearRegression":
            coefs = self.model.coef_
            indices = np.argsort(coefs)
            ax2.barh(range(len(indices)), coefs[indices], color='#2980b9')
            ax2.set_yticks(range(len(indices)))
            ax2.set_yticklabels([self.feature_names[i] for i in indices])
            ax2.set_title('Feature Coefficients')

        plt.tight_layout()
        return fig