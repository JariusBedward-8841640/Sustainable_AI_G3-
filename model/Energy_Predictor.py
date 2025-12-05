import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sklearn.preprocessing as pp
import joblib
from typing import List, Tuple, Dict, Union

class EnergyModelTrainer:
    def __init__(self, model_type: str = "RandomForest", test_size: float = 0.2, anomaly_contamination: float = 0.05):
        """
        Args:
            model_type: "RandomForest" or "LinearRegression".
            test_size: Percentage of data to hold back for testing.
            anomaly_contamination: The expected percentage of outliers in the dataset (e.g., 0.05 = 5%).
        """
        self.model_type = model_type
        self.test_size = test_size
        
        # --- 1. Regressor Selection ---
        if self.model_type == "RandomForest":
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == "LinearRegression":
            self.model = LinearRegression()
        else:
            raise ValueError("model_type must be 'RandomForest' or 'LinearRegression'")
            
        # --- 2. Anomaly Detection Model (Isolation Forest) ---
        # This learns the "shape" of normal data to detect outliers.
        self.anomaly_model = IsolationForest(contamination=anomaly_contamination, random_state=42)
        
        # Transformers
        self.scaler_X = pp.MinMaxScaler()
        self.scaler_y = pp.MinMaxScaler()

        # State storage
        self.metrics = {}
        self.is_trained = False
        self.feature_names = []

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

        # 1. Train Energy Predictor
        self.model.fit(X_train, y_train.ravel())

        # 2. Train Anomaly Detector (Unsupervised - learns from X_train features only)
        # It learns what "Normal" inputs look like.
        self.anomaly_model.fit(X_train)
        
        self.is_trained = True

        # Evaluate Regressor
        self._evaluate(X_test, y_test)
        
        print(f"Training complete. R2 Score: {self.metrics['r2']:.4f}")
        print("Anomaly Detection Module: Active")

    def _evaluate(self, X_test_scaled, y_test_scaled):
        preds_scaled = self.model.predict(X_test_scaled)
        y_test_real = self.scaler_y.inverse_transform(y_test_scaled).flatten()
        preds_real = self.scaler_y.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()

        mse = mean_squared_error(y_test_real, preds_real)
        r2 = r2_score(y_test_real, preds_real)
        
        self.metrics = {"mse": mse, "r2": r2}
        self.y_test_real = y_test_real
        self.preds_real = preds_real

    def _prepare_input(self, input_data: dict) -> np.array:
        """Helper to safely format input dictionary to 2D array."""
        try:
            raw_values = [input_data[f] for f in self.feature_names]
        except KeyError as e:
            raise ValueError(f"Missing input feature: {e}")

        # Flatten accidental lists
        clean_values = []
        for val in raw_values:
            if hasattr(val, "__len__") and not isinstance(val, str):
                clean_values.append(val[0]) 
            else:
                clean_values.append(val)

        return np.array([clean_values])

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
        
        # IsolationForest returns -1 for Anomaly, 1 for Normal
        prediction = self.anomaly_model.predict(X_scaled)[0]
        
        if prediction == -1:
            return {"is_anomaly": True, "status": "ANOMALY DETECTED: Input pattern is unusual."}
        else:
            return {"is_anomaly": False, "status": "Normal Operation"}

    def visualize_performance(self):
        """Generates plots."""
        if not self.is_trained:
            raise ValueError("Model not trained.")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot 1: Actual vs Predicted
        ax1.scatter(self.y_test_real, self.preds_real, alpha=0.6, color='#2c3e50')
        min_val, max_val = min(self.y_test_real), max(self.y_test_real)
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Fit')
        ax1.set_xlabel('Actual Energy (kWh)')
        ax1.set_ylabel('Predicted Energy (kWh)')
        ax1.set_title(f'Accuracy: {self.model_type} (R2: {self.metrics["r2"]:.2f})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: Feature Importance or Coefficients
        if self.model_type == "RandomForest":
            importances = self.model.feature_importances_
            indices = np.argsort(importances)
            ax2.barh(range(len(indices)), importances[indices], color='#27ae60')
            ax2.set_yticks(range(len(indices)))
            ax2.set_yticklabels([self.feature_names[i] for i in indices])
            ax2.set_title('Random Forest: Feature Importance')
            
        elif self.model_type == "LinearRegression":
            coefs = self.model.coef_
            indices = np.argsort(coefs)
            ax2.barh(range(len(indices)), coefs[indices], color='#2980b9')
            ax2.set_yticks(range(len(indices)))
            ax2.set_yticklabels([self.feature_names[i] for i in indices])
            ax2.set_title('Linear Regression: Feature Coefficients')

        plt.tight_layout()
        return fig