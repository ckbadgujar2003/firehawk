import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib

class FireRiskPredictor:
    """XGBoost model for predicting forest fire risk levels"""
    
    def __init__(self, model_path=None):
        """Initialize the fire risk predictor
        
        Args:
            model_path (str, optional): Path to a saved model file. If None, a new model will be created.
        """
        self.model = None
        self.feature_columns = [
            'temperature', 'humidity', 'wind_speed', 'rainfall', 
            'vegetation_density', 'slope', 'elevation', 'distance_to_water',
            'days_since_last_fire', 'soil_moisture'
        ]
        
        # Load model if path is provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def preprocess_data(self, data):
        """Preprocess the data for training or prediction
        
        Args:
            data (pd.DataFrame): Raw data with feature columns
            
        Returns:
            pd.DataFrame: Preprocessed data
        """
        # Make a copy to avoid modifying the original data
        processed_data = data.copy()
        
        # Handle missing values
        for col in self.feature_columns:
            if col in processed_data.columns:
                processed_data[col].fillna(processed_data[col].mean(), inplace=True)
        
        # Normalize numerical features (optional, XGBoost can handle unnormalized data)
        # for col in self.feature_columns:
        #     if col in processed_data.columns:
        #         mean = processed_data[col].mean()
        #         std = processed_data[col].std()
        #         processed_data[col] = (processed_data[col] - mean) / (std if std > 0 else 1)
        
        return processed_data
    
    def train(self, data, test_size=0.2, random_state=42):
        """Train the XGBoost model with the provided data
        
        Args:
            data (pd.DataFrame): Training data with features and 'risk' column
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            dict: Training metrics including RMSE and R^2
        """
        # Preprocess data
        processed_data = self.preprocess_data(data)
        
        # Split features and target
        X = processed_data[self.feature_columns]
        y = processed_data['risk']
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # XGBoost parameters
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.01,
            'reg_lambda': 1,
            'random_state': random_state
        }
        
        # Train model
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(X_train, y_train, 
                     eval_set=[(X_test, y_test)],
                     eval_metric='rmse',
                     early_stopping_rounds=10,
                     verbose=False)
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        
        # Feature importance
        feature_importance = self.model.feature_importances_
        
        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'feature_importance': dict(zip(self.feature_columns, feature_importance))
        }
        
        return metrics
    
    def predict(self, features):
        """Predict fire risk using the trained model
        
        Args:
            features (pd.DataFrame): Features for prediction
            
        Returns:
            np.ndarray: Predicted risk values
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Train or load a model first.")
        
        # Preprocess features
        processed_features = self.preprocess_data(features)
        
        # Ensure all required features are present
        missing_features = [col for col in self.feature_columns if col not in processed_features.columns]
        if missing_features:
            raise ValueError(f"Missing features for prediction: {missing_features}")
        
        # Make predictions
        X = processed_features[self.feature_columns]
        predictions = self.model.predict(X)
        
        # Clip predictions to [0, 1] range for risk scores
        predictions = np.clip(predictions, 0, 1)
        
        return predictions
    
    def predict_grid(self, grid_data):
        """Predict fire risk for a grid environment
        
        Args:
            grid_data (pd.DataFrame): Grid data with features and coordinates
            
        Returns:
            np.ndarray: 2D grid of predicted risk values
        """
        # Get predictions for each cell
        predictions = self.predict(grid_data)
        
        # Extract grid dimensions
        if 'x' in grid_data.columns and 'y' in grid_data.columns:
            max_x = grid_data['x'].max() + 1
            max_y = grid_data['y'].max() + 1
            
            # Create empty grid
            risk_grid = np.zeros((max_y, max_x))
            
            # Fill grid with predictions
            for i, row in grid_data.iterrows():
                x, y = int(row['x']), int(row['y'])
                risk_grid[y, x] = predictions[i]
            
            return risk_grid
        else:
            return predictions
    
    def save_model(self, path):
        """Save the trained model to a file
        
        Args:
            path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        joblib.dump(self.model, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load a trained model from a file
        
        Args:
            path (str): Path to the saved model
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")
        
        self.model = joblib.load(path)
        print(f"Model loaded from {path}")
    
    def plot_feature_importance(self, save_path=None):
        """Plot feature importance from the trained model
        
        Args:
            save_path (str, optional): Path to save the plot. If None, the plot is displayed.
        """
        if self.model is None:
            raise ValueError("No trained model. Train a model first.")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['Feature'], importance_df['Importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importance for Fire Risk Prediction')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_predictions(self, grid_data, predictions=None, save_path=None):
        """Plot the predicted fire risk as a heatmap
        
        Args:
            grid_data (pd.DataFrame): Grid data with coordinates
            predictions (np.ndarray, optional): Predicted risk values. If None, they will be calculated.
            save_path (str, optional): Path to save the plot. If None, the plot is displayed.
        """
        if predictions is None:
            predictions = self.predict(grid_data)
        
        # Extract grid dimensions
        if 'x' in grid_data.columns and 'y' in grid_data.columns:
            max_x = grid_data['x'].max() + 1
            max_y = grid_data['y'].max() + 1
            
            # Create empty grid
            risk_grid = np.zeros((max_y, max_x))
            
            # Fill grid with predictions
            for i, row in grid_data.iterrows():
                x, y = int(row['x']), int(row['y'])
                if i < len(predictions):
                    risk_grid[y, x] = predictions[i]
            
            # Plot heatmap
            plt.figure(figsize=(10, 8))
            plt.imshow(risk_grid, cmap='YlOrRd', interpolation='nearest')
            plt.colorbar(label='Fire Risk')
            plt.title('Predicted Fire Risk')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.grid(True, which='both', color='black', linewidth=0.5, alpha=0.2)
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            else:
                plt.show()
            
            return risk_grid
        else:
            raise ValueError("Grid data must contain 'x' and 'y' columns for plotting.")