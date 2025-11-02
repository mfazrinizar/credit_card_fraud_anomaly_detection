"""
Model training module for Credit Card Fraud Detection
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from typing import Dict, Tuple, List
import warnings


class AnomalyDetectionModel:
    """Class untuk training model anomaly detection"""
    
    def __init__(self):
        self.model_if = None
        self.best_model = None
        self.best_params = {}
        self.ensemble_models = {}
        self.ensemble_scores = None
        self.optimal_threshold = None
    
    def train_baseline_model(self, X: np.ndarray, contamination: float = 0.003) -> IsolationForest:
        """
        Train baseline Isolation Forest model
        
        Args:
            X: Feature matrix
            contamination: Expected proportion of anomalies
            
        Returns:
            Trained IsolationForest model
        """
        print("\n--- 4. Baseline Model: Isolation Forest ---")
        print(f"Training with contamination={contamination:.6f}")
        
        self.model_if = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42,
            n_jobs=-1
        )
        
        self.model_if.fit(X)
        print("✓ Baseline model trained successfully!")
        
        return self.model_if
    
    def hyperparameter_tuning(self, X: np.ndarray, y_true: np.ndarray) -> Tuple[Dict, np.ndarray]:
        """
        Perform grid search for optimal hyperparameters
        
        Args:
            X: Feature matrix
            y_true: True labels
            
        Returns:
            Tuple berisi (best_params, best_predictions)
        """
        print("\n--- Hyperparameter Tuning: Grid Search ---")
        
        param_grid = {
            'contamination': [0.003, 0.005, 0.008, 0.01, 0.015],
            'max_samples': [256, 512, 'auto'],
            'max_features': [0.5, 0.75, 1.0]
        }
        
        best_f1 = 0
        best_predictions = None
        results = []
        
        print("Testing parameter combinations...")
        total_combinations = len(param_grid['contamination']) * len(param_grid['max_samples']) * len(param_grid['max_features'])
        counter = 0
        
        for cont in param_grid['contamination']:
            for max_samp in param_grid['max_samples']:
                for max_feat in param_grid['max_features']:
                    counter += 1
                    
                    model = IsolationForest(
                        n_estimators=150,
                        contamination=cont,
                        max_samples=max_samp,
                        max_features=max_feat,
                        random_state=42,
                        n_jobs=-1
                    )
                    
                    model.fit(X)
                    y_pred = np.where(model.predict(X) == -1, 1, 0)
                    
                    precision = precision_score(y_true, y_pred, zero_division=0)
                    recall = recall_score(y_true, y_pred)
                    f1 = f1_score(y_true, y_pred)
                    
                    results.append({
                        'contamination': cont,
                        'max_samples': max_samp,
                        'max_features': max_feat,
                        'precision': precision,
                        'recall': recall,
                        'f1': f1
                    })
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        self.best_params = {
                            'contamination': cont,
                            'max_samples': max_samp,
                            'max_features': max_feat
                        }
                        best_predictions = y_pred
                        self.best_model = model
                    
                    if counter % 10 == 0:
                        print(f"  Progress: {counter}/{total_combinations} combinations tested...")
        
        print(f"\n✓ Grid search completed!")
        print(f"  Best F1-Score: {best_f1:.4f}")
        print(f"  Best Parameters:")
        print(f"    - Contamination: {self.best_params['contamination']}")
        print(f"    - Max Samples: {self.best_params['max_samples']}")
        print(f"    - Max Features: {self.best_params['max_features']}")
        
        return self.best_params, best_predictions
    
    def train_ensemble_models(self, X: np.ndarray, contamination: float = 0.003) -> Dict:
        """
        Train multiple anomaly detection models for ensemble
        
        Args:
            X: Feature matrix
            contamination: Expected proportion of anomalies
            
        Returns:
            Dictionary berisi trained models
        """
        print("\n--- Training Ensemble Models ---")
        
        # Model 1: Optimized Isolation Forest
        print("  [1/4] Training Isolation Forest...")
        model_if = IsolationForest(
            n_estimators=150,
            contamination=contamination,
            max_samples=self.best_params.get('max_samples', 'auto'),
            max_features=self.best_params.get('max_features', 1.0),
            random_state=42,
            n_jobs=-1
        )
        model_if.fit(X)
        
        # Model 2: OneClassSVM
        print("  [2/4] Training OneClassSVM...")
        model_svm = OneClassSVM(nu=0.01, kernel='rbf', gamma='auto')
        model_svm.fit(X)
        
        # Model 3: Local Outlier Factor
        print("  [3/4] Training Local Outlier Factor...")
        model_lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01, novelty=False)
        model_lof.fit_predict(X)
        
        # Model 4: Elliptic Envelope
        print("  [4/4] Training Elliptic Envelope...")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model_ee = EllipticEnvelope(contamination=0.01, random_state=42)
                model_ee.fit(X)
        except Exception as e:
            print(f"    Warning: EllipticEnvelope failed ({str(e)}), using IF as fallback")
            model_ee = model_if
        
        self.ensemble_models = {
            'isolation_forest': model_if,
            'one_class_svm': model_svm,
            'local_outlier_factor': model_lof,
            'elliptic_envelope': model_ee
        }
        
        print("✓ All ensemble models trained successfully!")
        
        return self.ensemble_models
    
    def compute_ensemble_scores(self, X: np.ndarray, weights: Dict[str, float] = None) -> np.ndarray:
        """
        Compute weighted ensemble scores
        
        Args:
            X: Feature matrix
            weights: Dictionary dengan weights untuk setiap model
            
        Returns:
            Ensemble scores array
        """
        if weights is None:
            weights = {
                'isolation_forest': 0.4,
                'one_class_svm': 0.2,
                'local_outlier_factor': 0.2,
                'elliptic_envelope': 0.2
            }
        
        print("\n--- Computing Ensemble Scores ---")
        print(f"Weights: {weights}")
        
        # Get scores from each model
        scores_if = -self.ensemble_models['isolation_forest'].decision_function(X)
        scores_svm = -self.ensemble_models['one_class_svm'].decision_function(X)
        scores_lof = -self.ensemble_models['local_outlier_factor'].negative_outlier_factor_
        
        try:
            scores_ee = -self.ensemble_models['elliptic_envelope'].decision_function(X)
        except:
            scores_ee = scores_if
        
        # Normalize scores to [0, 1]
        scaler = MinMaxScaler()
        scores_if_norm = scaler.fit_transform(scores_if.reshape(-1, 1)).flatten()
        scores_svm_norm = scaler.fit_transform(scores_svm.reshape(-1, 1)).flatten()
        scores_lof_norm = scaler.fit_transform(scores_lof.reshape(-1, 1)).flatten()
        scores_ee_norm = scaler.fit_transform(scores_ee.reshape(-1, 1)).flatten()
        
        # Weighted ensemble
        self.ensemble_scores = (
            weights['isolation_forest'] * scores_if_norm +
            weights['one_class_svm'] * scores_svm_norm +
            weights['local_outlier_factor'] * scores_lof_norm +
            weights['elliptic_envelope'] * scores_ee_norm
        )
        
        print("✓ Ensemble scores computed successfully!")
        
        return self.ensemble_scores
    
    def predict(self, X: np.ndarray, model_type: str = 'baseline') -> np.ndarray:
        """
        Make predictions using specified model
        
        Args:
            X: Feature matrix
            model_type: 'baseline', 'best', or 'ensemble'
            
        Returns:
            Predictions array (0=normal, 1=fraud)
        """
        if model_type == 'baseline':
            if self.model_if is None:
                raise ValueError("Baseline model not trained yet!")
            y_pred_raw = self.model_if.predict(X)
        elif model_type == 'best':
            if self.best_model is None:
                raise ValueError("Best model not found! Run hyperparameter_tuning first.")
            y_pred_raw = self.best_model.predict(X)
        elif model_type == 'ensemble':
            if self.ensemble_scores is None:
                raise ValueError("Ensemble scores not computed! Run compute_ensemble_scores first.")
            if self.optimal_threshold is None:
                # Use default threshold
                contamination = self.best_params.get('contamination', 0.003)
                threshold = np.percentile(self.ensemble_scores, (1 - contamination) * 100)
            else:
                threshold = self.optimal_threshold
            return (self.ensemble_scores > threshold).astype(int)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        return np.where(y_pred_raw == -1, 1, 0)
