"""
Preprocessing module for Credit Card Fraud Detection
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class DataPreprocessor:
    """Class untuk preprocessing dan feature engineering"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.scaler_top = StandardScaler()
        self.X_scaled = None
        self.y_true = None
        self.top_features = None
        self.X_top_scaled = None
    
    def scale_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale semua fitur menggunakan StandardScaler
        
        Args:
            df: DataFrame input
            
        Returns:
            Tuple berisi (X_scaled, y_true)
        """
        print("\n--- 2. Data Preprocessing: Feature Scaling ---")
        
        # Pisahkan fitur dan label
        X = df.drop('Class', axis=1)
        self.y_true = df['Class'].values
        
        # Scaling semua fitur
        self.X_scaled = self.scaler.fit_transform(X)
        
        print(f"✓ Features scaled successfully!")
        print(f"  Shape: {self.X_scaled.shape}")
        
        return self.X_scaled, self.y_true
    
    def select_top_features(self, df: pd.DataFrame, n_features: int = 15) -> Tuple[np.ndarray, list]:
        """
        Pilih top N fitur berdasarkan korelasi dengan Class
        
        Args:
            df: DataFrame input
            n_features: Jumlah fitur yang ingin dipilih
            
        Returns:
            Tuple berisi (X_top_scaled, top_features)
        """
        print(f"\n--- Feature Selection: Top {n_features} Features ---")
        
        # Hitung korelasi dengan Class
        correlations = df.corrwith(df['Class']).abs().sort_values(ascending=False)
        
        # Exclude 'Class' dari features
        self.top_features = correlations.head(n_features + 1).index.tolist()
        if 'Class' in self.top_features:
            self.top_features.remove('Class')
        
        print(f"\nTop 10 features by correlation with Fraud:")
        print(correlations.head(10))
        
        print(f"\nSelected {len(self.top_features)} features:")
        print(self.top_features)
        
        # Scale top features
        X_top = df[self.top_features].copy()
        self.X_top_scaled = self.scaler_top.fit_transform(X_top)
        
        return self.X_top_scaled, self.top_features
    
    def get_scaled_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return DataFrame dengan fitur yang sudah di-scale
        
        Args:
            df: DataFrame original
            
        Returns:
            DataFrame dengan fitur scaled
        """
        if self.X_scaled is None:
            raise ValueError("Please call scale_features() first!")
        
        X_columns = [col for col in df.columns if col != 'Class']
        df_scaled = pd.DataFrame(self.X_scaled, columns=X_columns)
        df_scaled['Class'] = self.y_true
        
        return df_scaled
