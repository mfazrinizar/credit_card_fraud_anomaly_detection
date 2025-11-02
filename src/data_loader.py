"""
Data loader module for Credit Card Fraud Detection
"""

import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter


class DataLoader:
    """Class untuk load dan manage dataset"""
    
    def __init__(self):
        self.df = None
        self.fraud_count = 0
        self.normal_count = 0
        self.contamination_rate = 0.0
    
    def load_from_kaggle(self, dataset_name: str = "mlg-ulb/creditcardfraud", 
                         file_path: str = "creditcard.csv") -> pd.DataFrame:
        """
        Load dataset dari Kaggle menggunakan kagglehub
        
        Args:
            dataset_name: Nama dataset di Kaggle
            file_path: Nama file CSV dalam dataset
            
        Returns:
            DataFrame berisi data credit card
        """
        print("--- 1. Data Collection: Loading Dataset from Kaggle ---")
        print("Downloading dataset...")
        
        self.df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            dataset_name,
            file_path,
        )
        
        # Hitung statistik dataset
        self.fraud_count = self.df['Class'].sum()
        self.normal_count = len(self.df) - self.fraud_count
        self.contamination_rate = self.fraud_count / len(self.df)
        
        print(f"\nDataset loaded successfully!")
        print(f"  Shape: {self.df.shape}")
        print(f"  Normal transactions: {self.normal_count:,}")
        print(f"  Fraud transactions: {self.fraud_count:,}")
        print(f"  Contamination rate: {self.contamination_rate:.6f} ({self.contamination_rate * 100:.4f}%)")
        
        return self.df
    
    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load dataset dari file CSV lokal
        
        Args:
            file_path: Path ke file CSV
            
        Returns:
            DataFrame berisi data credit card
        """
        print(f"--- Loading dataset from: {file_path} ---")
        
        self.df = pd.read_csv(file_path)
        
        # Hitung statistik dataset
        self.fraud_count = self.df['Class'].sum()
        self.normal_count = len(self.df) - self.fraud_count
        self.contamination_rate = self.fraud_count / len(self.df)
        
        print(f"\nDataset loaded successfully!")
        print(f"  Shape: {self.df.shape}")
        print(f"  Normal transactions: {self.normal_count:,}")
        print(f"  Fraud transactions: {self.fraud_count:,}")
        print(f"  Contamination rate: {self.contamination_rate:.6f} ({self.contamination_rate * 100:.4f}%)")
        
        return self.df
    
    def get_data_info(self):
        """Print informasi detail tentang dataset"""
        if self.df is None:
            print("No data loaded yet!")
            return
        
        print("\n--- Data Understanding ---")
        print("\nFirst 5 rows:")
        print(self.df.head())
        
        print(f"\nDataset Info:")
        print(self.df.info())
        
        print(f"\nClass Distribution:")
        print(self.df['Class'].value_counts())
        
        print(f"\nBasic Statistics:")
        print(self.df.describe())
