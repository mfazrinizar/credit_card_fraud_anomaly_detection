"""
Visualization module for Credit Card Fraud Detection
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple


class Visualizer:
    """Class untuk visualisasi data dan hasil"""
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer
        
        Args:
            style: Matplotlib style
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        sns.set_palette("husl")
    
    def plot_eda(self, df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Plot Exploratory Data Analysis
        
        Args:
            df: DataFrame with 'Time', 'Amount', and 'Class' columns
            save_path: Path to save figure (optional)
        """
        print("\n--- Generating EDA Plots ---")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Amount Distribution
        axes[0, 0].hist(df[df['Class'] == 0]['Amount'], bins=50, alpha=0.7, 
                       label='Normal', color='blue')
        axes[0, 0].hist(df[df['Class'] == 1]['Amount'], bins=50, alpha=0.7, 
                       label='Fraud', color='red')
        axes[0, 0].set_xlabel('Transaction Amount', fontsize=12)
        axes[0, 0].set_ylabel('Frequency', fontsize=12)
        axes[0, 0].set_title('Distribution: Amount', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].set_yscale('log')
        
        # Plot 2: Time Distribution
        axes[0, 1].hist(df[df['Class'] == 0]['Time'], bins=50, alpha=0.7, 
                       label='Normal', color='blue')
        axes[0, 1].hist(df[df['Class'] == 1]['Time'], bins=50, alpha=0.7, 
                       label='Fraud', color='red')
        axes[0, 1].set_xlabel('Time (seconds)', fontsize=12)
        axes[0, 1].set_ylabel('Frequency', fontsize=12)
        axes[0, 1].set_title('Distribution: Time', fontsize=14, fontweight='bold')
        axes[0, 1].legend()
        
        # Plot 3: Scatter Time vs Amount
        normal = df[df['Class'] == 0]
        fraud = df[df['Class'] == 1]
        
        axes[1, 0].scatter(normal['Time'], normal['Amount'], alpha=0.3, s=1, 
                          label='Normal', color='blue')
        axes[1, 0].scatter(fraud['Time'], fraud['Amount'], alpha=0.8, s=20, 
                          label='Fraud', color='red', marker='x')
        axes[1, 0].set_xlabel('Time (seconds)', fontsize=12)
        axes[1, 0].set_ylabel('Amount', fontsize=12)
        axes[1, 0].set_title('Scatter: Time vs Amount', fontsize=14, fontweight='bold')
        axes[1, 0].legend()
        
        # Plot 4: Class Distribution
        class_counts = df['Class'].value_counts()
        axes[1, 1].bar(['Normal', 'Fraud'], class_counts.values, 
                      color=['blue', 'red'], alpha=0.7, edgecolor='black')
        axes[1, 1].set_ylabel('Count', fontsize=12)
        axes[1, 1].set_title('Class Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].set_yscale('log')
        
        for i, v in enumerate(class_counts.values):
            axes[1, 1].text(i, v, f'{v:,}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ EDA plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_detection_results(self, df: pd.DataFrame, anomaly_scores: np.ndarray,
                              threshold: float, save_path: Optional[str] = None):
        """
        Plot anomaly detection results
        
        Args:
            df: DataFrame with predictions
            anomaly_scores: Anomaly scores
            threshold: Detection threshold
            save_path: Path to save figure (optional)
        """
        print("\n--- Generating Detection Results Plot ---")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Detected Anomalies
        df_temp = df.copy()
        df_temp['Anomaly_Score'] = anomaly_scores
        df_temp['Predicted'] = (anomaly_scores > threshold).astype(int)
        
        normal_pred = df_temp[df_temp['Predicted'] == 0]
        anomaly_pred = df_temp[df_temp['Predicted'] == 1]
        
        axes[0].scatter(normal_pred['Time'], normal_pred['Amount'], 
                       color='blue', label='Normal (Predicted)', alpha=0.3, s=1)
        axes[0].scatter(anomaly_pred['Time'], anomaly_pred['Amount'], 
                       color='red', label='Fraud (Predicted)', s=20, marker='X', alpha=0.8)
        axes[0].set_xlabel('Time (seconds)', fontsize=12)
        axes[0].set_ylabel('Amount', fontsize=12)
        axes[0].set_title('Detection Results: Time vs Amount', fontsize=14, fontweight='bold')
        axes[0].legend()
        
        # Plot 2: Score Distribution
        normal_true = df_temp[df_temp['Class'] == 0]
        fraud_true = df_temp[df_temp['Class'] == 1]
        
        axes[1].hist(normal_true['Anomaly_Score'], bins=100, alpha=0.6, 
                    label='Normal (True)', color='blue')
        axes[1].hist(fraud_true['Anomaly_Score'], bins=100, alpha=0.6, 
                    label='Fraud (True)', color='red')
        axes[1].axvline(threshold, color='green', linestyle='--', linewidth=2, 
                       label=f'Threshold={threshold:.3f}')
        axes[1].set_xlabel('Anomaly Score', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Anomaly Score Distribution', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Detection results plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame, 
                            pr_curves: Optional[dict] = None,
                            ensemble_scores: Optional[np.ndarray] = None,
                            y_true: Optional[np.ndarray] = None,
                            optimal_threshold: Optional[float] = None,
                            detection_by_amount: Optional[pd.Series] = None,
                            save_path: Optional[str] = None):
        """
        Plot comprehensive model comparison
        
        Args:
            comparison_df: DataFrame with model comparison
            pr_curves: Dict with PR curve data
            ensemble_scores: Ensemble anomaly scores
            y_true: True labels
            optimal_threshold: Optimal threshold value
            detection_by_amount: Detection rate by amount range
            save_path: Path to save figure (optional)
        """
        print("\n--- Generating Model Comparison Plot ---")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: F1-Score Comparison
        models = comparison_df['Model'].values
        f1_scores = comparison_df['F1-Score'].values
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][:len(models)]
        
        axes[0, 0].bar(range(len(models)), f1_scores, color=colors, 
                      edgecolor='black', linewidth=1.5)
        axes[0, 0].set_xticks(range(len(models)))
        axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
        axes[0, 0].set_ylabel('F1-Score', fontsize=12)
        axes[0, 0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylim([0, max(f1_scores) * 1.2])
        
        for i, v in enumerate(f1_scores):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Precision-Recall Curve
        if pr_curves:
            for model_name, curve_data in pr_curves.items():
                axes[0, 1].plot(curve_data['recall'], curve_data['precision'], 
                              linewidth=2, label=f"{model_name} (AUC={curve_data.get('auc', 0):.3f})")
            
            axes[0, 1].set_xlabel('Recall', fontsize=12)
            axes[0, 1].set_ylabel('Precision', fontsize=12)
            axes[0, 1].set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Ensemble Score Distribution
        if ensemble_scores is not None and y_true is not None:
            axes[1, 0].hist(ensemble_scores[y_true == 0], bins=100, alpha=0.6, 
                          label='Normal', color='blue')
            axes[1, 0].hist(ensemble_scores[y_true == 1], bins=100, alpha=0.6, 
                          label='Fraud', color='red')
            
            if optimal_threshold is not None:
                axes[1, 0].axvline(optimal_threshold, color='green', linestyle='--', 
                                 linewidth=2, label=f'Threshold={optimal_threshold:.3f}')
            
            axes[1, 0].set_xlabel('Ensemble Score', fontsize=12)
            axes[1, 0].set_ylabel('Frequency', fontsize=12)
            axes[1, 0].set_title('Ensemble Score Distribution', fontsize=14, fontweight='bold')
            axes[1, 0].legend()
            axes[1, 0].set_yscale('log')
        
        # Plot 4: Detection Rate by Amount
        if detection_by_amount is not None:
            x_pos = np.arange(len(detection_by_amount))
            axes[1, 1].bar(x_pos, detection_by_amount.values, color='#FF6B6B', 
                          edgecolor='black', linewidth=1.5)
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(detection_by_amount.index, rotation=45)
            axes[1, 1].set_ylabel('Detection Rate (%)', fontsize=12)
            axes[1, 1].set_xlabel('Transaction Amount Range', fontsize=12)
            axes[1, 1].set_title('Fraud Detection by Amount', fontsize=14, fontweight='bold')
            
            for i, v in enumerate(detection_by_amount.values):
                axes[1, 1].text(i, v + 2, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Model comparison plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
