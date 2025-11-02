"""
Evaluation module for model performance assessment
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_recall_fscore_support,
    precision_score, recall_score, f1_score,
    precision_recall_curve, auc
)
from typing import Dict, Tuple


class ModelEvaluator:
    """Class untuk evaluasi performa model"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str = "Model") -> Dict:
        """
        Evaluate model performance
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Nama model untuk display
            
        Returns:
            Dictionary berisi metrics
        """
        print(f"\n--- {model_name} Evaluation ---")
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        print(f"  TN: {cm[0, 0]:,}  FP: {cm[0, 1]:,}")
        print(f"  FN: {cm[1, 0]:,}  TP: {cm[1, 1]:,}")
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Normal', 'Fraud'], 
                                   digits=4))
        
        # Metrics
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"\nKey Metrics:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Store results
        self.results[model_name] = {
            'confusion_matrix': cm,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tn': cm[0, 0],
            'fp': cm[0, 1],
            'fn': cm[1, 0],
            'tp': cm[1, 1]
        }
        
        return self.results[model_name]
    
    def optimize_threshold(self, y_true: np.ndarray, scores: np.ndarray, 
                          f_beta: float = 2.0) -> Tuple[float, int, Dict]:
        """
        Find optimal threshold using F-beta score
        
        Args:
            y_true: True labels
            scores: Anomaly scores
            f_beta: Beta value for F-beta score (beta > 1 emphasizes recall)
            
        Returns:
            Tuple berisi (optimal_threshold, optimal_idx, metrics_at_threshold)
        """
        print(f"\n--- Threshold Optimization (F-beta={f_beta}) ---")
        
        # Compute precision-recall curve
        precision_curve, recall_curve, thresholds_curve = precision_recall_curve(y_true, scores)
        pr_auc = auc(recall_curve, precision_curve)
        
        print(f"Precision-Recall AUC: {pr_auc:.4f}")
        
        # F-beta score
        f_beta_scores = ((1 + f_beta**2) * precision_curve * recall_curve) / \
                        (f_beta**2 * precision_curve + recall_curve + 1e-10)
        
        optimal_idx = np.argmax(f_beta_scores)
        optimal_threshold = thresholds_curve[optimal_idx] if optimal_idx < len(thresholds_curve) else thresholds_curve[-1]
        
        print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
        print(f"At this threshold:")
        print(f"  Precision: {precision_curve[optimal_idx]:.4f}")
        print(f"  Recall:    {recall_curve[optimal_idx]:.4f}")
        print(f"  F{f_beta}-Score: {f_beta_scores[optimal_idx]:.4f}")
        
        metrics = {
            'threshold': optimal_threshold,
            'precision': precision_curve[optimal_idx],
            'recall': recall_curve[optimal_idx],
            'f_beta_score': f_beta_scores[optimal_idx],
            'pr_auc': pr_auc,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'thresholds_curve': thresholds_curve
        }
        
        return optimal_threshold, optimal_idx, metrics
    
    def compare_models(self, models_results: Dict = None) -> pd.DataFrame:
        """
        Compare multiple models performance
        
        Args:
            models_results: Dictionary dengan hasil evaluasi multiple models
                          If None, menggunakan self.results
            
        Returns:
            DataFrame berisi comparison table
        """
        if models_results is None:
            models_results = self.results
        
        print("\n--- Model Comparison ---")
        
        comparison_data = []
        for model_name, metrics in models_results.items():
            comparison_data.append({
                'Model': model_name,
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1-Score': metrics['f1_score'],
                'TP': metrics.get('tp', 0),
                'FP': metrics.get('fp', 0),
                'TN': metrics.get('tn', 0),
                'FN': metrics.get('fn', 0)
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\n" + df_comparison.to_string(index=False))
        
        return df_comparison
    
    def analyze_errors(self, df: pd.DataFrame, y_pred: np.ndarray) -> Dict:
        """
        Analyze false positives and false negatives
        
        Args:
            df: Original DataFrame dengan Class column
            y_pred: Predicted labels
            
        Returns:
            Dictionary berisi error analysis
        """
        print("\n--- Error Analysis ---")
        
        # Add predictions to dataframe
        df_temp = df.copy()
        df_temp['Predicted'] = y_pred
        
        # False Positives (Normal predicted as Fraud)
        false_positives = df_temp[(df_temp['Class'] == 0) & (df_temp['Predicted'] == 1)]
        
        # False Negatives (Fraud predicted as Normal)
        false_negatives = df_temp[(df_temp['Class'] == 1) & (df_temp['Predicted'] == 0)]
        
        print(f"\nFalse Positives: {len(false_positives):,}")
        print(f"False Negatives: {len(false_negatives):,}")
        
        if len(false_negatives) > 0:
            print("\nSample of Missed Frauds (False Negatives):")
            print(false_negatives[['Time', 'Amount']].head(10))
        
        if len(false_positives) > 0:
            print("\nSample of False Alarms (False Positives):")
            print(false_positives[['Time', 'Amount']].head(10))
        
        return {
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'fp_count': len(false_positives),
            'fn_count': len(false_negatives)
        }
    
    def get_summary(self) -> str:
        """
        Get summary of all evaluations
        
        Returns:
            String berisi summary
        """
        summary = "\n" + "=" * 80 + "\n"
        summary += "EVALUATION SUMMARY\n"
        summary += "=" * 80 + "\n\n"
        
        for model_name, metrics in self.results.items():
            summary += f"{model_name}:\n"
            summary += f"  Precision: {metrics['precision']:.4f}\n"
            summary += f"  Recall:    {metrics['recall']:.4f}\n"
            summary += f"  F1-Score:  {metrics['f1_score']:.4f}\n"
            summary += f"  TP: {metrics['tp']:,}  FP: {metrics['fp']:,}\n"
            summary += f"  TN: {metrics['tn']:,}  FN: {metrics['fn']:,}\n\n"
        
        return summary
