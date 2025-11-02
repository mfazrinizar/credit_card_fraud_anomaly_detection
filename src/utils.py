"""
Utility functions for the project
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any


def create_output_directory(output_dir: str = "output") -> Path:
    """
    Create output directory if it doesn't exist
    
    Args:
        output_dir: Directory name
        
    Returns:
        Path object
    """
    path = Path(output_dir)
    path.mkdir(exist_ok=True)
    return path


def print_section_header(title: str, width: int = 80, char: str = "="):
    """
    Print formatted section header
    
    Args:
        title: Section title
        width: Width of header
        char: Character to use for border
    """
    print("\n" + char * width)
    print(title.center(width))
    print(char * width)


def print_subsection_header(title: str, width: int = 80, char: str = "-"):
    """
    Print formatted subsection header
    
    Args:
        title: Subsection title
        width: Width of header
        char: Character to use for border
    """
    print(f"\n{char * 3} {title} {char * (width - len(title) - 5)}")


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format value as percentage
    
    Args:
        value: Value to format (0-1 range)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_number(value: int) -> str:
    """
    Format number with comma separators
    
    Args:
        value: Number to format
        
    Returns:
        Formatted number string
    """
    return f"{value:,}"


def calculate_contamination_multiplier(base_rate: float, 
                                       target_multiplier: float = 2.0) -> float:
    """
    Calculate adjusted contamination rate
    
    Args:
        base_rate: Base contamination rate
        target_multiplier: Multiplier for adjustment
        
    Returns:
        Adjusted contamination rate
    """
    return base_rate * target_multiplier


def get_top_anomalies(df: pd.DataFrame, scores: np.ndarray, 
                      n_top: int = 10) -> pd.DataFrame:
    """
    Get top N anomalies based on scores
    
    Args:
        df: Original DataFrame
        scores: Anomaly scores
        n_top: Number of top anomalies to return
        
    Returns:
        DataFrame with top anomalies
    """
    df_temp = df.copy()
    df_temp['Anomaly_Score'] = scores
    
    # Sort by score (descending for most anomalous)
    top_anomalies = df_temp.nlargest(n_top, 'Anomaly_Score')
    
    return top_anomalies


def compute_detection_rate_by_amount(df: pd.DataFrame, 
                                     y_pred: np.ndarray,
                                     amount_bins: list,
                                     amount_labels: list) -> pd.Series:
    """
    Compute detection rate by amount range
    
    Args:
        df: DataFrame with 'Class' and 'Amount' columns
        y_pred: Predictions
        amount_bins: Bins for amount ranges
        amount_labels: Labels for bins
        
    Returns:
        Series with detection rates
    """
    df_temp = df.copy()
    df_temp['Amount_Bin'] = pd.cut(df_temp['Amount'], 
                                    bins=amount_bins, 
                                    labels=amount_labels)
    df_temp['Predicted'] = y_pred
    
    # Calculate detection rate
    fraud_by_bin = df_temp[df_temp['Class'] == 1].groupby('Amount_Bin', observed=True).size()
    detected_by_bin = df_temp[(df_temp['Class'] == 1) & (df_temp['Predicted'] == 1)].groupby('Amount_Bin', observed=True).size()
    detection_rate = (detected_by_bin / fraud_by_bin * 100).fillna(0)
    
    return detection_rate


def save_results_summary(results: Dict[str, Any], output_path: str):
    """
    Save results summary to text file
    
    Args:
        results: Dictionary with results
        output_path: Path to save file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("CREDIT CARD FRAUD DETECTION - RESULTS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"✓ Results summary saved to: {output_path}")


def get_model_summary_stats(metrics: Dict[str, float]) -> str:
    """
    Generate formatted summary statistics for a model
    
    Args:
        metrics: Dictionary with model metrics
        
    Returns:
        Formatted summary string
    """
    summary = []
    summary.append(f"Precision: {metrics['precision']:.4f}")
    summary.append(f"Recall:    {metrics['recall']:.4f}")
    summary.append(f"F1-Score:  {metrics['f1_score']:.4f}")
    
    if 'tp' in metrics:
        summary.append(f"TP: {metrics['tp']:,}  FP: {metrics['fp']:,}")
        summary.append(f"TN: {metrics['tn']:,}  FN: {metrics['fn']:,}")
    
    return "\n".join(summary)


def print_progress_bar(iteration: int, total: int, prefix: str = '', 
                       suffix: str = '', length: int = 50):
    """
    Print progress bar
    
    Args:
        iteration: Current iteration
        total: Total iterations
        prefix: Prefix string
        suffix: Suffix string
        length: Character length of bar
    """
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    
    if iteration == total:
        print()
