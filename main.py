"""
Main script for Credit Card Fraud Detection using Unsupervised Learning
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.models import AnomalyDetectionModel
from src.evaluation import ModelEvaluator
from src.visualization import Visualizer

def main():
    """Main pipeline for fraud detection"""
    
    print("=" * 80)
    print("CREDIT CARD FRAUD DETECTION - UNSUPERVISED LEARNING")
    print("=" * 80)
    
    # Create output directory
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # 1. Data Loading
    # ========================================================================
    loader = DataLoader()
    
    try:
        # Try loading from Kaggle
        df = loader.load_from_kaggle()
    except Exception as e:
        print(f"Warning: Could not load from Kaggle ({e})")
        print("Please download the dataset manually and use load_from_csv()")
        return
    
    contamination_rate = loader.contamination_rate
    
    # ========================================================================
    # 2. Exploratory Data Analysis
    # ========================================================================
    visualizer = Visualizer()
    visualizer.plot_eda(df, save_path=str(output_dir / "01_eda.png"))
    
    # ========================================================================
    # 3. Preprocessing
    # ========================================================================
    preprocessor = DataPreprocessor()
    X_scaled, y_true = preprocessor.scale_features(df)
    X_top_scaled, top_features = preprocessor.select_top_features(df, n_features=15)
    
    # ========================================================================
    # 4. Model Training - Baseline
    # ========================================================================
    model = AnomalyDetectionModel()
    
    # Baseline model
    model.train_baseline_model(X_scaled, contamination=contamination_rate * 2)
    y_pred_baseline = model.predict(X_scaled, model_type='baseline')
    
    # ========================================================================
    # 5. Model Evaluation - Baseline
    # ========================================================================
    evaluator = ModelEvaluator()
    baseline_metrics = evaluator.evaluate_model(y_true, y_pred_baseline, "Baseline IF")
    
    # ========================================================================
    # 6. Advanced Model Training
    # ========================================================================
    print("\n" + "=" * 80)
    print("ADVANCED MODEL IMPROVEMENT")
    print("=" * 80)
    
    # Hyperparameter tuning
    best_params, y_pred_tuned = model.hyperparameter_tuning(X_top_scaled, y_true)
    tuned_metrics = evaluator.evaluate_model(y_true, y_pred_tuned, "Tuned IF")
    
    # Ensemble training
    model.train_ensemble_models(X_top_scaled, contamination=best_params['contamination'])
    ensemble_scores = model.compute_ensemble_scores(X_top_scaled)
    
    # ========================================================================
    # 7. Threshold Optimization
    # ========================================================================
    optimal_threshold, optimal_idx, opt_metrics = evaluator.optimize_threshold(
        y_true, ensemble_scores, f_beta=2.0
    )
    model.optimal_threshold = optimal_threshold
    
    # Predictions with different strategies
    y_pred_ensemble = model.predict(X_top_scaled, model_type='ensemble')
    ensemble_metrics = evaluator.evaluate_model(y_true, y_pred_ensemble, "Ensemble")
    
    # Optimized predictions
    y_pred_optimized = (ensemble_scores > optimal_threshold).astype(int)
    optimized_metrics = evaluator.evaluate_model(y_true, y_pred_optimized, "Optimized (F2)")
    
    # ========================================================================
    # 8. Model Comparison
    # ========================================================================
    comparison_df = evaluator.compare_models()
    
    # Calculate detection rate by amount
    amount_bins = [0, 50, 100, 500, 1000, np.inf]
    df['Amount_Bin'] = pd.cut(df['Amount'], bins=amount_bins, 
                              labels=['0-50', '50-100', '100-500', '500-1k', '1k+'])
    df['Predicted_Optimized'] = y_pred_optimized
    
    fraud_by_bin = df[df['Class'] == 1].groupby('Amount_Bin', observed=True).size()
    detected_by_bin = df[(df['Class'] == 1) & (df['Predicted_Optimized'] == 1)].groupby('Amount_Bin', observed=True).size()
    detection_rate = (detected_by_bin / fraud_by_bin * 100).fillna(0)
    
    # ========================================================================
    # 9. Visualization
    # ========================================================================
    # Detection results
    visualizer.plot_detection_results(
        df, ensemble_scores, optimal_threshold,
        save_path=str(output_dir / "02_detection_results.png")
    )
    
    # Model comparison
    pr_curves = {
        'Optimized': {
            'precision': opt_metrics['precision_curve'],
            'recall': opt_metrics['recall_curve'],
            'auc': opt_metrics['pr_auc']
        }
    }
    
    visualizer.plot_model_comparison(
        comparison_df,
        pr_curves=pr_curves,
        ensemble_scores=ensemble_scores,
        y_true=y_true,
        optimal_threshold=optimal_threshold,
        detection_by_amount=detection_rate,
        save_path=str(output_dir / "03_model_comparison.png")
    )
    
    # ========================================================================
    # 10. Error Analysis
    # ========================================================================
    error_analysis = evaluator.analyze_errors(df, y_pred_optimized)
    
    # ========================================================================
    # 11. Final Summary & Business Recommendations
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL SUMMARY & BUSINESS RECOMMENDATIONS")
    print("=" * 80)
    
    print(evaluator.get_summary())
    
    print("\nKey Insights:")
    print("1. Feature selection increases the model's focus on the most discriminative features")
    print("2. The ensemble approach provides more robust predictions")
    print("3. Threshold optimization balances the precision-recall trade-off")
    print(f"4. The optimized model detects {optimized_metrics['recall']*100:.1f}% of fraud "
          f"with precision {optimized_metrics['precision']*100:.1f}%")
    
    print("\nBusiness Recommendations:")
    print("1. Deploy the optimized model for real-time scoring")
    print("2. Set alert priority based on the ensemble score (high score = high priority)")
    print("3. Focus manual review on transactions with score > optimal threshold")
    print(f"4. Expected workload: {np.sum(y_pred_optimized):,} transactions for review "
          f"(~{np.sum(y_pred_optimized)/len(df)*100:.2f}% of total)")
    print(f"5. Expected fraud catch rate: {optimized_metrics['recall']*100:.1f}%")
    print(f"6. Trade-off: {(1-optimized_metrics['precision'])*100:.1f}% of alerts are false positives")
    # Save comparison table
    comparison_df.to_csv(output_dir / "model_comparison.csv", index=False)
    print(f"\nResults saved to: {output_dir}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    main()
