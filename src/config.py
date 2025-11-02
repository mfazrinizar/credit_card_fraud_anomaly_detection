"""
Configuration file for project settings
"""

# Model Parameters
MODEL_CONFIG = {
    'baseline': {
        'n_estimators': 100,
        'contamination': 'auto',  # Will be set from data
        'random_state': 42,
        'n_jobs': -1
    },
    'tuned': {
        'n_estimators': 150,
        'random_state': 42,
        'n_jobs': -1
    },
    'ensemble': {
        'weights': {
            'isolation_forest': 0.4,
            'one_class_svm': 0.2,
            'local_outlier_factor': 0.2,
            'elliptic_envelope': 0.2
        }
    }
}

# Grid Search Parameters
GRID_SEARCH_PARAMS = {
    'contamination': [0.003, 0.005, 0.008, 0.01, 0.015],
    'max_samples': [256, 512, 'auto'],
    'max_features': [0.5, 0.75, 1.0]
}

# Feature Selection
FEATURE_SELECTION = {
    'n_top_features': 15,
    'correlation_threshold': 0.1
}

# Threshold Optimization
THRESHOLD_CONFIG = {
    'f_beta': 2.0,  # Emphasize recall
    'target_recall': 0.50,
    'target_precision': 0.30
}

# Visualization
VIZ_CONFIG = {
    'style': 'seaborn-v0_8-darkgrid',
    'dpi': 300,
    'figsize_eda': (16, 12),
    'figsize_results': (16, 6),
    'figsize_comparison': (16, 12)
}

# Data
DATA_CONFIG = {
    'kaggle_dataset': 'mlg-ulb/creditcardfraud',
    'file_name': 'creditcard.csv',
    'target_column': 'Class',
    'amount_bins': [0, 50, 100, 500, 1000, float('inf')],
    'amount_labels': ['0-50', '50-100', '100-500', '500-1k', '1k+']
}

# Paths
OUTPUT_DIR = 'output'
PLOTS = {
    'eda': 'output/01_eda.png',
    'detection': 'output/02_detection_results.png',
    'comparison': 'output/03_model_comparison.png'
}
RESULTS_CSV = 'output/model_comparison.csv'
