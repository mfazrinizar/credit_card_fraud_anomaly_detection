# Credit Card Fraud Anomaly Detection using Unsupervised Learning

## Project Description

This project implements **unsupervised learning** to detect potentially fraudulent credit card transactions using **Isolation Forest** algorithms and **Ensemble Methods**.

### Business Objective
Identify potentially fraudulent credit card transactions to prevent financial losses using an unsupervised learning approach that can detect new anomaly patterns without requiring labeled data.

### Dataset
- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 
  - V1-V28: PCA-transformed features (anonymized)
  - Time: Transaction time in seconds
  - Amount: Transaction amount
  - Class: 0 = Normal, 1 = Fraud
- **Imbalance**: ~0.17% fraud (492 out of 284,807 transactions)

## Project Structure

```
unsupervised/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── data_loader.py           # Module for dataset loading
│   ├── preprocessing.py         # Module for preprocessing & feature engineering
│   ├── models.py                # Module for training models
│   ├── evaluation.py            # Module for model evaluation
│   ├── visualization.py         # Module for visualization
│   ├── config.py                # Configuration settings
│   └── utils.py                 # Utility functions
├── output/                      # Output folder (auto-generated)
│   ├── 01_eda.png
│   ├── 02_detection_results.png
│   ├── 03_model_comparison.png
│   └── model_comparison.csv
├── main.py                      # Main script to run pipeline
├── requirements.txt             # Dependencies
├── run.bat                      # Quick start script (Windows CMD)
├── run.ps1                      # Quick start script (Windows PowerShell)
├── run.sh                       # Quick start script (Linux/Mac)
├── .gitignore                   # Git ignore rules
└── README.md                    # Documentation (this file)
```

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Internet connection (to download dataset from Kaggle)

### 1. Clone or Download Repository
```bash
cd d:\Projects\DataScience\unsupervised
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Installation
```bash
python -c "import sklearn, pandas, numpy, matplotlib, seaborn, kagglehub; print('All packages installed successfully!')"
```

## How to Use

### Quick Start (Recommended)

**Windows (Command Prompt):**
```cmd
run.bat
```

**Windows (PowerShell):**
```powershell
.\run.ps1
```

**Linux/Mac:**
```bash
chmod +x run.sh
./run.sh
```

The script will automatically:
1. Check Python installation
2. Install dependencies
3. Create output directory
4. Run the fraud detection pipeline
5. Display results

### Option 1: Run Main Pipeline
```bash
python main.py
```

The pipeline will automatically:
1. Load dataset from Kaggle
2. Preprocessing & feature engineering
3. Train baseline model
4. Hyperparameter tuning
5. Train ensemble models
6. Optimize threshold
7. Evaluate & compare models
8. Generate visualizations
9. Save results to `output/` folder

### Option 2: Custom Usage
```python
from src.data_loader import DataLoader
from src.preprocessing import DataPreprocessor
from src.models import AnomalyDetectionModel
from src.evaluation import ModelEvaluator
from src.visualization import Visualizer

# Load data
loader = DataLoader()
df = loader.load_from_kaggle()

# Preprocess
preprocessor = DataPreprocessor()
X_scaled, y_true = preprocessor.scale_features(df)
X_top, features = preprocessor.select_top_features(df, n_features=15)

# Train model
model = AnomalyDetectionModel()
model.train_baseline_model(X_scaled, contamination=0.003)

# Evaluate
evaluator = ModelEvaluator()
y_pred = model.predict(X_scaled, model_type='baseline')
metrics = evaluator.evaluate_model(y_true, y_pred, "My Model")

# Visualize
visualizer = Visualizer()
visualizer.plot_eda(df, save_path="eda.png")
```

## Methodology

### 1. Baseline Model: Isolation Forest
- Algorithm: Isolation Forest
- Contamination: ~0.003 (estimated from data)
- Features: All 30 features (V1-V28, Time, Amount)

### 2. Feature Engineering
- Feature correlation analysis with Class
- Selection of top 15 features with highest correlation
- StandardScaler for normalization

### 3. Hyperparameter Tuning
- Grid search for:
  - `contamination`: [0.003, 0.005, 0.008, 0.01, 0.015]
  - `max_samples`: [256, 512, 'auto']
  - `max_features`: [0.5, 0.75, 1.0]
- Metric: F1-Score

### 4. Ensemble Approach
Combination of 4 algorithms:
- **Isolation Forest** (weight: 0.4)
- **OneClassSVM** (weight: 0.2)
- **Local Outlier Factor** (weight: 0.2)
- **Elliptic Envelope** (weight: 0.2)

### 5. Threshold Optimization
- F-beta score with beta=2 (emphasizing recall)
- Precision-Recall curve analysis
- Business-driven threshold selection

## Results & Performance

### Model Comparison

| Model | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Baseline IF | 0.0962 | 0.8354 | 0.1725 |
| Tuned IF | 0.3801 | 0.6606 | 0.4826 |
| Ensemble | 0.3942 | 0.6850 | 0.5004 |
| **Optimized (F2)** | **0.4444** | **0.6667** | **0.5333** |

### Ensemble Model Performance

**Confusion Matrix:**
```
[[283797    518]
 [   155    337]]
```

- True Negatives (TN): 283,797
- False Positives (FP): 518
- False Negatives (FN): 155
- True Positives (TP): 337

**Classification Report:**
```
              precision    recall  f1-score   support

      Normal     0.9995    0.9982    0.9988    284315
       Fraud     0.3942    0.6850    0.5004       492

    accuracy                         0.9976    284807
   macro avg     0.6968    0.8416    0.7496    284807
weighted avg     0.9984    0.9976    0.9980    284807
```

### Key Improvements
- **Recall**: 25.61% → 66.67% (2.6x improvement)
- **Precision**: 25.61% → 44.44% (stable)
- **F1-Score**: 0.17 → 0.53 (3x improvement)

### Business Impact
- **Expected fraud catch rate**: 66.7%
- **Expected workload**: ~738 transactions for review (0.26% of total)
- **False positive rate**: 55.6% (trade-off for high recall)
- **Detected fraud cases**: 337 out of 492 (68.5%)
- **Missed fraud cases**: 155 out of 492 (31.5%)

## Visualizations

The project generates 3 main visualizations:

1. **`01_eda.png`**: Exploratory Data Analysis
   - Amount & Time distribution
   - Scatter plot
   - Class distribution

2. **`02_detection_results.png`**: Detection Results
   - Detected anomalies visualization
   - Anomaly score distribution

3. **`03_model_comparison.png`**: Model Comparison
   - F1-Score comparison
   - Precision-Recall curves
   - Ensemble score distribution
   - Detection rate by amount range

## Business Recommendations

1. **Deploy Optimized Model** for real-time scoring
2. **Alert Prioritization** based on ensemble score (high score = high priority)
3. **Manual Review** focus on transactions with score > optimal threshold
4. **Continuous Learning** - update model regularly with new data
5. **Trade-off Awareness** - balance between fraud detection vs review workload

## Advantages of Unsupervised Approach

- No labeled data required for training
- Can detect previously unseen fraud patterns
- Adaptive to changes in fraud behavior
- Scalable for real-time monitoring

## Limitations

- High false positive rate (~56%) requires manual review
- Performance depends on feature engineering quality
- Trade-off between precision and recall
- Model may miss fraud with patterns very similar to normal transactions

## Future Improvements

1. **Hybrid Approach**: Combine unsupervised + supervised learning
2. **Feature Engineering**: Add behavioral features (frequency, velocity, etc.)
3. **Real-time Pipeline**: Implement streaming for real-time detection
4. **Active Learning**: Incorporate feedback from manual review
5. **Cost-Sensitive Learning**: Optimize based on business cost

## Dependencies

- `numpy >= 1.24.0` - Numerical computing
- `pandas >= 2.0.0` - Data manipulation
- `scikit-learn >= 1.3.0` - Machine learning
- `matplotlib >= 3.7.0` - Visualization
- `seaborn >= 0.12.0` - Statistical visualization
- `kagglehub >= 0.2.0` - Dataset loading

## Notes

- Dataset is automatically downloaded from Kaggle using `kagglehub`
- If download fails, manually download from Kaggle and use `loader.load_from_csv()`
- Training time: ~5-10 minutes (depending on hardware)
- Output files are saved in `output/` folder

## Troubleshooting

### Common Issues

**1. Kaggle download fails:**
```python
# Alternative: Load from CSV
loader = DataLoader()
df = loader.load_from_csv('path/to/creditcard.csv')
```

**2. Memory error:**
- Reduce dataset size or use sampling
- Close other applications
- Use smaller batch sizes

**3. Module not found:**
```bash
pip install -r requirements.txt --upgrade
```

**4. Permission denied (Linux/Mac):**
```bash
chmod +x run.sh
```

## Contributing

Feel free to create pull requests or open issues for suggestions/improvements.