
# NBA Player Career Longevity Prediction Model
## Technical Documentation

### Model Overview
This machine learning model predicts whether an NBA player will have a long career (≥5 seasons) or a short career (<5 seasons) based on early-career statistics and other indicators. The model uses a stacked ensemble approach combining Random Forest, Gradient Boosting, and Logistic Regression with L1 regularization.

### Performance Metrics
- **F1 Score:** 0.8909
- **AUC-ROC:** 0.8322
- **Accuracy:** 0.8261
- **Precision at 75% Recall:** 0.9268

### Key Predictors
The top 5 most important features for predicting career longevity are:
Feature  Importance
    TRB    0.077415
    BPM    0.073852
    DRB    0.064652
     WS    0.064307
    DWS    0.037681

### Model Configuration
The optimized ensemble model uses the following configuration:

#### Random Forest:
{
    "max_depth": 14,
    "min_samples_leaf": 8,
    "min_samples_split": 16,
    "n_estimators": 134
}

#### Gradient Boosting:
{
    "learning_rate": 0.05958008171890075,
    "max_depth": 2,
    "min_samples_split": 8,
    "n_estimators": 58,
    "subsample": 0.908897907718663
}

#### Logistic Regression:
{
    "C": 1.8737005942368123
}

### Uncertainty Analysis
Bootstrap resampling (200 iterations) yielded the following uncertainty estimates:
- **Accuracy:** 0.8249 (95% CI: 0.7391-0.9130)
- **F1 Score:** 0.8894 (95% CI: 0.8302-0.9424)
- **AUC:** 0.8299 (95% CI: 0.7274-0.9372)

### Usage Guidelines
This model should be used as a decision support tool, not as the sole determinant for contract or draft decisions. Best practices include:
1. Consider the model's predicted probability alongside traditional scouting
2. Pay attention to feature importance to understand what's driving a specific prediction
3. Recognize that the model is trained on historical data and may not capture emerging trends
4. Re-evaluate predictions as more data becomes available for a player
    