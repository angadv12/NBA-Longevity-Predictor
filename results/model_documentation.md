
# NBA Player Career Longevity Prediction Model
## Technical Documentation

### Model Overview
This machine learning model predicts whether an NBA player will have a long career (≥5 seasons) or a short career (<5 seasons) based on early-career statistics and other indicators. The model uses a stacked ensemble approach combining Random Forest, Gradient Boosting, and Logistic Regression with L1 regularization.

### Performance Metrics
- **F1 Score:** 0.8990
- **AUC-ROC:** 0.8320
- **Accuracy:** 0.8261
- **Precision at 75% Recall:** 0.8947

### Key Predictors
The top 5 most important features for predicting career longevity are:
Feature  Importance
    DRB    0.081275
    TRB    0.075708
    BPM    0.070101
     WS    0.066250
    DWS    0.046130

### Model Configuration
The optimized ensemble model uses the following configuration:

#### Random Forest:
{
    "max_depth": 4,
    "min_samples_leaf": 4,
    "min_samples_split": 16,
    "n_estimators": 554
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
    "C": 0.7810932022121826
}

### Uncertainty Analysis
Bootstrap resampling (200 iterations) yielded the following uncertainty estimates:
- **Accuracy:** 0.8419 (95% CI: 0.7652-0.9043)
- **F1 Score:** 0.9057 (95% CI: 0.8555-0.9447)
- **AUC:** 0.8510 (95% CI: 0.7614-0.9295)

### Usage Guidelines
This model should be used as a decision support tool, not as the sole determinant for contract or draft decisions. Best practices include:
1. Consider the model's predicted probability alongside traditional scouting
2. Pay attention to feature importance to understand what's driving a specific prediction
3. Recognize that the model is trained on historical data and may not capture emerging trends
4. Re-evaluate predictions as more data becomes available for a player
    