
# NBA Player Career Longevity Prediction Model
## Technical Documentation

### Model Overview
This machine learning model predicts whether an NBA player will have a long career (≥5 seasons) or a short career (<5 seasons) based on early-career statistics and other indicators. The model uses a stacked ensemble approach combining Random Forest, Gradient Boosting, and Logistic Regression with L1 regularization.

### Performance Metrics
- **F1 Score:** 0.9952
- **AUC-ROC:** 0.9995
- **Accuracy:** 0.9920
- **Precision at 75% Recall:** 1.0000

### Key Predictors
The top 5 most important features for predicting career longevity are:
  Feature  Importance
        G    0.208410
       MP    0.146776
TRB_Total    0.138385
       WS    0.117537
PTS_Total    0.083999

### Model Configuration
The optimized ensemble model uses the following configuration:

#### Random Forest:
{
    "max_depth": 16,
    "min_samples_leaf": 1,
    "min_samples_split": 5,
    "n_estimators": 661
}

#### Gradient Boosting:
{
    "learning_rate": 0.13958350559263474,
    "max_depth": 2,
    "min_samples_split": 11,
    "n_estimators": 237,
    "subsample": 0.7465447373174767
}

#### Logistic Regression:
{
    "C": 4.754571532049581
}

### Uncertainty Analysis
Bootstrap resampling (200 iterations) yielded the following uncertainty estimates:
- **Accuracy:** 0.9909 (95% CI: 0.9680-1.0000)
- **F1 Score:** 0.9945 (95% CI: 0.9810-1.0000)
- **AUC:** 0.9995 (95% CI: 0.9968-1.0000)

### Usage Guidelines
This model should be used as a decision support tool, not as the sole determinant for contract or draft decisions. Best practices include:
1. Consider the model's predicted probability alongside traditional scouting
2. Pay attention to feature importance to understand what's driving a specific prediction
3. Recognize that the model is trained on historical data and may not capture emerging trends
4. Re-evaluate predictions as more data becomes available for a player
    