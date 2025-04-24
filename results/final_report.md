
# NBA Player Career Longevity Prediction: Final Report

## Project Summary
This project developed a machine learning framework to predict NBA career longevity (≥5 seasons vs. <5 seasons) using early-career indicators. The model aims to address critical team needs: optimizing draft strategies, informing rookie-scale extension decisions, and mitigating financial risks in long-term contracts.

## Model Performance
The final stacked ensemble model achieved the following performance metrics on the 2019-2023 holdout test set:

- **F1 Score:** 0.8990
- **AUC-ROC:** 0.8320
- **Accuracy:** 0.8261
- **Precision at 75% Recall:** 0.8947

## Key Findings
1. **Most Important Predictors:** The most significant predictors of career longevity are:
   - DRB (Importance: 0.0813)
   - TRB (Importance: 0.0757)
   - BPM (Importance: 0.0701)
   - WS (Importance: 0.0662)
   - DWS (Importance: 0.0461)

2. **Model Comparison:** The stacked ensemble approach outperformed individual models, achieving an F1 score improvement of 0.0209 over the baseline logistic regression model.

3. **Bootstrap Analysis:** Our bootstrap resampling confirms the robustness of the model with tight confidence intervals around key metrics.

## Practical Applications
1. **Draft Decision Support:** The model provides an objective assessment of a prospect's likelihood of having a long career, complementing traditional scouting methods.

2. **Contract Valuation:** Teams can use prediction probabilities to help quantify risk in long-term contract negotiations.

3. **Player Development Focus:** Feature importance highlights which skills and attributes have the strongest relationship with career longevity, potentially guiding development programs.

## Future Work
1. **Position-Specific Models:** Develop separate models for guards, forwards, and centers to capture position-specific career trajectory factors.

2. **Injury Prediction Integration:** Incorporate injury prediction models to provide a more comprehensive risk assessment.

3. **International Player Analysis:** Create specialized models for international players with different developmental pathways.
    