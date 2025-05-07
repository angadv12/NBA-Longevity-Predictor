"""
NBA Player Career Longevity Prediction Project - Complete Solution
This script implements a machine learning framework to predict NBA career longevity
using early-career indicators and statistics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import joblib
import shap
import os
import json
import datetime

# Create a results directory if it doesn't exist
for d in ['results', 'models', 'visualizations']:
    os.makedirs(d, exist_ok=True)

# Data Loading
def load_data():
    """
    Load the pre-separated datasets.
    Returns X_train, y_train, X_test, y_test, and other necessary data.
    """
    print("Loading datasets...")
    
    # Load train data (1977-2018)
    df = pd.read_csv('data/AVERAGED_year_1_2.csv')

    # Data Splitting
    X = df.drop(['Player', 'long_career', 'Yrs'], axis=1, errors='ignore')
    y = df['long_career']

    train_mask = (df['Draft_Year'] <= 2010)
    test_mask = (df['Draft_Year'] >= 2011)

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    X_train = X_train.drop(['Draft_Year'], axis=1, errors='ignore')
    X_test = X_test.drop(['Draft_Year'], axis=1, errors='ignore')

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    # Get the original dataframes for later use
    train_data = df_train
    test_data = df_test
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    return X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, train_data, test_data, scaler

# Baseline Models
def create_baseline_models(X_train_scaled, y_train, X_test_scaled, y_test, test_data):
    """
    Create and evaluate baseline models for comparison.
    """
    print("\nCreating baseline models...")
    baseline_results = {}
    
    # Simple logistic regression baseline
    log_reg = LogisticRegression(penalty='l1', solver='liblinear', C=0.01, random_state=42)
    log_reg.fit(X_train_scaled, y_train)
    
    # Evaluate on test set
    log_y_pred = log_reg.predict(X_test_scaled)
    log_y_prob = log_reg.predict_proba(X_test_scaled)[:, 1]
    
    log_accuracy = accuracy_score(y_test, log_y_pred)
    log_f1 = f1_score(y_test, log_y_pred)
    log_auc = roc_auc_score(y_test, log_y_prob)
    
    print("Logistic Regression Baseline Performance:")
    print(f"Accuracy: {log_accuracy:.4f}")
    print(f"F1 Score: {log_f1:.4f}")
    print(f"AUC-ROC: {log_auc:.4f}")
    
    baseline_results['logistic_regression'] = {
        'accuracy': log_accuracy,
        'f1': log_f1,
        'auc': log_auc
    }
    
    # Create draft position based baseline
    if 'draft_position' in test_data.columns:
        threshold = 15  # First-round picks often have better careers
        
        # Create predictions based on draft position
        draft_preds = (test_data['draft_position'] <= threshold).astype(int)
        
        draft_accuracy = accuracy_score(y_test, draft_preds)
        draft_f1 = f1_score(y_test, draft_preds)
        
        print("\nDraft Position Baseline Performance:")
        print(f"Accuracy: {draft_accuracy:.4f}")
        print(f"F1 Score: {draft_f1:.4f}")
        
        baseline_results['draft_position'] = {
            'accuracy': draft_accuracy,
            'f1': draft_f1
        }
    
    # Export baseline results
    with open('results/baseline_model_results.json', 'w') as f:
        json.dump(baseline_results, f, indent=4)
    
    return log_reg, baseline_results

# Stacked Ensemble Model
def create_ensemble_model(X_train_scaled, y_train):
    """
    Create the stacked ensemble model as described in the paper.
    """
    print("\nCreating stacked ensemble model...")
    # Base models
    random_forest = RandomForestClassifier(
        n_estimators=500, 
        max_depth=15,
        random_state=42
    )
    
    logistic_regression = LogisticRegression(
        penalty='l1', 
        C=0.01,
        solver='liblinear',
        random_state=42
    )
    
    gradient_boosting = GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=100,
        random_state=42
    )
    
    # Create the ensemble
    estimators = [
        ('rf', random_forest),
        ('lr', logistic_regression),
        ('gb', gradient_boosting)
    ]
    
    stacked_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(random_state=42),
        cv=5
    )
    
    # Fit the model
    print("Training ensemble model...")
    stacked_model.fit(X_train_scaled, y_train)
    print("Ensemble model training complete!")
    
    return stacked_model

# Model Evaluation
def evaluate_model(model, X, y, model_name="Model", plot_figures=True, save_figures=True):
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)

    precision, recall, thresholds = precision_recall_curve(y, y_prob)
    target_recall = 0.75
    closest_idx = np.argmin(np.abs(recall - target_recall))
    precision_at_75_recall = precision[closest_idx]

    print(f"{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Precision at 75% Recall: {precision_at_75_recall:.4f}")

    if plot_figures:
        # Confusion matrix (no bar annotations)
        cm = confusion_matrix(y, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{model_name} Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        if save_figures:
            plt.savefig(f'visualizations/{model_name.replace(" ", "_")}_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # ROC curve
        fpr, tpr, _ = roc_curve(y, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'{model_name} ROC Curve')
        plt.legend()
        if save_figures:
            plt.savefig(f'visualizations/{model_name.replace(" ", "_")}_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    metrics = {'accuracy': accuracy, 'f1': f1, 'auc': auc, 'precision_at_75_recall': precision_at_75_recall}
    if 'cm' in locals(): metrics['confusion_matrix'] = cm.tolist()
    return metrics

# Feature Importance Analysis
def analyze_feature_importance(ensemble_model, X_train, X_train_scaled, save_figures=True):
    print("\nAnalyzing feature importance...")
    rf_model = ensemble_model.named_estimators_['rf']
    importances = rf_model.feature_importances_
    importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
    importance_df.sort_values('Importance', ascending=False, inplace=True)
    importance_df.reset_index(drop=True, inplace=True)
    importance_df.to_csv('results/feature_importance.csv', index=False)

    # Plot top 15 features with annotation
    plt.figure(figsize=(12, 8))
    ax = sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
    # annotate bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3)
    plt.title('Top 15 Most Important Features for NBA Career Longevity')
    plt.tight_layout()
    if save_figures:
        plt.savefig('visualizations/feature_importance_top15.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Top 10 Most Important Features:")
    print(importance_df.head(10))
    
    # Use SHAP for additional insights
    try:
        # Create SHAP explainer
        explainer = shap.TreeExplainer(rf_model)
        
        # If dataset is large, use a sample
        if X_train_scaled.shape[0] > 500:
            sample_indices = np.random.choice(X_train_scaled.shape[0], 500, replace=False)
            X_sample = X_train_scaled[sample_indices]
            # Create a DataFrame with feature names for SHAP plots
            X_sample_df = pd.DataFrame(X_sample, columns=X_train.columns)
        else:
            X_sample = X_train_scaled
            X_sample_df = pd.DataFrame(X_sample, columns=X_train.columns)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_sample)
        
        # Plot summary
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample_df, plot_type="bar", show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        if save_figures:
            plt.savefig('visualizations/shap_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Show detailed summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample_df, show=False)
        plt.title('SHAP Summary Plot')
        plt.tight_layout()
        if save_figures:
            plt.savefig('visualizations/shap_summary_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("SHAP analysis completed and saved.")
        return importance_df, rf_model, explainer
    except Exception as e:
        print(f"Error in SHAP analysis: {e}")
        return importance_df, rf_model, None

# Bootstrap Resampling for Uncertainty Quantification
def bootstrap_evaluation(model, X, y, n_iterations=200, save_figures=True):
    print("\nPerforming bootstrap evaluation...")
    accuracies, f1_scores, aucs = [], [], []
    np.random.seed(42)
    for i in range(n_iterations):
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_sample, y_sample = X[indices], y.iloc[indices]
        y_pred = model.predict(X_sample)
        y_prob = model.predict_proba(X_sample)[:, 1]
        accuracies.append(accuracy_score(y_sample, y_pred))
        f1_scores.append(f1_score(y_sample, y_pred))
        aucs.append(roc_auc_score(y_sample, y_prob))

    results = {}
    # calculate confidence intervals
    for name, vals in [('accuracy', accuracies), ('f1_score', f1_scores), ('auc', aucs)]:
        lower, upper = np.percentile(vals, [2.5, 97.5])
        mean = np.mean(vals)
        results[name] = {'mean': mean, 'lower_ci': lower, 'upper_ci': upper}
        print(f"{name}: {mean:.4f} (95% CI: {lower:.4f}-{upper:.4f})")

    # visualize distributions with annotations
    if save_figures:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # Add overall title
        fig.suptitle('Bootstrap Distributions', fontsize=16)
        for ax, data, title in zip(axes, [accuracies, f1_scores, aucs], ['Accuracy', 'F1 Score', 'AUC']):
            counts, bins, patches = ax.hist(data, bins=30, alpha=0.7)
            ax.axvline(np.mean(data), color='red')
            ax.set_title(f'{title} Distribution')
            # annotate bars with smaller font size to avoid overlap
            for count, patch in zip(counts, patches):
                ax.annotate(f'{count:.0f}',
                            (patch.get_x() + patch.get_width() / 2, count),
                            textcoords='offset points',
                            xytext=(0, 3),
                            ha='center',
                            fontsize=6)
        plt.tight_layout()
        # Adjust layout to accommodate the suptitle
        plt.subplots_adjust(top=0.85)
        plt.savefig('visualizations/bootstrap_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()

    # save bootstrap results
    with open('results/bootstrap_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    return results

# Model Comparison
def compare_all_models(X_train_scaled, y_train, X_test_scaled, y_test, ensemble, save_figures=True):
    print("\nComparing all models...")
    base_rf = RandomForestClassifier(
        n_estimators=500, 
        max_depth=15,
        random_state=42
    )
    
    base_lr = LogisticRegression(
        penalty='l1', 
        C=0.01,
        solver='liblinear',
        random_state=42
    )
    
    base_gb = GradientBoostingClassifier(
        learning_rate=0.1,
        max_depth=3,
        n_estimators=100,
        random_state=42
    )
    models = {
        'Logistic Regression': base_lr,
        'Random Forest': base_rf,
        'Gradient Boosting': base_gb,
        'Ensemble': ensemble
    }
    results = {}
    for name, model in models.items():
        print(f"Evaluating {name}...")
        if name != 'Ensemble': model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        results[name] = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1 Score': f1_score(y_test, y_pred),
            'AUC': roc_auc_score(y_test, y_prob)
        }
    comparison_df = pd.DataFrame(results).T
    comparison_df.to_csv('results/model_comparison.csv')

    if save_figures:
        ax = comparison_df.plot(kind='bar', figsize=(12, 6))
        plt.title('Model Performance Comparison')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        # annotate bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3)
        plt.tight_layout()
        plt.savefig('visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Combined ROC curves
        plt.figure(figsize=(8, 6))
        for name, model in models.items():
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_val = roc_auc_score(y_test, y_prob)
            plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for All Models')
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.savefig('visualizations/all_models_roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()

    print("\nModel Comparison:")
    print(comparison_df)
    return comparison_df


# Prediction function for new players
def create_prediction_function(model, scaler, X_train):
    """
    Create and test a function for making predictions on new players.
    """
    def predict_career_longevity(player_data):
        """
        Make career longevity predictions for new NBA players
        
        Parameters:
        player_data : DataFrame with the same features used in training
        
        Returns:
        DataFrame with predictions and probabilities
        """
        # Preprocess the data (assuming player_data has the same structure as training data)
        # Drop columns not used in prediction
        id_cols = ['player_id', 'player_name', 'draft_year']
        features = player_data.drop(id_cols, axis=1, errors='ignore')
        
        # Check if features match the expected columns
        missing_cols = set(X_train.columns) - set(features.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")
        
        # Ensure column order matches training data
        features = features[X_train.columns]
        
        # Scale the features
        features_scaled = scaler.transform(features)
        
        # Make predictions
        probabilities = model.predict_proba(features_scaled)[:, 1]
        predictions = model.predict(features_scaled)
        
        # Add results to original data
        results = player_data.copy()
        results['long_career_probability'] = probabilities
        results['predicted_long_career'] = predictions
        results['prediction_confidence'] = np.maximum(probabilities, 1-probabilities)
        
        # Add interpretation
        def interpret_prediction(prob):
            if prob > 0.9:
                return "Very High Confidence (>90%) of Long Career"
            elif prob > 0.75:
                return "High Confidence (>75%) of Long Career"
            elif prob > 0.5:
                return "Moderate Confidence of Long Career"
            elif prob > 0.25:
                return "Low Confidence of Long Career (likely short)"
            else:
                return "Very Low Confidence (<25%) of Long Career (very likely short)"
        
        results['interpretation'] = results['long_career_probability'].apply(interpret_prediction)
        
        return results
    
    # Save the prediction function and example to a file
    with open('results/prediction_function.py', 'w') as f:
        f.write("""
import pandas as pd
import numpy as np
import joblib

def predict_career_longevity(player_data):
    \"\"\"
    Make career longevity predictions for new NBA players
    
    Parameters:
    player_data : DataFrame with the same features used in training
    
    Returns:
    DataFrame with predictions and probabilities
    \"\"\"
    # Load the model and scaler
    model = joblib.load('models/nba_career_longevity_model.pkl')
    scaler = joblib.load('models/nba_career_longevity_scaler.pkl')
    
    # Get required columns from the model
    required_columns = pd.read_csv('results/required_columns.csv')['columns'].tolist()
    
    # Preprocess the data
    # Drop columns not used in prediction
    id_cols = ['player_id', 'player_name', 'draft_year']
    features = player_data.drop(id_cols, axis=1, errors='ignore')
    
    # Check if features match the expected columns
    missing_cols = set(required_columns) - set(features.columns)
    if missing_cols:
        raise ValueError(f"Missing columns in input data: {missing_cols}")
    
    # Ensure column order matches training data
    features = features[required_columns]
    
    # Scale the features
    features_scaled = scaler.transform(features)
    
    # Make predictions
    probabilities = model.predict_proba(features_scaled)[:, 1]
    predictions = model.predict(features_scaled)
    
    # Add results to original data
    results = player_data.copy()
    results['long_career_probability'] = probabilities
    results['predicted_long_career'] = predictions
    results['prediction_confidence'] = np.maximum(probabilities, 1-probabilities)
    
    # Add interpretation
    def interpret_prediction(prob):
        if prob > 0.9:
            return "Very High Confidence (>90%) of Long Career"
        elif prob > 0.75:
            return "High Confidence (>75%) of Long Career"
        elif prob > 0.5:
            return "Moderate Confidence of Long Career"
        elif prob > 0.25:
            return "Low Confidence of Long Career (likely short)"
        else:
            return "Very Low Confidence (<25%) of Long Career (very likely short)"
    
    results['interpretation'] = results['long_career_probability'].apply(interpret_prediction)
    
    return results

# Example usage
# new_players = pd.read_csv('new_nba_players.csv')
# predictions = predict_career_longevity(new_players)
# predictions[['player_name', 'long_career_probability', 'predicted_long_career', 'interpretation']].head()
""")
    
    # Save required columns for prediction
    pd.DataFrame({'columns': X_train.columns}).to_csv('results/required_columns.csv', index=False)
    
    # Save the scaler
    joblib.dump(scaler, 'models/nba_career_longevity_scaler.pkl')
    
    print("Prediction function generated and saved to 'results/prediction_function.py'")
    
    return predict_career_longevity

# Export all statistics to a summary file
def export_statistics(ensemble_metrics, importance_df, model_comparison, bootstrap_results):
    """
    Export all statistics to summary files.
    """
    print("\nExporting statistics...")
    
    # Create a summary statistics file
    summary = {
        'execution_date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'ensemble_metrics': ensemble_metrics,
        'top_features': importance_df.head(10).to_dict('records'),
        'model_comparison': model_comparison.to_dict(),
        'bootstrap_results': bootstrap_results,
    }
    
    # Save to JSON
    with open('results/project_summary_statistics.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Create a text summary file for quick reference
    with open('results/summary_report.txt', 'w') as f:
        f.write("NBA PLAYER CAREER LONGEVITY PREDICTION - SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("EXECUTION DATE: " + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n\n")
        
        f.write("ENSEMBLE MODEL PERFORMANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {ensemble_metrics['accuracy']:.4f}\n")
        f.write(f"F1 Score: {ensemble_metrics['f1']:.4f}\n")
        f.write(f"AUC-ROC: {ensemble_metrics['auc']:.4f}\n")
        f.write(f"Precision at 75% Recall: {ensemble_metrics['precision_at_75_recall']:.4f}\n\n")
        
        f.write("TOP 10 MOST IMPORTANT FEATURES\n")
        f.write("-" * 40 + "\n")
        for i, row in importance_df.head(10).iterrows():
            f.write(f"{i+1}. {row['Feature']}: {row['Importance']:.4f}\n")
        f.write("\n")
        
        f.write("MODEL COMPARISON\n")
        f.write("-" * 40 + "\n")
        f.write(model_comparison.to_string() + "\n\n")
        
        f.write("BOOTSTRAP RESULTS (95% CONFIDENCE INTERVALS)\n")
        f.write("-" * 40 + "\n")
        for metric, values in bootstrap_results.items():
            f.write(f"{metric.replace('_', ' ').title()}: {values['mean']:.4f} (95% CI: {values['lower_ci']:.4f}-{values['upper_ci']:.4f})\n")
        
        f.write("\n\nAll detailed results available in the 'results' directory.\n")
    
    print("Statistics exported to 'results/project_summary_statistics.json' and 'results/summary_report.txt'")

# Main function
def main():
    """
    Main function to run the entire project.
    """
    print("=" * 80)
    print("NBA PLAYER CAREER LONGEVITY PREDICTION PROJECT")
    print("=" * 80)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Load and prepare data
    X_train, y_train, X_test, y_test, X_train_scaled, X_test_scaled, train_data, test_data, scaler = load_data()
    
    # 2. Create baseline models
    log_reg, baseline_results = create_baseline_models(X_train_scaled, y_train, X_test_scaled, y_test, test_data)
    
    # 3. Create initial ensemble model
    ensemble_model = create_ensemble_model(X_train_scaled, y_train)
    
    # 6. Evaluate model
    ensemble_metrics = evaluate_model(ensemble_model, X_test_scaled, y_test, "Ensemble", plot_figures=True)
    
    # 7. Analyze feature importance
    importance_df, rf_model, shap_explainer = analyze_feature_importance(ensemble_model, X_train, X_train_scaled)
    
    # 8. Perform bootstrap evaluation
    bootstrap_results = bootstrap_evaluation(ensemble_model, X_test_scaled, y_test, n_iterations=1000)
    
    # 9. Compare all models
    model_comparison = compare_all_models(X_train_scaled, y_train, X_test_scaled, y_test, ensemble_model)
    
    # 10. Create prediction function
    predict_function = create_prediction_function(ensemble_model, scaler, X_train)
    
    # 13. Export all statistics
    export_statistics(ensemble_metrics, importance_df, model_comparison, bootstrap_results)
    
    print("\n" + "=" * 80)
    print("PROJECT EXECUTION COMPLETE")
    print("=" * 80)
    print("\nAll results, visualizations, and models have been saved to their respective directories.")
    print("- Models saved to: 'models/'")
    print("- Visualizations saved to: 'visualizations/'")
    print("- Results and reports saved to: 'results/'")
    
    return {
        'ensemble_model': ensemble_model,
        'metrics': ensemble_metrics,
        'importance': importance_df,
        'bootstrap': bootstrap_results,
        'comparison': model_comparison
    }

# Run the entire project
if __name__ == "__main__":
    main()