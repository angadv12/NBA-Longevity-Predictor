
import pandas as pd
import numpy as np
import joblib

def predict_career_longevity(player_data):
    """
    Make career longevity predictions for new NBA players
    
    Parameters:
    player_data : DataFrame with the same features used in training
    
    Returns:
    DataFrame with predictions and probabilities
    """
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
