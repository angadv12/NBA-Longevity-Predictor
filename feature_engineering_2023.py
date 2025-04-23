import pandas as pd
from feature_engineering_2018 import engineer_features

df = pd.read_csv('CLEAN_nba_draft_classes_2019_2023.csv')

if __name__ == "__main__":
    # Load the cleaned data
    df = pd.read_csv('CLEAN_nba_draft_classes_2019_2023.csv')
    engineered_features = engineer_features(df)
    model_df = pd.concat([df, engineered_features], axis=1)

    if 'Yrs' in model_df.columns:
        model_df['Long_Career'] = (model_df['Yrs'] >= 5).astype(int)
        print(f"Class distribution - Long careers: {model_df['Long_Career'].sum()}, Short careers: {len(model_df) - model_df['Long_Career'].sum()}")

    model_df.to_csv('nba_engineered_data_2023.csv', index=False)