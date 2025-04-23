import pandas as pd

df = pd.read_csv('CLEAN_nba_draft_classes_1977_2018.csv')

def engineer_features(df):
    # empty df for storing features
    features_df = pd.DataFrame(index=df.index)

    # 1. Durability Factor
    if 'G' in df.columns and 'Yrs' in df.columns:
        max_possible_games = df['Yrs'] * 82  # approx max games per season
        features_df['Durability_Factor'] = df['G'] / max_possible_games
    
    # 2. Versatility Score
    if all(col in df.columns for col in ['APG', 'RPG']):
        features_df['Versatility_Score'] = df['APG'] + df['RPG']

    # 3. Draft Position Value x BPM
    if 'Pk' in df.columns and 'BPM' in df.columns:
        features_df['Draft_Pick_Value'] = 1 / df['Pk']  # lower pick = higher value
        features_df['Draft_Pick_x_BPM'] = features_df['Draft_Pick_Value'] * df['BPM']
    
    # 4. Scoring Efficiency
    if 'FG%' in df.columns and 'FT%' in df.columns and '3P%' in df.columns:
        # TS approximation
        features_df['Scoring_Efficiency'] = df['FG%'] * 0.5 + df['3P%'] * 0.4 + df['FT%'] * 0.1

    # 5. Production Per Minute metrics
    if 'MP' in df.columns:
        if 'PTS_Total' in df.columns:
            features_df['Points_Per_Min'] = df['PTS_Total'] / df['MP']
        if 'TRB_Total' in df.columns:
            features_df['Rebounds_Per_Min'] = df['TRB_Total'] / df['MP']
        if 'AST_Total' in df.columns:
            features_df['Assists_Per_Min'] = df['AST_Total'] / df['MP']

    # 6. Win Production metrics
    if 'WS' in df.columns and 'G' in df.columns:
        features_df['WS_Per_Game'] = df['WS'] / df['G']

    # 7. Draft Pedigree - bin draft positions
    if 'Pk' in df.columns:
        features_df['Lottery_Pick'] = (df['Pk'] <= 14).astype(int)
    
    return features_df

if __name__ == "__main__":
    # Load the cleaned data
    df = pd.read_csv('CLEAN_nba_draft_classes_1977_2018.csv')
    engineered_features = engineer_features(df)
    model_df = pd.concat([df, engineered_features], axis=1)

    if 'Yrs' in model_df.columns:
        model_df['Long_Career'] = (model_df['Yrs'] >= 5).astype(int)
        print(f"Class distribution - Long careers: {model_df['Long_Career'].sum()}, Short careers: {len(model_df) - model_df['Long_Career'].sum()}")

    model_df.to_csv('nba_engineered_data_2018.csv', index=False)