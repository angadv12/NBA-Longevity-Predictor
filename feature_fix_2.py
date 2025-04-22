import pandas as pd

df = pd.read_csv('nba_draft_classes_1977_2018.csv')

missing_stats = ['Yrs', 'G', 'MP', 'PTS_Total', 'TRB_Total', 'AST_Total', 'FG%', 
                '3P%', 'FT%', 'MPG', 'PPG', 'RPG', 'APG', 'WS', 'WS/48', 'BPM', 'VORP']
missing_stats_mask = df[missing_stats].isna().any(axis=1)
players_with_missing_stats = df[missing_stats_mask]
print(f"Found {len(players_with_missing_stats)} players with missing statistics")

df_clean = df.dropna()
missing_stats_mask = df_clean[missing_stats].isna().any(axis=1)
players_with_missing_stats = df_clean[missing_stats_mask]
print(f"Found {len(players_with_missing_stats)} players with missing statistics")

df_clean.to_csv('CLEAN_nba_draft_classes_1977_2018.csv', index=False)