import pandas as pd

df = pd.read_csv('nba_draft_classes_1977_2018.csv')

# df = df.drop('Unnamed: 0_level_0', axis=1)

print(df.head())

# remove any row that matches column name patterns
header_indicators = ['Pk', 'Tm', 'Player', 'College', 'Yrs', 'G', 'MP', 'PTS', 'TRB', 'AST', 'FG%', 
                     '3P%', 'FT%', 'WS', 'WS/48', 'BPM', 'VORP']
mask = df.apply(lambda row: not any(str(val) in header_indicators for val in row), axis=1)
df = df[mask]

df.rename(columns={'Unnamed: 1_level_0': 'Pk',
                   'Player': 'Tm',
                   'Round 1': 'Player',
                   'Round 1.1': 'College',
                   'Unnamed: 5_level_0': 'Yrs',
                   'Totals': 'G',
                   'Totals.1': 'MP',
                   'Totals.2': 'PTS_Total',
                   'Totals.3': 'TRB_Total',
                   'Totals.4': 'AST_Total',
                   'Shooting': 'FG%',
                   'Shooting.1': '3P%',
                   'Shooting.2': 'FT%',
                   'Per Game': 'MPG',
                   'Per Game.1': 'PPG',
                   'Per Game.2': 'RPG',
                   'Per Game.3': 'APG',
                   'Advanced': 'WS',
                   'Advanced.1': 'WS/48',
                   'Advanced.2': 'BPM',
                   'Advanced.3': 'VORP',
                   }, 
                   inplace=True)

# only grab first round picks
df = df[df['Pk'] <= 30]
df = df[df['Draft_Year'] >= 1980]

df.to_csv("nba_draft_classes_1977_2018.csv", index=False)