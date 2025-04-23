import pandas as pd

df = pd.read_csv('Seasons_Stats.csv')

print(df.head())

# drop columns
df = df.drop(df.columns[0], axis=1)
df = df.drop(columns=['blanl', 'blank2', 'GS', 'Tm', 'Pos', 'Age'])

# limit years
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df[(df['Year'] >= 1980)]

df.to_csv('data/cleaning.csv', index=False)