import pandas as pd

def normalize_player_names(df):
    # if needed, implement name normalization here
    return df

def main():
    # 1) Load the combined first-two-years CSV
    df = pd.read_csv('data/first_two_years_1980_2016.csv')
    clean_df = pd.read_csv('data/CLEAN_nba_draft_classes_1980_2016.csv')

    # 2) Sum up any mid-season splits so each player has one row per Season & Year_Number
    #    We group on the identifying columns and sum all other numeric columns.
    id_cols = ['Player', 'Draft_Year', 'Season', 'Year_Number']
    numeric_cols = df.select_dtypes(include='number').columns.difference(['Season','Year_Number','Draft_Year'])
    agg_dict = {col: 'sum' for col in numeric_cols}
    season_df = (
        df
        .groupby(id_cols, as_index=False)
        .agg(agg_dict)
    )

    # 3) Keep only players who have exactly two seasons (Year_Number 1 & 2)
    counts = season_df.groupby('Player')['Year_Number'].nunique()
    valid_players = counts[counts == 2].index
    season_df = season_df[season_df['Player'].isin(valid_players)]

    # 4) For each player, average their stats across Year_Number 1 and 2
    #    We drop Season and Year_Number, then take the mean of all numeric columns.
    avg_df = (
        season_df
        .drop(columns=['Season', 'Year_Number'])
        .groupby(['Player', 'Draft_Year'], as_index=False)
        .mean()
    )

    print(avg_df.isnull().sum())

    # sort by draft year
    avg_df['Draft_Year'] = pd.to_numeric(avg_df['Draft_Year'], errors='coerce')
    avg_df = avg_df.sort_values(by='Draft_Year')
    print(avg_df.head(30))

    clean_df['Draft_Year'] = pd.to_numeric(clean_df['Draft_Year'], errors='coerce')
    # extract the unique Yrs per Player & Draft_Year
    yrs_info = clean_df[['Player', 'Draft_Year', 'Yrs']].drop_duplicates()
    avg_df = avg_df.merge(yrs_info, on=['Player', 'Draft_Year'], how='left')
    
    # flag long careers (>=5 years)
    avg_df['long_career'] = (pd.to_numeric(avg_df['Yrs'], errors='coerce') >= 5).astype(int)

    # 5) Save to a new CSV
    avg_df.to_csv('data/AVERAGED_year_1_2.csv', index=False)
    print(f"Saved {len(avg_df)} players (with long_career flag) to data/AVERAGED_year_1_2.csv")

if __name__ == '__main__':
    main()
