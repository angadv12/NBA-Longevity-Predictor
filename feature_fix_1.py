import pandas as pd
df = pd.read_csv('nba_draft_classes_1977_2018.csv')
import time
from io import StringIO
import requests
import re
from bs4 import BeautifulSoup

df = df.drop('Unnamed: 0_level_0', axis=1)

# remove any row that matches column name patterns
header_indicators = ['Pk', 'Tm', 'Player', 'College', 'Yrs', 'G', 'MP', 'PTS', 'TRB', 'AST', 'FG%', 
                     '3P%', 'FT%', 'WS', 'WS/48', 'BPM', 'VORP']
mask = df.apply(lambda row: not any(str(val) in header_indicators for val in row), axis=1)
df = df[mask]

# rename column headers
df.rename(columns={'Unnamed: 1_level_0': 'Pk',
                   'Player': 'Tm',
                   'Player': 'Player',
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

# scrape missing data using basketball_reference_scraper
missing_stats = ['Yrs', 'G', 'MP', 'PTS_Total', 'TRB_Total', 'AST_Total', 'FG%', 
                '3P%', 'FT%', 'MPG', 'PPG', 'RPG', 'APG', 'WS', 'WS/48', 'BPM', 'VORP']
missing_stats_mask = df[missing_stats].isna().any(axis=1)
players_with_missing_stats = df[missing_stats_mask]
print(f"Found {len(players_with_missing_stats)} players with missing statistics")

stat_mapping = {
    # TOTALS stats
    'G': 'G',
    'MP': 'MP', 
    'PTS_Total': 'PTS',
    'TRB_Total': 'TRB',  
    'AST_Total': 'AST',
    
    # PER_GAME stats
    'MPG': 'MP',
    'PPG': 'PTS', 
    'RPG': 'TRB',
    'APG': 'AST',
    'FG%': 'FG%',
    '3P%': '3P%',
    'FT%': 'FT%',
    
    # ADVANCED stats
    'WS': 'WS',
    'WS/48': 'WS/48',
    'BPM': 'BPM',
    'VORP': 'VORP'
}

# Custom function to get player stats directly (avoiding the issues with the scraper library)
def get_player_stats(player_name, draft_year, college=None):
    """
    Custom function to get player stats directly from Basketball Reference
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Step 1: Find the player's page
    player_found = False
    player_url = None
    
    # Try to get a more specific match using draft year and college
    search_url = f"https://www.basketball-reference.com/search/search.fcgi?search={player_name}"
    try:
        response = requests.get(search_url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Check if we're redirected directly to a player page
        if 'players' in response.url:
            player_url = response.url
            player_found = True
            print(f"Direct match found for {player_name}")
        else:
            # Look for search results
            search_results = soup.find('div', {'id': 'players'})
            if search_results:
                results = search_results.find_all('div', {'class': 'search-item'})
                print(f"Found {len(results)} possible matches for {player_name}")
                
                # Look for the best match considering draft year and college
                best_match = None
                for result in results:
                    # Extract relevant data
                    result_link = result.find('a', href=True)
                    if not result_link:
                        continue
                    
                    result_url = f"https://www.basketball-reference.com{result_link['href']}"
                    result_text = result.get_text()
                    
                    # Check if draft year is in the result
                    year_match = re.search(r'(\d{4})-(\d{4}|\d{2})', result_text)
                    player_college = None
                    college_div = result.find('div', {'class': 'search-item-url'})
                    if college_div:
                        player_college = college_div.get_text().strip()
                    
                    # If we have a year match and it includes the draft year
                    if year_match and str(draft_year) in year_match.group(0):
                        # If college is provided and matches, this is likely our player
                        if college and player_college and college.lower() in player_college.lower():
                            best_match = result_url
                            break
                        # Otherwise, just take the year match as a good candidate
                        elif not best_match:
                            best_match = result_url
                
                if best_match:
                    player_url = best_match
                    player_found = True
                elif results:
                    # Just take the first result if no better matches
                    result_link = results[0].find('a', href=True)
                    if result_link:
                        player_url = f"https://www.basketball-reference.com{result_link['href']}"
                        player_found = True
    except Exception as e:
        print(f"Error searching for {player_name}: {e}")
    
    if not player_found or not player_url:
        print(f"No player page found for {player_name}")
        return None, None, None
    
    # Step 2: Get the player stats
    totals_df = None
    per_game_df = None
    advanced_df = None
    
    try:
        response = requests.get(player_url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract Totals table
        totals_table = soup.find('table', {'id': 'totals'})
        if totals_table:
            html_string = str(totals_table)
            totals_df = pd.read_html(StringIO(html_string))[0]
            # Fix multi-level columns if present
            if isinstance(totals_df.columns, pd.MultiIndex):
                totals_df.columns = totals_df.columns.get_level_values(1)
            
        # Extract Per Game table
        per_game_table = soup.find('table', {'id': 'per_game'})
        if per_game_table:
            html_string = str(per_game_table)
            per_game_df = pd.read_html(StringIO(html_string))[0]
            if isinstance(per_game_df.columns, pd.MultiIndex):
                per_game_df.columns = per_game_df.columns.get_level_values(1)
            
        # Extract Advanced table
        advanced_table = soup.find('table', {'id': 'advanced'})
        if advanced_table:
            html_string = str(advanced_table)
            advanced_df = pd.read_html(StringIO(html_string))[0]
            if isinstance(advanced_df.columns, pd.MultiIndex):
                advanced_df.columns = advanced_df.columns.get_level_values(1)
    
    except Exception as e:
        print(f"Error getting stats for {player_name}: {e}")
    
    return totals_df, per_game_df, advanced_df

# Track progress
updated_players = 0
failed_players = []

# Create a mapping between your CSV columns and Basketball Reference columns
stat_mapping = {
    # From TOTALS stats
    'G': 'G',
    'MP': 'MP', 
    'PTS_Total': 'PTS',
    'TRB_Total': 'TRB',  
    'AST_Total': 'AST',
    
    # From PER_GAME stats
    'MPG': 'MP',
    'PPG': 'PTS', 
    'RPG': 'TRB',
    'APG': 'AST',
    'FG%': 'FG%',
    '3P%': '3P%',
    'FT%': 'FT%',
    
    # From ADVANCED stats
    'WS': 'WS',
    'WS/48': 'WS/48',
    'BPM': 'BPM',
    'VORP': 'VORP'
}

for index, player in players_with_missing_stats.iterrows():
    player_name = player['Player']  # Adjust based on your actual player name column
    draft_year = player['Draft_Year']
    college = player['College'] if 'College' in player and not pd.isna(player['College']) else None
    
    print(f"Processing {index+1}/{len(players_with_missing_stats)}: {player_name} (Draft: {draft_year})")
    
    totals_df, per_game_df, advanced_df = get_player_stats(player_name, draft_year, college)
    
    if totals_df is not None or per_game_df is not None or advanced_df is not None:
        # Calculate Yrs if available
        if 'Yrs' in missing_stats and pd.isna(player['Yrs']) and totals_df is not None:
            # Count seasons excluding Career row
            career_rows = totals_df[totals_df['Season'] == 'Career'].shape[0]
            df.at[index, 'Yrs'] = len(totals_df) - career_rows
        
        # totals stats
        if totals_df is not None and 'Career' in totals_df['Season'].values:
            career_totals = totals_df[totals_df['Season'] == 'Career'].iloc[0]
            for csv_col, br_col in stat_mapping.items():
                if csv_col in ['G', 'MP', 'PTS_Total', 'TRB_Total', 'AST_Total']:
                    if pd.isna(player[csv_col]) and br_col in career_totals:
                        df.at[index, csv_col] = career_totals[br_col]
        
        # per game stats
        if per_game_df is not None and 'Career' in per_game_df['Season'].values:
            career_per_game = per_game_df[per_game_df['Season'] == 'Career'].iloc[0]
            for csv_col, br_col in stat_mapping.items():
                if csv_col in ['MPG', 'PPG', 'RPG', 'APG', 'FG%', '3P%', 'FT%']:
                    if pd.isna(player[csv_col]) and br_col in career_per_game:
                        df.at[index, csv_col] = career_per_game[br_col]
        
        # advanced stats
        if advanced_df is not None and 'Career' in advanced_df['Season'].values:
            career_advanced = advanced_df[advanced_df['Season'] == 'Career'].iloc[0]
            for csv_col, br_col in stat_mapping.items():
                if csv_col in ['WS', 'WS/48', 'BPM', 'VORP']:
                    if pd.isna(player[csv_col]) and br_col in career_advanced:
                        df.at[index, csv_col] = career_advanced[br_col]
        
        updated_players += 1
        print(f"  Successfully updated stats for {player_name}")
    else:
        failed_players.append(player_name)
        print(f"  No stats found for {player_name}")
    
    # save progress every 10 players
    if updated_players % 10 == 0:
        df.to_csv('nba_draft_classes_in_progress.csv', index=False)
    
    time.sleep(2)

print(f"Updated {updated_players} players")
print(f"Failed to update {len(failed_players)} players")
print("Failed players:", failed_players)

# save to csv
df.to_csv("nba_draft_classes_1977_2018.csv", index=False)