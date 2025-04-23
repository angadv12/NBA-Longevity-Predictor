import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from io import StringIO

def get_draft_class_robust(year):
    """
    More robust function to get draft class data from Basketball Reference
    """
    print(f"Collecting draft class for year: {year}")
    url = f'https://www.basketball-reference.com/draft/NBA_{year}.html'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        r = requests.get(url, headers=headers)
        r.raise_for_status()  # Raise exception for bad status codes
        
        # Print status to debug
        print(f"Status code: {r.status_code}")
        
        soup = BeautifulSoup(r.content, 'html.parser')
        
        # Check for comments in the HTML that might contain the actual table
        comments = soup.find_all(string=lambda text: isinstance(text, str) and 'table class="sortable stats_table"' in text)
        
        if comments:
            # Extract table from comment
            comment = comments[0]
            soup_comment = BeautifulSoup(str(comment), 'html.parser')
            table = soup_comment.find('table')
        else:
            # Try looking for the table directly
            table = soup.find('table', {'id': 'draft', 'class': 'stats_table'}) or \
                   soup.find('table', {'id': 'stats'}) or \
                   soup.find('table', {'class': 'sortable stats_table'}) or \
                   soup.find('table')
        
        if table is None:
            print(f"No draft table found for year {year}")
            return pd.DataFrame()
            
        # Use StringIO to wrap the HTML string
        html_string = str(table)
        
        # Try to get column names first
        headers = [th.text for th in table.find_all('th') if th.get('scope') == 'col']
        if not headers:
            headers = [th.text for th in table.find_all('th')]
        
        print(f"Found headers: {headers}")
        
        try:
            # Try to read with pandas
            dfs = pd.read_html(StringIO(html_string))
            if not dfs:
                print(f"pandas read_html found no tables for year {year}")
                return pd.DataFrame()
                
            df = dfs[0]
            
            # Clean up column names if multi-level
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            
            # Check if "Player" column exists, if not try to find it
            if 'Player' not in df.columns:
                player_cols = [col for col in df.columns if 'Player' in str(col) or 'Name' in str(col)]
                if player_cols:
                    df = df.rename(columns={player_cols[0]: 'Player'})
                else:
                    # If still no Player column, look at first few rows to determine which column contains names
                    print(f"First few columns: {list(df.columns[:5])}")
                    print(f"First few rows: {df.iloc[:3, :5]}")
                    
                    # Assign the likely player name column
                    # This is a guess - often it's the 2nd or 3rd column
                    if len(df.columns) > 2:
                        df = df.rename(columns={df.columns[2]: 'Player'})
                    
            # Clean up the dataframe - drop rows with no player data
            df = df[df['Player'].notna()]
            
            # Drop any rows that are header repeats
            if 'Player' in df.columns:
                df = df[~df['Player'].str.contains('Player', na=False)]
            
            # Add year to the dataframe
            df['Draft_Year'] = year
            
            return df
            
        except Exception as e:
            print(f"Error parsing table with pandas for year {year}: {e}")
            
            # Fallback: Manual parsing of the table
            rows = []
            for tr in table.find_all('tr')[1:]:  # Skip header row
                row = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
                if row and len(row) > 1:  # Ensure we have actual data
                    rows.append(row)
            
            if rows:
                # Create manually parsed dataframe
                df = pd.DataFrame(rows)
                # Use first row as header if it looks like a header
                if df.iloc[0].str.contains('Pk|Player|Pos|College').any():
                    df.columns = df.iloc[0]
                    df = df.iloc[1:]
                else:
                    # Assign default column names
                    df.columns = ['Pk', 'Player', 'Pos', 'Age', 'Team', 'College'] + \
                                 [f'Col{i}' for i in range(7, len(df.columns) + 1)]
                
                # Ensure Player column exists
                if 'Player' not in df.columns and len(df.columns) > 1:
                    df = df.rename(columns={df.columns[1]: 'Player'})
                    
                df['Draft_Year'] = year
                return df
            
            print(f"Failed to manually parse table for year {year}")
            return pd.DataFrame()
            
    except requests.exceptions.RequestException as e:
        print(f"Request error for year {year}: {e}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Failed to collect draft data for year {year}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Collect draft classes
    draft_classes = pd.DataFrame()
    for year in range(1980, 2016):  # 1980-2015
        draft_class = get_draft_class_robust(year)
        if not draft_class.empty:
            print(f"Successfully collected data for {year}: {len(draft_class)} players")
            draft_classes = pd.concat([draft_classes, draft_class], ignore_index=True)
        else:
            print(f"No data collected for year {year}")
        
        time.sleep(2)

    # Save the raw data
    if not draft_classes.empty:
        draft_classes.to_csv("nba_draft_classes_1980_2015.csv", index=False)
        print(f"Total players collected: {len(draft_classes)}")
        print(draft_classes.head())
    else:
        print("No draft data was collected.")