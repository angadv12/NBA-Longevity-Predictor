import pandas as pd
import time
import re

# —————————————————————————————————————————————————————————————————————
def normalize(name):
    """Lowercase, strip trailing *, Jr./Sr./IV, convert 'Last, First' → 'First Last'."""
    if not isinstance(name, str):
        return ''
    # strip HOF asterisks or other trailing symbols
    name = re.sub(r'[\*\+]+$', '', name)
    # strip suffixes
    name = re.sub(r'\s+(Jr\.|Sr\.|I{1,3}|IV)$', '', name)
    # flip "Last, First"
    if ',' in name:
        last, first = name.split(',', 1)
        name = f'{first.strip()} {last.strip()}'
    return re.sub(r'\s+', ' ', name).strip().lower()

# —————————————————————————————————————————————————————————————————————
def load_draft_csv(path='data/CLEAN_nba_draft_classes_1980_2016.csv'):
    df = pd.read_csv(path)
    # ensure correct types
    df['Pk'] = pd.to_numeric(df['Pk'], errors='coerce')
    # drop any bad rows
    df = df.dropna(subset=['Player','Pk','Draft_Year'])
    df['Player_norm'] = df['Player'].map(normalize)
    return df

def first_rounders_for_year(draft_df, year):
    """Return set of normalized names for picks 1–30 in a given draft year."""
    df = draft_df[draft_df['Draft_Year'] == year]
    fr = df[df['Pk'].between(1, 30)]
    return set(fr['Player_norm'])

# —————————————————————————————————————————————————————————————————————
def collect_first_two_years(
    draft_csv='data/CLEAN_nba_draft_classes_1980_2016.csv',
    stats_csv='data/cleaning.csv',
    start_year=1980,
    end_year=2016,
    output_csv='data/first_two_years_1980_2016.csv'
):
    # 1) Load draft and stats
    draft_df = load_draft_csv(draft_csv)
    stats    = pd.read_csv(stats_csv)
    # rename Year→Season if needed
    if 'Year' in stats.columns:
        stats.rename(columns={'Year':'Season'}, inplace=True)
    stats['Player_norm'] = stats['Player'].map(normalize)

    season_cache = {}
    chunks = []

    for draft_year in range(start_year, end_year+1):
        print(f"\n--- Draft {draft_year} ---")
        picks = first_rounders_for_year(draft_df, draft_year)
        if not picks:
            print("  ▶️ no first‐round picks data, skipping")
            continue

        for offset, yr_num in [(1,1),(2,2)]:
            season = draft_year + offset
            if season not in season_cache:
                season_cache[season] = stats[stats['Season']==season]
            df_season = season_cache[season]
            sel = df_season[df_season['Player_norm'].isin(picks)].copy()
            if sel.empty:
                print(f"  Year {yr_num} (Season {season}): 0 matches")
                continue

            sel['Draft_Year']  = draft_year
            sel['Year_Number'] = yr_num
            chunks.append(sel)
            print(f"  Year {yr_num} (Season {season}): {len(sel)} players")

        time.sleep(0.3)

    if chunks:
        result = pd.concat(chunks, ignore_index=True)
        result.drop(columns=['Player_norm'], inplace=True)
        result = result[~((result['Season'] == 2017) & (result['Year_Number'] == 1))]
        result.to_csv(output_csv, index=False)
        print(f"\n✔️ Done! {len(result)} rows → {output_csv}")
    else:
        print("❌ No data collected.")

# —————————————————————————————————————————————————————————————————————
if __name__ == '__main__':
    collect_first_two_years()
