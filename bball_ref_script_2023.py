import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
from io import StringIO
from bball_ref_script_2018 import get_draft_class_robust

if __name__ == "__main__":
    # Collect draft classes
    draft_classes = pd.DataFrame()
    for year in range(2019, 2024):  # 2019-2023
        draft_class = get_draft_class_robust(year)
        if not draft_class.empty:
            print(f"Successfully collected data for {year}: {len(draft_class)} players")
            draft_classes = pd.concat([draft_classes, draft_class], ignore_index=True)
        else:
            print(f"No data collected for year {year}")
        
        time.sleep(2)

    # Save the raw data
    if not draft_classes.empty:
        draft_classes.to_csv("nba_draft_classes_2019_2023.csv", index=False)
        print(f"Total players collected: {len(draft_classes)}")
        print(draft_classes.head())
    else:
        print("No draft data was collected.")