import pandas as pd
import time
from bball_ref_script_1980_2015 import get_draft_class_robust

if __name__ == "__main__":
    # Collect draft classes
    draft_classes = pd.DataFrame()
    for year in range(2016, 2021):  # 2016-2020
        draft_class = get_draft_class_robust(year)
        if not draft_class.empty:
            print(f"Successfully collected data for {year}: {len(draft_class)} players")
            draft_classes = pd.concat([draft_classes, draft_class], ignore_index=True)
        else:
            print(f"No data collected for year {year}")
        
        time.sleep(2)

    # Save the raw data
    if not draft_classes.empty:
        draft_classes.to_csv("nba_draft_classes_2016_2020.csv", index=False)
        print(f"Total players collected: {len(draft_classes)}")
        print(draft_classes.head())
    else:
        print("No draft data was collected.")