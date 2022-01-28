# Import necessary packages
import webtools.output_readability_enhancer as enh
import selenium.webdriver as webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.options import Options
import os

def automatic_websearch(keyword_list, searchengine_list, default_keyword, csv_path):
    """
    keyword_list (please use whitespace as separator gets passed to every searchengine in searchengine_list.
    New Tab for every keyword_list - item.
    Searchengine Format:
    Search for default_keyword on every searchengine, save the url and pass the default_keyword as argument
    Also generates .csv File csv_path for search-progress tracking.
    """
    # Only import pandas if automatic websearch function is needed
    import pandas as pd

    # Generate new options File
    options = Options()
    # Avoid PopUps
    options.set_preference("dom.disable_open_during_load", False)
    # Set max Tabs to inf
    options.set_preference("dom.popup_maximum", -1)

    # Firefox as Browser, auto download GeckoDriver and set options
    browser = webdriver.Firefox(
        executable_path=GeckoDriverManager().install(), options=options
    )

    enh.print_rd("Initiating online search..")

    # Create Dataframe-Matrix for current Search
    df_searchresults = pd.DataFrame(columns=searchengine_list, index=keyword_list)

    # Merge Dataframe with the current possible searchresults with the df_searchresults_user.csv
    # Replace NaN Values with new User Values
    # NaN: Not Searched
    # 1: Searched, success
    # 0: Searched, no success
    try:
        df_searchresults_merged = df_searchresults.fillna(
            pd.read_csv(csv_path, sep=";", index_col=0)
        )
    except Exception as e:
        enh.print_rd(e)
        df_searchresults_merged = df_searchresults

    try:
        # Save as .csv
        df_searchresults_merged.to_csv(path_or_buf=csv_path, sep=";")
    except Exception as e:
        enh.print_rd(e)

    # Open Browser with random url
    # Will be overwritten on first run
    browser.get("https://www.python.org/")

    # Set counter for tabs
    ctr = 0

    # Loop through searchengines
    for searchengine in searchengine_list:

        # Loop through searchitems
        for searchitem in keyword_list:

            # Only search if not already searched (not searched == NaN) // searched (1 or 0)
            if pd.isna(df_searchresults_merged.loc[searchitem, searchengine]):

                if ctr > 0:
                    # Open new tab
                    browser.execute_script("window.open('');")

                # Switch to the new tab
                browser.switch_to.window(browser.window_handles[ctr])
                # Increase Counter
                ctr += 1
                # Replace whitespace with "+" (if necessary) and open URL
                browser.get(
                    searchengine.replace(default_keyword, searchitem.replace(" ", "+"))
                )

    # Switch to the first tab
    browser.switch_to.window(browser.window_handles[0])
    enh.print_rd(f"{__name__} terminated.")

    if not ctr:
        browser.quit()
        browser = []
        print(f"All elements in {os.path.basename(csv_path)} have already been searched.")

    return browser


def quit_gecko(browser):
    """
    Quit gecko on user-input
    """
    while 1:
        quit_gecko = input("Quit gecko? (1/0)")

        try:
            quit_gecko = int(quit_gecko)
        except Exception as e:
            print("Error:")
            print(e)
            print("Please input 1 or 0!")

        if quit_gecko == 1:
            browser.quit()
            print("Gecko closed.")
            break
        elif quit_gecko == 0:
            break
        else:
            print("Please input 1 or 0!")
