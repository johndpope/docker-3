"""
Package for Text highlighting in Selenium -- Tested on Firefox 92.0.1
"""

# Import necessary packages
import webtools.output_readability_enhancer as enh
from Levenshtein import distance
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import selenium.webdriver as webdriver
from webdriver_manager.firefox import GeckoDriverManager
import time


def _apply_style(s, element):
    """
    Applies style s to element
    """

    driver = element._parent
    driver.execute_script(
        "arguments[0].setAttribute('style', arguments[1]);", element, s
    )


def _highlight_static_local(element):
    """
    Highlights a Selenium Webdriver element
    """

    element_style = "background: transparent; border: 3px solid red;"
    _apply_style(element_style, element)


def _highlight_dynamic_local(element):
    """
    Highlights (blinks) a Selenium Webdriver element -- Use with ThreadPoolExecutor
    """

    # Switch between new and old style for blinking effect
    blink_time = 0.5
    element_style = "background: transparent; border: 3px solid red;"
    while 1:
        original_style = element.get_attribute("style")
        time.sleep(blink_time)
        _apply_style(element_style, element)
        time.sleep(blink_time)
        _apply_style(original_style, element)


def _element_search(browser, keyword_list):
    """
    Searches for elements containing keywords
    """

    # Switch to first tab
    browser.switch_to.window(browser.window_handles[0])
    element_list = []
    # Create list with all elements containing the keywords from keyword_list
    for keyword_item in keyword_list:
        element_list.extend(
            browser.find_elements_by_xpath(
                "//*[contains(text(),'"
                + keyword_item.replace("'", "").replace('"', "")
                + "')]"
            )
        )

    return element_list


def _keyword_list_extension(keyword_list):
    """
    Add every keyword with Capital first letters and with all Capitals to keyword list
    """
    keywords_cap = [keyword.capitalize() for keyword in keyword_list]
    keywords_upper = [keyword.upper() for keyword in keyword_list]
    keywords_lower = [keyword.lower() for keyword in keyword_list]

    keyword_list.extend(keywords_cap)
    keyword_list.extend(keywords_upper)
    keyword_list.extend(keywords_lower)

    # Remove duplicates
    keyword_list_cleaned = list(set(keyword_list))
    print(keyword_list_cleaned)
    return keyword_list_cleaned


def _keyword_list_levenshtein(browser, keyword_list):
    """
    Add every keyword with Capital first letters and with all Capitals to keyword list
    """

    # Switch to first tab
    browser.switch_to.window(browser.window_handles[0])

    # Init parameters for ErrorHandling
    timeout = 6

    # Try to get body text
    try:
        element_present = EC.presence_of_element_located((By.TAG_NAME, "body"))
        WebDriverWait(browser, timeout).until(element_present)
    except TimeoutException:
        enh.print_rd("Timed out waiting for page to load")

    try:
        # Get entire text on current page
        text_browser = browser.find_element_by_tag_name("body").get_attribute(
            "innerText"
        )
    except Exception as e:
        text_browser = []
        enh.print_rd(e)

    # Initialize new list
    keyword_list_levensthein = keyword_list.copy()

    # Compare every item in text_browser with keyword_list and calculate Levenshtein distance
    for textitem in text_browser.split():
        for keyword in keyword_list:
            # Select the right keywords
            if 4 < len(keyword) <= len(textitem) and distance(keyword, textitem) <= 2:
                keyword_list_levensthein.append(textitem)
            elif 2 < len(keyword) <= len(textitem) and distance(keyword, textitem) <= 1:
                keyword_list_levensthein.append(textitem)
            elif len(keyword) == 2:
                keyword_list_levensthein.append(keyword.capitalize())
                keyword_list_levensthein.append(keyword.upper())
                keyword_list_levensthein.append(keyword.lower())

    # Remove duplicates
    keyword_list_cleaned = list(set(keyword_list_levensthein))

    return keyword_list_cleaned


def highlight_static(browser, keyword_list_init, levenshtein_bool):
    """
    Highlights a Selenium Webdriver element in the first Tab of the given browser-window
    levenshtein = bool : morph keyword list with levenshtein distance or just upper-, lower-case variations
    """

    # Initial Bool for UserInput
    new_page = 1
    while new_page:
        # Remove dublicates
        keyword_list_init = list(set(keyword_list_init))

        # Morph the existing keyword list for more results
        if levenshtein_bool:
            keyword_list = _keyword_list_levenshtein(browser, keyword_list_init)
        else:
            keyword_list = _keyword_list_extension(keyword_list_init)

        # Search for elements with keywords
        element_list = _element_search(browser, keyword_list)
        # No elements found
        if not element_list:

            enh.print_rd("No elements found on this page...")

            new_page = int(
                enh.input_rd(
                    "New Page? (1) Add Keywords? (2) Remove Keyword? (3) Exit Function (0)"
                )
            )

        # Elements found
        else:
            # Get original style and highlight every element on page with keyword
            orig_style_list = []
            for element_item in element_list:
                orig_style_list.append(element_item.get_attribute("style"))
                _highlight_static_local(element_item)

            enh.print_rd(f"{len(element_list)} elements found on this page:")

            print("List of used keywords:")
            print(keyword_list)

            new_page = int(
                enh.input_rd(
                    "New Page? (1) Add Keywords? (2) Remove Keyword? (3) Exit Function (0)"
                )
            )

        # Add or remove Keywords
        if new_page == 2:
            append_bool = 1
            while append_bool:

                enh.print_rd("Current Keyword-List:", "above")
                print(keyword_list_init)

                user_add_input = enh.input_rd("Type in new keyword:")
                keyword_list_init.append(user_add_input)

                enh.print_rd(f'Successfully added: "{user_add_input}"')

                append_bool = int(input("Add another one? (1/0)"))
        elif new_page == 3:
            remove_bool = 1
            while remove_bool:

                enh.print_rd("Current Keyword-List:", "above")
                print(keyword_list_init)

                while 1:
                    try:
                        user_remove_input = enh.input_rd("Type in keyword to remove:")
                        keyword_list_init.remove(user_remove_input)
                    except Exception as e:
                        enh.print_rd(e)
                        continue

                    enh.print_rd(f'Successfully removed: "{user_remove_input}"')
                    break

                remove_bool = int(input("Remove another one? (1/0)"))

            # Restore original style in order to highlight again (without removed keywords)
            for orig_style_item, element_item in zip(orig_style_list, element_list):
                _apply_style(orig_style_item, element_item)
    # If user-input: terminate function

    enh.print_rd(f"{__name__} terminated.")

    # Return keyword list
    return keyword_list_init


def open_url_and_highlight(url, highlight_list, levenshtein_bool):
    """
    Opens given URL and highlights elements from highlight list
    Levenshtein bool = true: extends the highlight_list with similiar keywords
    """

    # Firefox as Browser, auto download GeckoDriver and set options
    browser = webdriver.Firefox(executable_path=GeckoDriverManager().install())

    browser.get(url)

    highlight_static(browser, highlight_list, levenshtein_bool=levenshtein_bool)

    return browser
