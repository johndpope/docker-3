# Import Necessary packages
import os
from webdriver_manager.firefox import GeckoDriverManager
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
import time
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
import webtools.output_readability_enhancer as enh
from webtools.automatic_websearch import quit_gecko


def automatic_springer_download(start_page, download_dir):
    """
    Goes through every page and downloads all pdfs for given Springer-Search-URL

    Save to download_dir
    """

    # Generate new options File
    options = Options()
    # Avoid PopUps
    options.set_preference("dom.disable_open_during_load", False)
    # Set max Tabs to inf
    options.set_preference("dom.popup_maximum", -1)
    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.manager.showWhenStarting", False)
    options.set_preference("browser.download.dir", download_dir)
    options.set_preference("browser.helperApps.alwaysAsk.force", False)
    options.set_preference("plugin.scan.plid.all", False)
    options.set_preference("plugin.scan.Acrobat", "99.0")
    options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")
    options.set_preference("pdfjs.disabled", True)

    # Configure Firefox with options
    browser = webdriver.Firefox(
        executable_path=GeckoDriverManager().install(), options=options
    )

    # Open the start_page
    browser.get(start_page)

    # Init parameters for ErrorHandling
    timeout = 5
    tries = 4

    # Uncheck "Include preview-only content" checkbox if checked
    try:
        if browser.find_element_by_id("results-only-access-checkbox").get_attribute(
            "checked"
        ):
            browser.find_element_by_id("results-only-access-checkbox").click()
    except Exception as e:
        print(e)

    # Init list for all element hrefs
    element_hrefs = []

    # Get all hrefs for all books on all pages
    for ctr in range(int(browser.find_element_by_class_name("number-of-pages").text)):

        # Search for every URl with "book" in it
        for element_item in browser.find_elements_by_class_name("title"):
            try:
                if element_item.get_attribute("href").find("/book/") != (-1):
                    element_hrefs.append(element_item.get_attribute("href"))
            finally:
                continue

        # Go to next page
        for i in range(tries):
            try:
                browser.find_element_by_class_name("next").click()
            except Exception as e:
                if i < tries - 1:  # i is zero indexed
                    # Wait
                    time.sleep(1)
                    # Try to click away the cookies pop up
                    try:
                        browser.find_element_by_xpath("//button[@class='cc-button cc-button--contrast cc-banner__button cc-banner__button-accept']").click()
                    finally:
                        continue
                else:
                    enh.print_rd(e)

    # Print the number of books
    enh.print_rd(f"{len(element_hrefs)} books found.")

    # Init Name List for all book names
    name_list = []

    # Open every href and click on download
    for ctr, element_href in enumerate(element_hrefs):

        print(element_href)
        # Open Book href
        browser.get(element_href)
        # Wait
        time.sleep(1)

        # Check for Download Button Presence
        try:
            element_present = EC.presence_of_element_located(
                (By.PARTIAL_LINK_TEXT, "Download book PDF")
            )
            WebDriverWait(browser, timeout).until(element_present)
        except TimeoutException:
            enh.print_rd("Timed out waiting for page to load")

        # Try to click on Download button
        for i in range(tries):
            try:
                browser.find_element_by_partial_link_text("Download book PDF").click()
            except Exception as e:
                if i < tries - 1:  # i is zero indexed
                    # Wait
                    time.sleep(1)
                    # Try to click away the cookies pop up
                    try:
                        browser.find_element_by_xpath("//button[@class='cc-button cc-button--contrast cc-banner__button cc-banner__button-accept']").click()
                    finally:
                        continue
                else:
                    enh.print_rd(e)
            else:
                # Append Name to list if successful
                name_list.append(browser.find_element_by_class_name("page-title").text)
            break

    # Write Name List to file
    with open(os.path.join(download_dir, "0_book_titles.txt") , "w") as f:
        for row in name_list:
            f.write(row.replace("\n", "\n\t") + "\n")

    # Quit Gecko
    quit_gecko(browser)


def automatic_pearson_book_download(start_page, savepath):
    """
 
    """
    os.makedirs(savepath, exist_ok=True)

    # Generate new options File
    options = Options()
    # Avoid PopUps
    options.set_preference("dom.disable_open_during_load", False)
    # Set max Tabs to inf
    options.set_preference("dom.popup_maximum", -1)
    options.set_preference("browser.download.folderList", 2)
    options.set_preference("browser.download.manager.showWhenStarting", False)
    options.set_preference("browser.download.dir", savepath)
    options.set_preference("browser.helperApps.alwaysAsk.force", False)
    options.set_preference("plugin.scan.plid.all", False)
    options.set_preference("plugin.scan.Acrobat", "99.0")
    options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")
    options.set_preference("pdfjs.disabled", True)

    # Configure Firefox with options
    browser = webdriver.Firefox(
        executable_path=GeckoDriverManager().install(), options=options
    )

    # Open the start_page
    browser.get(start_page)
    classnames = browser.find_elements_by_class_name("list-group-item")
    print(classnames[:20])

    classname_disp_list = []
    for classname in classnames:
        if classname.is_displayed() and "list-group-item d-flex has-sub-list" in classname.get_attribute("class"):
            classname_disp_list.append(classname)
    
    for classname in classname_disp_list:
        classname.click()
    
    # buttons = browser.find_elements_by_class_name('btn')
    # button_disp_list = []
    # for button in buttons:
    #     if button.is_displayed():
    #         button_disp_list.append(button)

    # for button in button_disp_list:
    #     button.click()
    #     browser.switch_to.window(browser.window_handles[0])



    





 
