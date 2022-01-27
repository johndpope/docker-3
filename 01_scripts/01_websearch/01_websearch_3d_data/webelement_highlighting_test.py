import selenium.webdriver as webdriver
from webdriver_manager.firefox import GeckoDriverManager
from selenium.webdriver.firefox.options import Options
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))

from webtools.webelement_highlight import highlight_static

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

# Call random URL
browser.get("https://www.browserstack.com/")

# List of Keywords to search for
keyword_list = ["Sign in", "Pricing"]

# Highlight Keywords
highlight_static(browser, keyword_list)
# highlight_static(browser, string_list)

# Quit gecko on user-input
quit_gecko = 0
while not quit_gecko:
    quit_gecko = int(input("Quit gecko? (1/0)"))

browser.quit()
