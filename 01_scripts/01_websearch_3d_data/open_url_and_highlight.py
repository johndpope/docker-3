import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))

from webtools.webelement_highlight import open_url_and_highlight
import webtools.automatic_websearch as aw

# Generate Data Lists
systemlist = ["cad", "stl", "scan", "3D", "model", "digital"]
toothlist = ["dental", "tooth", "teeth", "jaw"]

# Create highlight list with all keywords
highlight_list = [
    "deep",
    "learning",
    "voxel",
    "artificial",
    "intelligence",
    "generative",
    "generator",
    "neural",
    "network",
]

highlight_list.extend(systemlist)
highlight_list.extend(toothlist)
url = "https://www.nature.com/search?q=dental+cad&page=2"

browser = open_url_and_highlight(url, highlight_list, levenshtein_bool=True)

aw.quit_gecko(browser)
