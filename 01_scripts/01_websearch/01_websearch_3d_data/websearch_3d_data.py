import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))

from webtools.webelement_highlight import highlight_static
import webtools.automatic_websearch as aw

# Generate Data Lists
systemlist = ["cad", "stl", "scan", "3D", "model", "digital"]
toothlist = ["dental", "tooth", "teeth", "jaw"]

# Put Search-Data in big list
keyword_list = []
for string1 in toothlist:
    for string2 in systemlist:
        keyword_list.append(string1 + " " + string2)

# Search keyword for searchengine_list Creation, will be replaced by searchitems from searchlist
default_keyword = "teeth"

# List with searchengines after search with default_keyword
searchengine_list = [
    "https://www.kaggle.com/search?q=teeth",
    "https://datasetsearch.research.google.com/search?query=teeth&docid=L2cvMTFqY2txOGdkeQ%3D%3D",
    "https://www.openml.org/search?q=teeth&type=data",
    "https://www.google.com/search?q=teeth+site%3Aarchive.ics.uci.edu%2Fml&sa=Search&cof=AH%3Acenter%3BLH%3A130%3BL%3Ahttp%3A%2F%2Farchive.ics.uci.edu%2Fml%2Fassets%2Flogo.gif%3BLW%3A384%3BAWFID%3A869c0b2eaa8d518e%3B&domains=ics.uci.edu&sitesearch=ics.uci.edu",
    "https://www.nature.com/search?q=teeth&journal=",
    "https://github.com/awesomedata/awesome-public-datasets/search?q=teeth",
]

# Results csv Filename
csv_name = (__file__ + "_results").replace(".py", "")

# Create highlight list with all keywords
highlight_list = [
    "deep",
    "learning",
    "voxel",
    "artificial",
    "intelligence",
    "generative",
    "neural",
    "network",
]
highlight_list.extend(systemlist)
highlight_list.extend(toothlist)

# Automatic Websearch
browser = aw.automatic_websearch(
    keyword_list, searchengine_list, default_keyword, csv_name
)

# Highlight Keywords
highlight_list_ext = highlight_static(browser, highlight_list, levenshtein_bool=True)
print(highlight_list_ext)


aw.quit_gecko(browser)
