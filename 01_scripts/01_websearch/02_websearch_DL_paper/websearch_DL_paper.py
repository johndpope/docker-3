import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))

from webtools.webelement_highlight import highlight_static
from webtools.automatic_websearch import automatic_websearch

# # Generate Data Lists
# systemlist = ["3D", "deep learning", "deep", "generative", "GAN", "autoencoder"]
# toothlist = ["dental", "tooth", "teeth"]

# Generate Data Lists
systemlist = ["3D", "deep learning"]
toothlist = ["dental"]

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
    "GAN",
    "autoencoder",
    "DCGAN",
]
highlight_list.extend(systemlist)
highlight_list.extend(toothlist)


# Put Search-Data in big list
keyword_list = []
for string1 in toothlist:
    for string2 in systemlist:
        keyword_list.append(string1 + " " + string2)

# Search keyword for searchengine_list Creation, will be replaced by searchitems from searchlist
default_keyword = "teeth"

# List with searchengines after search with default_keyword
searchengine_list = [
    "https://pubmed.ncbi.nlm.nih.gov/?term=teeth",
    "https://scholar.google.com/scholar?hl=de&as_sdt=0%2C5&q=teeth&btnG=",
]

# Results csv Filename
csv_path = os.path.abspath((__file__).replace(".py", "_results.csv"))


# Automatic Websearch
browser = automatic_websearch(
    keyword_list, searchengine_list, default_keyword, csv_path
)

# Highlight Keywords
highlight_list_ext = highlight_static(browser, highlight_list, levenshtein_bool=True)
print(highlight_list_ext)

# Quit gecko on user-input
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
