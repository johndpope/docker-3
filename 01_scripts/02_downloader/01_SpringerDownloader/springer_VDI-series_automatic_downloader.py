import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))

from webtools.automatic_download import automatic_springer_download

download_dir = r"C:\Users\bra45451\Downloads"
url = "https://link.springer.com/search?facet-series=%223482%22&facet-content-type=%22Book%22&showAll=true"

automatic_springer_download(url, download_dir)
