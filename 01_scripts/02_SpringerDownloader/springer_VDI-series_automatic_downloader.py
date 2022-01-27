import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules"))

from webtools.automatic_springer_download import automatic_springer_download

automatic_springer_download(
    "https://link.springer.com/search?facet-series=%223482%22&facet-content-type=%22Book%22&showAll=true"
)
