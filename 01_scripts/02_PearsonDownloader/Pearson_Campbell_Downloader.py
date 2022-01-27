import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules"))

from webtools.automatic_download import automatic_pearson_book_download

automatic_pearson_book_download(
    "https://elibrary.pearson.de/book/99.150005/9783863268671", savepath=os.path.join(os.path.dirname(__file__), "download")
)
