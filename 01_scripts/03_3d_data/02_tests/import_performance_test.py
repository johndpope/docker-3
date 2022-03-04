import os
import time
import sys

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))

t0 = time.time()
import pcd_tools.data_processing as dp
import img_tools.image_processing as ip
print(f"Time Fresh Import: {time.time()-t0}")

t1 = time.time()
import pcd_tools.data_processing as dp
import img_tools.image_processing as ip
print(f"Time already imported: {time.time()-t1}")

def import_fun():
    import pcd_tools.data_processing as dp
    import img_tools.image_processing as ip

t2 = time.time()
import_fun()
print(f"Time function import: {time.time()-t2}")
