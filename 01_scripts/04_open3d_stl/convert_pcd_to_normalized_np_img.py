import open3d as o3d
import numpy as np
from scipy import spatial
import glob
import os
import matplotlib.pyplot as plt

foldername = "grid"

load_pathname = f"G:\\ukr_data\\Einzelzaehne_sorted\\{foldername}\\"
np_savepath = f"G:\\ukr_data\\Einzelzaehne_sorted\\{foldername}\\np_norm\\"

if not os.path.exists(np_savepath):
    os.mkdir(np_savepath)

np_savename = "einzelzaehne"
files = glob.glob(load_pathname + "*.pcd")
grid_size = (256, 256, 1)

train_images = np.zeros((len(files), 256, 256, 1)).astype("float32")
normvals = np.load(f"{load_pathname}normvals.npy")

# # old relative norm
# for num, filename in enumerate(files):
#     pcd = o3d.io.read_point_cloud(filename)
#     pcd_arr = np.asarray(pcd.points)
#     pcd_img = pcd_arr[:, 2].reshape(grid_size)
#     pcd_img[pcd_img != 10] = pcd_img[pcd_img != 10] + np.abs(
#         pcd_img[pcd_img != 10].min()
#     )

#     pcd_img[pcd_img != 10] = (
#         pcd_img[pcd_img != 10] - pcd_img[pcd_img != 10].max() / 2
#     ) / (pcd_img[pcd_img != 10].max() / 2)

#     pcd_img[pcd_img == 10] = -1
#     train_images[num, :, :, :] = pcd_img

for num, filename in enumerate(files):
    pcd = o3d.io.read_point_cloud(filename)
    pcd_arr = np.asarray(pcd.points)
    pcd_img = pcd_arr[:, 2].reshape(grid_size)
    pcd_img[pcd_img != 10] = pcd_img[pcd_img != 10] + np.abs(
        pcd_img[pcd_img != 10].min()
    )

    pcd_img[pcd_img != 10] = (
        pcd_img[pcd_img != 10] - pcd_img[pcd_img != 10].max() / 2
    ) / (pcd_img[pcd_img != 10].max() / 2)

    pcd_img[pcd_img == 10] = -1
    train_images[num, :, :, :] = pcd_img

np.save(np_savepath + np_savename, train_images)

plt.figure()

plt.rc("font", size=15)

plt.imshow(
    train_images[0, :, :, :],
    cmap="viridis",
)
plt.colorbar()
plt.show()
