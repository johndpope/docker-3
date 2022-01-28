import numpy as np
import open3d as o3d
import glob
import os

stl_path = r"G:\ukr_data\Einzelzaehne_sorted"
stl_files = glob.glob(os.path.join(stl_path, "*.stl"))

pcd = o3d.io.read_triangle_mesh(stl_files[0]).sample_points_uniformly(
    number_of_points=10000
)

pcd_arr = np.asarray(pcd.points)

print(pcd_arr.shape)

np.save(os.path.join(r"G:\docker", "pcd_arr.npy"), pcd_arr)
o3d.io.write_point_cloud(os.path.join(r"G:\docker", "pcd.xyz"), pcd)

# mystr = "hallo"
# print(mystr)

# print(mystr[::-1])


# arr = np.arange(0, 9).reshape((3, 3))
# print(arr)

# sum_arr = arr.sum()
# print(sum_arr)

# suma = 9
# myarr = [1, 3, 5, 6, 11, 23]
# flag = False
# for num1 in myarr:
#     if flag:
#         break
#     for num2 in myarr:
#         if num1 != num2:
#             calc_sum = num1 + num2
#             if suma - calc_sum == 0:
#                 flag = True
#                 print(f"finish: {num1} + {num2} = {suma}")
#                 break
