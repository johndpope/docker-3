from multiprocessing import Value
import numpy as np
from scipy import spatial
import glob
import os
import matplotlib.pyplot as plt
import PIL
from hashlib import sha256
from tqdm import tqdm
import json
import copy

import open3d as o3d


plot_bool = True

grid_size = [256, 256]
z_threshold = 4
normbounds = [0, 1]
frame_size = 0.1
nan_val = 15
conversion_type = "abs"
invertY =  False 
keep_xy_ratio = False 
numpoints = 100000
rotation_deg_xyz = None

# Directories
stl_dir = r"G:\ukr_data\Einzelzaehne_sorted"
pcd_dir = rf"W:\ukr_data\Einzelzaehne_sorted\grid"
cfg_dir = rf"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\cfg"
img_dir_base = rf"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images"

# max_exp_cfg_path = glob.glob(os.path.join(stl_dir, f"*{numpoints}*"))

# if max_exp_cfg_path[0].split(".")[-1] == "npz":
#     expansion_max = np.load(max_exp_cfg_path[0])["expansion_max"]
# elif max_exp_cfg_path[0].split(".")[-1] == "json":
#     with open(max_exp_cfg_path[0]) as f:
#         expansion_max = np.array(json.load(f)["expansion_max"])



files = glob.glob(os.path.join(stl_dir, "*.stl"))
filepath_stl = files[0]

conversion_types = ["abs", "rel"]
if conversion_type not in conversion_types:
    raise ValueError(f"conversion type must be in {conversion_types}")

# Import Mesh (stl) and Create PointCloud from cropped mesh
pcd = o3d.io.read_triangle_mesh(filepath_stl)

# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R) :
    import math
    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])

## get bounding volume
aabb = pcd.get_axis_aligned_bounding_box()
aabb.color = (1, 0, 0)
obb = pcd.get_oriented_bounding_box()
# print(obb.R)
# print(obb.R.T)
# print(obb.center)
print(f"{obb.extent = }")
obb.color = (0, 1, 0)
# rot_mat = np.round(obb.R.T, decimals=3)
rot_mat = obb.R.T
print(f"{rot_mat = }")
euler_rot_rad = rotationMatrixToEulerAngles(rot_mat)%np.pi -np.pi
euler_rot_deg = euler_rot_rad*180/np.pi
print(f"{euler_rot_deg = }")
print(f"{np.diag(rot_mat) = }")

R = pcd.get_rotation_matrix_from_xyz((np.asarray(euler_rot_rad)))

print(f"{R = }")
pcd = pcd.rotate(rot_mat, center=obb.center)

# if any(np.diag(rot_mat) < 0):
#     y_rot = np.array( ([-1, 0, 0], [0, 1, 0], [0, 0, -1]) )
#     pcd = pcd.rotate(y_rot, center=obb.center)

obb2 = pcd.get_oriented_bounding_box()
# print(obb.R)
# print(obb.R.T)
# print(obb.center)
print(f"{obb.extent = }")
obb2.color = (0, 0, 1)

pcd = pcd.sample_points_uniformly(
            number_of_points=numpoints)

while 1: 
    geometries = [pcd, obb,obb2, aabb]

    # Visualize PointCloud
    if plot_bool:

        viewer = o3d.visualization.Visualizer()
        viewer.create_window()
        for geometry in geometries:
            viewer.add_geometry(geometry)

        opt = viewer.get_render_option()
        opt.show_coordinate_frame = True
        opt.background_color = np.asarray([1, 1, 1])
        viewer.run()

    direc = input("Specify 180 deg rotation axis (x,y,z) or 0 for exit:\n")
    viewer.destroy_window()

    if direc ==  "0":
        print("Finished.")
        break
    elif direc == "x":
        R = pcd.get_rotation_matrix_from_xyz([np.pi,0,0])
    elif direc == "y":
        R = pcd.get_rotation_matrix_from_xyz([0, np.pi,0])
    elif direc == "z":  
        R = pcd.get_rotation_matrix_from_xyz([0, 0, np.pi])

    print(f"Rotating 180 deg around: {direc}-axis..")

    pcd = pcd.rotate(R, center=obb.center)
    obb2 = pcd.get_oriented_bounding_box()
    obb2.color = (0, 0, 1)
    
# # Execute transformations if specified
# if rotation_deg_xyz is not None:
#     rotation_deg_xyz = np.asarray(rotation_deg_xyz)
#     pcd.rotate(
#         pcd.get_rotation_matrix_from_xyz((rotation_deg_xyz / 180 * np.pi)))

# # Convert Open3D.o3d.geometry.PointCloud to numpy array and get boundaries
# pcd_arr = np.asarray(pcd.points)

# # Crop the pcd_arr
# pcd_arr = pcd_arr[pcd_arr[:,
#                             2] > (np.max(pcd_arr, axis=0)[2] - z_threshold)]

# # Max Expansion after crop
# pcd_expansion_max = np.max(pcd_arr, axis=0) - np.min(pcd_arr, axis=0)
# print(f"{pcd_expansion_max = }")

# # Swap x and y values if max(x)<max(y)
# if rotation_deg_xyz is None and (pcd_expansion_max[0] <
#                                     pcd_expansion_max[1]):
#     pcd_expansion_max[:2] = pcd_expansion_max[[1, 0]]
#     pcd_arr[:, :2] = pcd_arr[:, [1, 0]]
#     # print("Axis swapped!")

# # Normalize the pointcloud to min(pcd) = zeros
# pcd_arr = pcd_arr - np.min(pcd_arr, axis=0)

# if conversion_type == "abs":
#     # convert to arr, json cant store arrays
#     expansion_max = np.array(expansion_max)
# elif conversion_type == "rel":
#     expansion_max[:2] = pcd_expansion_max.max() * np.ones(shape=(1,2))

# # Rigid Body transformation -> put the body in the middle of the xy meshgrid
# pcd_arr[:, :2] += (expansion_max[:2] - pcd_expansion_max[:2]) / 2

# # Create frame around tooth
# pcd_arr[:, :2] += frame_size  # min(pcd) = ones*frame_size now
# expansion_max[:2] += 2 * frame_size

# ## get bounding volume
# pcd_bound = o3d.geometry.PointCloud()
# pcd_bound.points = o3d.utility.Vector3dVector(pcd_arr)
# aabb = pcd_bound.get_axis_aligned_bounding_box()
# aabb.color = (1, 0, 0)
# obb = pcd_bound.get_oriented_bounding_box()
# obb.color = (0, 1, 0)
# euler_rot_bound = rotationMatrixToEulerAngles(obb.R)*180/np.pi
# print(f"{euler_rot_bound = }")
# # Create Vectors for mesh creation
# x_vec = np.linspace(0, expansion_max[0], grid_size[0])
# y_vec = np.linspace(0, expansion_max[1], grid_size[1])

# if invertY:
#     # Invert the y_vec -> the cartesian point (0,0,0) will be at img[-1,0]: lower left corner
#     y_vec = y_vec[::-1]

# # Create meshgrid
# [X, Y] = np.meshgrid(x_vec, y_vec)

# # Create points from meshgrid
# points = np.c_[X.ravel(), Y.ravel()]

# # Create tree with pcd points
# tree = spatial.cKDTree(pcd_arr[:, :2])

# # initialize zvals array
# zvals = np.zeros(np.shape(points[:, 0]))

# # Define ball radies as the max distance between 2 point neighbors in the current pcd
# pcd.points = o3d.utility.Vector3dVector(pcd_arr)
# query_ball_radius = np.max(pcd.compute_nearest_neighbor_distance())

# # Go through mesh grid points and search for nearest points with ballradius
# for ctr, results in enumerate(
#         tree.query_ball_point(points, query_ball_radius)):
#     if results:
#         # Save max z value from results if results is not empty
#         zvals[ctr] = pcd_arr[results, 2].max()
#     else:
#         # Save nan_val if results = empty
#         zvals[ctr] = nan_val

# # Concatenate the xy meshgrid and the zvals from loop
# new_pcd_arr = np.concatenate(
#     [points, zvals[:, np.newaxis]], axis=1)

# # Generate empty Open3D.o3d.geometry.PointCloud
# new_pcd = o3d.geometry.PointCloud()
# new_pcd.points = o3d.utility.Vector3dVector(new_pcd_arr)

# # # Print Relation
# # print(f"zvals_nan/zvals = {zvals[zvals == nan_val].shape[0] / zvals.shape[0]}")

# # o3d.visualization.draw_geometries([chair, aabb, obb],
# #                                   zoom=0.7,
# #                                   front=[0.5439, -0.2333, -0.8060],
# #                                   lookat=[2.4615, 2.1331, 1.338],
# #                                   up=[-0.1781, -0.9708, 0.1608])






# #     viewer.destroy_window()

# #     o3d.visualization.draw_geometries([new_pcd])

# # # Save the new pcd
# # if save_path_pcd:
# #     os.makedirs(os.path.dirname(save_path_pcd), exist_ok=True)
# #     o3d.io.write_point_cloud(save_path_pcd, new_pcd)        
# # if save_path_npy:
# #     os.makedirs(os.path.dirname(save_path_npy), exist_ok=True)
# #     np.save(save_path_npy, new_pcd_arr)
