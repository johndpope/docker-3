import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import time
import copy

# Visualize all data
visu = False

# File and pathnames
filename = "1917_2_2v.stl"
save_filename = filename.replace(".stl", "_open3d_bearbeitet.stl")
load_pathname = "G:\\ukr_data\\Einzelzaehne_unsorted\\1917_2\\"
save_pathname = "G:\\ukr_data\\bearbeitet\\"

# Import Mesh (stl)
mesh = o3d.io.read_triangle_mesh(load_pathname + filename)
mesh.compute_vertex_normals()

# Visualize the original mesh
if visu:
    o3d.visualization.draw_geometries([mesh])
print(f"Original Mesh: {mesh}")

# Create new mesh
mesh_zcrop = copy.deepcopy(mesh)

# Get Triangles, normals and vertices as np-arrays
mesh_zcrop_tri_arr = np.asarray(mesh_zcrop.triangles)
mesh_zcrop_tri_nor_arr = np.asarray(mesh_zcrop.triangle_normals)
mesh_zcrop_vert_arr = np.asarray(mesh_zcrop.vertices)

# Crop data: only keep trianlges with normals that are pointing up (z>0)
mesh_zcrop.triangles = o3d.utility.Vector3iVector(
    mesh_zcrop_tri_arr[mesh_zcrop_tri_nor_arr[:, 2] > 0]
)
mesh_zcrop.triangle_normals = o3d.utility.Vector3dVector(
    mesh_zcrop_tri_nor_arr[mesh_zcrop_tri_nor_arr[:, 2] > 0]
)

print(f"New cropped Mesh: {mesh_zcrop}")
# Visualize the new mesh
# o3d.visualization.draw_geometries([mesh_zcrop])

# Create PointCloud from cropped mesh
# Number of Points in pointcloud
numpoints = 10000000
pcd = mesh_zcrop.sample_points_uniformly(number_of_points=numpoints)
print(f"Pointcloud from croppped Mesh: {pcd}")

# Convert Open3D.o3d.geometry.PointCloud to numpy array and get boundaries
xyz_load = np.asarray(pcd.points)
z_max = xyz_load[:, 2].max()
z_min = xyz_load[:, 2].min()

# Boundaries
print(f"Max Bound: {pcd.get_max_bound()}")
print(f"Min Bound: {pcd.get_min_bound()}")

# Print boundaries
print(f"zmax: {z_max}")
print(f"zmin: {z_min}")
print(f"xmin: {xyz_load[:, 0].min()}")
print(f"xmax: {xyz_load[:, 0].max()}")
print(f"ymin: {xyz_load[:, 1].min()}")
print(f"ymax: {xyz_load[:, 1].max()}")

# Declare relative z threashold
z_threshold_rel = (z_max - z_min) / 2

# Pass xyz to Open3D.o3d.geometry.PointCloud
pcd1 = o3d.geometry.PointCloud()

# Only keep points with z values bigger than z_max - z_threshold_rel
pcd1.points = o3d.utility.Vector3dVector(
    xyz_load[xyz_load[:, 2] > (z_max - z_threshold_rel)]
)
print(f"PCD after z-crop: {pcd1}")

# Visualize PointCloud
if visu:
    o3d.visualization.draw_geometries([pcd1])

# Function Inlier, Outlier
def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])


# Downsample the pointcloud with voxel grid // downsample to 1 point/voxel
voxelsize = 0.05
pcd1_vox_downsample = pcd1.voxel_down_sample(voxel_size=voxelsize)
print(
    f"Downsample the point cloud with a voxelsize of {voxelsize}: {pcd1_vox_downsample}"
)
if visu:
    o3d.visualization.draw_geometries([pcd1_vox_downsample])

# Statistical oulier removal
cl, ind = pcd1_vox_downsample.remove_statistical_outlier(nb_neighbors=100, std_ratio=10)

if visu:
    display_inlier_outlier(pcd1, ind)

# New pcd without outliers
pcd1_inlier = pcd1_vox_downsample.select_by_index(ind)


o3d.visualization.draw_geometries([pcd1_inlier])

print(f"{len(pcd1_vox_downsample.points) - len(pcd1_inlier.points)} Points removed.")
print(f"After statistical outlier removal: {pcd1_inlier}")

# Create voxelgrid from downsampled pcd // pcd points should already be centerpoints of voxelgrid
voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
    pcd1_inlier, voxel_size=voxelsize
)

if visu:
    o3d.visualization.draw_geometries([voxel_grid])

print(f"Voxel Grid with {voxelsize = }: {voxel_grid}")


# Compute ISS Keypoints for downsampled pcd
tic = time.time()
keypoints = o3d.geometry.keypoint.compute_iss_keypoints(pcd1_inlier)
toc = 1000 * (time.time() - tic)
print("ISS Computation took {:.0f} [ms]".format(toc))

# Paint keypoints and pcd in different colours
keypoints.paint_uniform_color([1.0, 0.75, 0.0])
pcd1_vox_downsample.paint_uniform_color([0.5, 0.5, 0.5])

if visu:
    o3d.visualization.draw_geometries([keypoints, pcd1_vox_downsample])
