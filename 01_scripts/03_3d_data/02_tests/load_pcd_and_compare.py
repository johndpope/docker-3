import open3d as o3d
import glob
import os


pcd_dir = r"W:\ukr_data\Einzelzaehne_sorted\grid\pcd-360f394-abs-keepRatioXY-invertY-rot_3d-full\256x256\pcd_grid"
pcd_paths = glob.glob(os.path.join(pcd_dir, "*.pcd"))

path1 = pcd_paths[0]
path2 = pcd_paths[-1]

pcd1 = o3d.io.read_point_cloud(path1)
pcd2 = o3d.io.read_point_cloud(path2)

visu_list = [pcd1, pcd2]

o3d.visualization.draw_geometries(visu_list, width=720, height=720, window_name=f"Rotated Tooth", left=1000, top=300)