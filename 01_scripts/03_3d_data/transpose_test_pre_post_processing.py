import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import PIL

numpoints = 256*256*10
filenum = 13

filename_pcd = rf"G:\ukr_data\Einzelzaehne_sorted\grid\7a9a625\256x256\pcd_grid\einzelzahn_grid_{filenum}_7a9a625.pcd"
filename_stl = rf"G:\ukr_data\Einzelzaehne_sorted\einzelzahn_{filenum}.stl"
filename_png = rf"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\7a9a625\grayscale\256x256\img\img_{filenum}_7a9a625.png"

img = PIL.Image.open(filename_png)
img.show()


stl_file  = o3d.io
stl_pcd = o3d.io.read_triangle_mesh(filename_stl).sample_points_uniformly(
            number_of_points=numpoints)

grid_pcd = o3d.io.read_point_cloud(filename_pcd)

geometries = [grid_pcd, stl_pcd]

viewer = o3d.visualization.Visualizer()
viewer.create_window()
for geometry in geometries:
    viewer.add_geometry(geometry)
opt = viewer.get_render_option()
opt.show_coordinate_frame = True
opt.background_color = np.asarray([0.5, 0.5, 0.5])
viewer.run()
viewer.destroy_window()

# o3d.visualization.draw_geometries([stl_pcd, grid_pcd])

# plt.imshow(myarr)
# plt.colorbar()
# plt.show()
# print(myarr)
# print(myarr.reshape(-1,1))