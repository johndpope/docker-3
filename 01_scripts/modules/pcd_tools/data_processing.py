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
import json

# ------------------------------- #


## Math
# Checks if a matrix is a valid rotation matrix.
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


# Calculates rotation matrix to euler angles
# The result is the same as MATLAB except the order
# of the euler angles ( x and z are swapped ).
def rotationMatrixToEulerAngles(R):
    import math
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])



def get_param_hash_from_img_path(img_dir: str, cfg_search_dir: str = []):
    # Finds the matching param_hash in cfg_search_dir for the specified img_dir
    if cfg_search_dir:
        param_hashes = [cfg_file.split(".")[0].split("_")[-1] for cfg_file in glob.glob(os.path.join(cfg_search_dir, "pcd_to_grid_cfg*.npz"))]
        param_hash = [param_hash for param_hash in param_hashes if param_hash in img_dir][0]
    else:
        param_hash = img_dir.split(f"images-")[-1].split("-")[0]

    if param_hash:
        return param_hash
    else:
        raise ValueError(f"Could not find a matching param_hash for {img_dir}")


def image_conversion_L_RGB(source_img_dir, rgb_dir):
    """
    Load grayscale pngs and convert to rgb  

    Save files with same name to rgb_dir  
    
    img_dir     : directory with grayscale .png images  

    rgb_dir     : directory to save rgb images  
    """

    print('Loading images from "%s"' % source_img_dir)
    # Create the directory with all parents
    if not os.path.exists(rgb_dir):
        os.makedirs(rgb_dir, exist_ok=True)

    # Get all .png files
    image_filenames = sorted(glob.glob(os.path.join(source_img_dir, "*.png")))
    if len(glob.glob(os.path.join(rgb_dir, "*.png"))) >= len(image_filenames):
        print("Conversion aborted. RGB Images already exist.")
    else:
        # Convert every file to RGB and save in "rgb"
        for img_file, num in zip(
                image_filenames,
                tqdm(
                    iterable=range(len(image_filenames)),
                    desc="Converting to RGB..",
                    ascii=False,
                    ncols=100,
                ),
        ):

            PIL.Image.open(img_file).convert("RGB").save(
                os.path.join(rgb_dir, os.path.basename(img_file)))
        print("Done.")


def image_conversion_RGB_L(img: np.array,
                           conv_type: str = "luminance_int_round") -> np.array:
    """"
    Converts RGB img-array of shape (grid, grid, 3) or (3, grid, grid) to L img of shape (grid, grid)
    conv_type must be in ["luminance_float_exact", "luminance_int_round", "min", "max"]
    """

    type_list = ["luminance_float_exact", "luminance_int_round", "min", "max"]

    if conv_type not in type_list:
        raise ValueError(f'conv_type must be in {type_list}')

    # Check correct shape
    if img.shape[0] == 3:
        print("Transpose")
        img = img.transpose((1, 2, 0))

    if conv_type == "luminance_float_exact":
        img_L = (img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 +
                 img[:, :, 2] * 0.114)
    elif conv_type == "luminance_int_round":
        img_L = np.asarray(PIL.Image.fromarray(img, "RGB").convert("L"))
    elif conv_type == "min":
        img_L = img.min(axis=2)
    elif conv_type == "max":
        img_L = img.max(axis=2)

    return img_L



def points_to_pcd(points, color=None):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        pcd.paint_uniform_color((color))
    return pcd

def fit_plane_to_pcd(mat):
        """
        Returns xyz-array of optimal fit plane for mat as well as the Rotation matrix for the plane-frame

        A B C opt for AX + BY + CZ + 1 = 0
        Z = alpha*X + beta*Y + gamma
        Optimal fit for point cloud in mat
        """
        # Adapted from https://de.scribd.com/doc/31477970/Regressions-et-trajectoires-3D

        meanmat = mat.mean(axis=0)
        xm = meanmat[0]
        ym = meanmat[1]
        zm = meanmat[2]

        mat_new = mat-meanmat
        E = np.matmul(mat_new.T, mat_new)
        w, v = np.linalg.eig(E)

        A = np.empty(shape=(3))
        B = np.empty(shape=(3))
        C = np.empty(shape=(3))
        delta = np.empty(shape=(3))

        for ctr in range(3):
            Vi = v[:,ctr]
            # roh = - (xm*Vi[0] + ym*Vi[1] + zm*Vi[2])/(Vi[0]**2 + Vi[1]**2 + Vi[2]**2)
            # ui = -Vi[0]*roh
            # vi = -Vi[1]*roh
            # wi = -Vi[2]*roh
            # print(f"Test: {Vi[0]**2 + Vi[1]**2 + Vi[2]**2} must be 1") 
            ai = Vi[0]/Vi[2]
            bi = Vi[1]/Vi[2]
            denom = (ai*xm+bi*ym+zm)
            Ai = -ai/denom
            Bi = -bi/denom
            Ci = -1/denom

            delta_sum_part = 0
            for x,y,z in mat:
                delta_sum_part += (Ai*x + Bi*y +Ci*z +1)**2

            delta[ctr] = 1/(Ai**2+Bi**2+Ci**2)* delta_sum_part
            A[ctr] = Ai
            B[ctr] = Bi
            C[ctr] = Ci

        min_indx = np.argmin(delta)
        # print(f"{delta.min() = }")

        Aopt = A[min_indx]
        Bopt = B[min_indx]
        Copt = C[min_indx]

        alpha = -Aopt/Copt
        beta = -Bopt/Copt
        gamma = -1/Copt

        # # Calc all plane_mats
        # plane_mats = np.tile(mat, (len(delta),1,1))

        # for ctr in range(len(delta)):
        #     alpha = -A[ctr]/C[ctr]
        #     beta = -B[ctr]/C[ctr]
        #     gamma = -1/C[ctr]
        #     plane_mats[ctr,:,2] = mat[:,0]*alpha + mat[:,1]*beta + np.ones(shape=(mat.shape[0]))*gamma  

        # plane_mat = copy.deepcopy(mat)

        # Create Vectors for mesh creation
        x_vec = np.linspace(mat[:,0].min(), mat[:,0].max(), 50)
        y_vec = np.linspace(mat[:,1].min(), mat[:,1].max(), 50)
        # Create meshgrid
        [X, Y] = np.meshgrid(x_vec, y_vec)
        X = X.flatten()[:,np.newaxis]
        Y = Y.flatten()[:,np.newaxis]
        plane_mat = np.concatenate([X,Y,np.zeros(shape=X.shape)], axis=1)

        plane_mat[:,2] = plane_mat[:,0]*alpha + plane_mat[:,1]*beta + np.ones(shape=(plane_mat.shape[0]))*gamma

        ## Frame for Plane

        # Initialize Points for axis-vec creation
        point1 = np.array([0, 0, 0], dtype=np.float64)
        point2 = np.array([1, 0, 0], dtype=np.float64)
        point3 = np.array([0, 1, 0], dtype=np.float64)

        # Calculate the z-components of the 3 points
        point1[2] = point1[0]*alpha + point1[1]*beta + gamma
        point2[2] = point2[0]*alpha + point2[1]*beta + gamma
        point3[2] = point3[0]*alpha + point3[1]*beta + gamma

        # Define and calculate the 3 axis vectors
        xaxis = point2-point1
        yaxis = point3-point1
        zaxis = np.cross(xaxis, yaxis).reshape((1,3))
        # Normalize the vectors
        xaxis = xaxis / np.linalg.norm(xaxis)
        yaxis = yaxis / np.linalg.norm(yaxis)
        zaxis = zaxis / np.linalg.norm(zaxis)
        
        # Create rotation matrix using column mayer notation 
        # http://renderdan.blogspot.com/2006/05/rotation-matrix-from-axis-vectors.html
        rot_mat = np.concatenate([xaxis[np.newaxis, :], yaxis[np.newaxis, :], zaxis], axis = 0).T

        return plane_mat, rot_mat

def pcd_bounding_rot(pcd, rot_3d_mode, rotY_noask = True, full_visu = False, triangle_normal_z_threshold = 0.9):
    """
    Rotate provided 3D-model depending on bounding-box Rotation.

    rotY_noask: rotate around Y after bounding box rotation if True

    """
    import open3d as o3d

    # Rotate
    obb = pcd.get_oriented_bounding_box()
    obb.color = (0,1,0)
    # Calculate the inverse Rotation (inv(R)=R.T for rot matrices)
    rot_mat = obb.R.T

    if full_visu:
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=obb.center)
        visu_list = [pcd.sample_points_uniformly(number_of_points=1000000), frame]
        o3d.visualization.draw_geometries(visu_list, width=720, height=720, window_name=f"Original", left=1000, top=300)

    # Choose rotation mode
    if rot_3d_mode == "z":
        print("STL-Z-Rot")
        euler_rot_rad = rotationMatrixToEulerAngles(rot_mat)
        R = pcd.get_rotation_matrix_from_xyz((np.asarray([0,0,euler_rot_rad[-1]])))
        pcd = pcd.rotate(R, center=obb.center)
    elif rot_3d_mode == "full" or rot_3d_mode == "bb":
        print("STL-BB-Rot")
        pcd.rotate(rot_mat, center=obb.center)

        # Define x,y,z rotation matrices for user input post processing
        x_rot = np.array( ([1, 0, 0], [0, -1, 0], [0, 0, -1]) )[np.newaxis, :,:]
        y_rot = np.array( ([-1, 0, 0], [0, 1, 0], [0, 0, -1]) )[np.newaxis, :,:]
        z_rot = np.array( ([-1, 0, 0], [0, -1, 0], [0, 0, 1]) )[np.newaxis, :,:]
        # Put all matrices in one array for later use
        rots_xyz = np.concatenate([x_rot, y_rot, z_rot], axis=0)

        # Rotates around y without asking
        if rotY_noask:
            pcd.rotate(rots_xyz[1], center=obb.center)

        # Create frame for visu
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=obb.center)

        while 1:
            # Get oriented and aligned bounding boxes
            # if rot_3d_mode == "full" they have to match exactly obb=aabb
            obb2 = pcd.get_oriented_bounding_box()
            obb2.color = (0,0,1)
            aabb = pcd.get_axis_aligned_bounding_box()
            aabb.color = (0, 0, 1)
            # Visualize 
            o3d.visualization.draw_geometries([pcd.sample_points_uniformly(number_of_points=1000000), frame, aabb, obb, obb2], width=720, height=720, window_name=f"Rotated Tooth", left=1000, top=300)
            
            # Get user input for post-processing
            rot_axis = input("\nTurn 180 degress around axis: \n0: x-red \n1: y-green\n2: z-blue \n3: Show Visu again.\n9: Finish\nUser-Input: ")
            
            # Process user input
            if rot_axis in ["0","1","2"]:
                rot = rots_xyz[int(rot_axis)]
                pcd.rotate(rot, center=obb.center)
            elif rot_axis == "3":
                continue
            elif rot_axis == "9":
                break
            else:
                print("Please insert from list [0, 1, 2, 3, 9].")

        if rot_3d_mode == "full":
            print("STL-Plane-Rot")
            # Get triangles and triangle-normals from stl
            pcd_new = copy.deepcopy(pcd)
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=obb.center)
            triangles = np.asarray(pcd_new.triangles)
            normals = np.asarray(pcd_new.triangle_normals)

            # Create bool-matrix for criteria : only z components bigger than triangle_normal_z_threshold
            criteria = normals[:,2]>triangle_normal_z_threshold

            # Overwrite old values with new cropped pcd
            pcd_new.triangles = o3d.utility.Vector3iVector(triangles[criteria])
            pcd_new.triangle_normals = o3d.utility.Vector3dVector(normals[criteria])
            pcd_new = pcd_new.sample_points_uniformly(number_of_points=1000)

            # Calculate best fit plane for pcd_new points with principal component analysis
            plane_mat, rot_mat_plane = fit_plane_to_pcd(mat=np.asarray(pcd_new.points))

            # Create pcd object from point-array
            plane_pcd = points_to_pcd(points=plane_mat, color = [0,0,1])
            
            if full_visu:
                visu_list = [pcd_new,  frame]
                o3d.visualization.draw_geometries(visu_list, width=720, height=720, window_name=f"Only Cropped", left=1000, top=300)
                visu_list.append(plane_pcd)
                o3d.visualization.draw_geometries(visu_list, width=720, height=720, window_name=f"Cropped + Plane", left=1000, top=300)

                visu_list = [pcd.sample_points_uniformly(number_of_points=1000000),  frame]
                o3d.visualization.draw_geometries(visu_list, width=720, height=720, window_name=f"Rotated Tooth BB", left=1000, top=300)
                visu_list.append(plane_pcd)
                o3d.visualization.draw_geometries(visu_list, width=720, height=720, window_name=f"Rotated Tooth BB + Plane before PlaneRot", left=1000, top=300)

            # Rotate with inverse Orientation of the calculated plane
            pcd = pcd.rotate(rot_mat_plane.T, center=obb.center)

            visu_list = [pcd.sample_points_uniformly(number_of_points=1000000), frame]
            o3d.visualization.draw_geometries(visu_list, width=720, height=720, window_name=f"Rotated Tooth after PlaneRot", left=1000, top=300)

            if full_visu:
                # Calculate plane_mat again for the rotated pcd
                # must be parallel to xy plane now
                plane_mat_rot, _ = fit_plane_to_pcd(mat=np.asarray(pcd_new.rotate(rot_mat_plane.T, center=obb.center).points))
                visu_list.append(points_to_pcd(points=plane_mat_rot, color=[0,1,0]))

                o3d.visualization.draw_geometries(visu_list, width=720, height=720, window_name=f"Rotated Tooth after PlaneRot with Plane", left=1000, top=300)

    return pcd



# ------------------------------- #

## Processing

class DataCreatorParams:

    @classmethod
    def __init__(cls, z_threshold, normbounds, frame_size, nan_val,
                 stl_dir, pcd_dir, cfg_dir, img_dir_base, 
                 conversion_type=None, invertY=None, keep_xy_ratio=None, rot_3d=None, rot_2d=None, rot_3d_mode=None, 
                 rot_2d_center=None, rot_2d_mode=None, rot_2d_show_img=None, reduced_data_set=None, reduce_num=None):

        cls.z_threshold = z_threshold
        cls.normbounds = normbounds
        cls.frame_size = frame_size
        cls.nan_val = nan_val
        cls.conversion_type = conversion_type
        cls.invertY = invertY
        cls.keep_xy_ratio = keep_xy_ratio
        cls.rot_3d = rot_3d
        cls.rot_3d_mode = rot_3d_mode if rot_3d or rot_3d is None else None
        cls.rot_2d = rot_2d
        cls.rot_2d_mode = rot_2d_mode if rot_2d or rot_2d is None else None
        cls.rot_2d_show_img = rot_2d_show_img if rot_2d_show_img is not None else True 
        cls.rot_2d_center = rot_2d_center if rot_2d or rot_2d is None else None
        cls.stl_dir = stl_dir
        cls.pcd_dir = pcd_dir
        cls.cfg_dir = cfg_dir
        cls.img_dir_base = img_dir_base
        cls.reduced_data_set=reduced_data_set if reduced_data_set is not None else False
        cls.reduce_num=reduce_num if reduce_num is not None else 0

        


class DatasetCreator(DataCreatorParams):

    def __init__(self, grid_size, rotation_deg_xyz = None):
        self.grid_size = grid_size
        self.rotation_deg_xyz = rotation_deg_xyz


    @property
    def numpoints(self): 
        return self.grid_size[0] * self.grid_size[1] * 10

    @property
    def num_stl(self): 
        return len(glob.glob(os.path.join(self.stl_dir, "*.stl")))

    @property
    def filepaths_stl(self):
        return sorted(glob.glob(os.path.join(self.stl_dir, "*.stl")))

    @property
    def num_images(self):
        if self.reduced_data_set:
            return self.num_stl-self.reduce_num
        else:
            return self.num_stl

    def prepare_stl(self, stl_new_dir = None):
        """
        Loads .stl files from directory, rotates them depending on their bounding-box orientation and saves the new files to stl_new_dir.

        stl_dir:        Directory with .stl files

        stl_new_dir:    New Directory for rotated .stl files

        rot_3d_mode:   "full": all axis rotation with user-input,  "z": only z axis

        """
        import open3d as o3d

        # Check function input
        rot_3d_modes = ["full", "z", "bb"]
        if not self.rot_3d_mode in rot_3d_modes:
            raise ValueError(f"rot_3d_mode must be in {rot_3d_modes}")

        if stl_new_dir is None:
            new_folder =  f"rot_3d_{self.rot_3d_mode}"
            stl_new_dir = os.path.join(self.stl_dir, new_folder)

        os.makedirs(stl_new_dir, exist_ok=True)

        if len(glob.glob(os.path.join(stl_new_dir, "*.stl"))) < self.num_stl:
            for filepath_stl in self.filepaths_stl:
                filepath_stl_new = os.path.join(filepath_stl).replace(self.stl_dir, stl_new_dir)
                if not os.path.exists(filepath_stl_new):
                    pcd_old=o3d.io.read_triangle_mesh(filepath_stl)
                    pcd_old.compute_vertex_normals()
                    pcd = pcd_bounding_rot(pcd_old, rot_3d_mode=self.rot_3d_mode)
                    o3d.io.write_triangle_mesh(filepath_stl_new, pcd)
                else:
                    print(f"{os.path.basename(filepath_stl_new)} skipped.")

            print(f"STL-Rotation finished. \nSaved at: {stl_new_dir}")
        else:
            print("Rotated stls already created.")

        self.stl_dir = stl_new_dir

    def calc_max_expansion(self, plot_bool: bool = False):
        """
        Imports stl-mesh, converts to pointcloud and outputs the max expansion of the model in xyz as expansion_max

        numpoints   : number of initial pointcloud points

        save_as     : ["json", "npz"], can be both

        plot_bool   : plot data?
        """
        import open3d as o3d

        cfg_search_hint = f"*{self.numpoints}*.npz"

        self.max_exp_cfg_path = glob.glob(
            os.path.join(self.stl_dir, cfg_search_hint))

        if not self.max_exp_cfg_path:
            pcd_expansion_max_x = np.array([])
            pcd_expansion_max_y = np.array([])

            for filepath_stl, ctr in zip(
                    self.filepaths_stl,
                    tqdm(
                        range(len(self.filepaths_stl)),
                        desc="Calculating max expansion of all pcd files..",
                        ascii=False,
                        ncols=100,
                    ),
            ):

                # Import Mesh (stl) and Create PointCloud from cropped mesh
                pcd = o3d.io.read_triangle_mesh(
                    filepath_stl).sample_points_uniformly(
                        number_of_points=self.numpoints)

                pcd_arr = np.asarray(pcd.points)

                # Crop the pcd_arr
                pcd_arr = pcd_arr[pcd_arr[:, 2] > (np.max(pcd_arr, axis=0)[2] -
                                                   self.z_threshold)]

                # Max Expansion after crop
                pcd_expansion_max = np.max(pcd_arr, axis=0) - np.min(pcd_arr,
                                                                     axis=0)

                # Swap x and y axis if max(x)<max(y)
                if (pcd_expansion_max[0] < pcd_expansion_max[1]) and not self.rot_3d:
                    pcd_expansion_max[:2] = pcd_expansion_max[[1, 0]]
                    # print("Axis swapped!")

                # First run
                if ctr == 0:
                    expansion_max = pcd_expansion_max
                else:
                    expansion_max = np.maximum(expansion_max,
                                               pcd_expansion_max)
                    # print(f"{ctr}: {expansion_max = }")
                    # print(f"{ctr}: {os.path.basename(filepath_stl)}")
                    # print(f"{ctr}: {pcd_expansion_max = }")

                pcd_expansion_max_x = np.append(pcd_expansion_max_x,
                                                pcd_expansion_max[0])
                pcd_expansion_max_y = np.append(pcd_expansion_max_y,
                                                pcd_expansion_max[1])

            if plot_bool:
                plt.figure(1)
                plt.rc("font", size=15)
                plt.plot(np.arange(len(self.filepaths_stl)),
                         pcd_expansion_max_x)
                plt.xlabel("Filenumber")
                plt.ylabel("Max x Expansion in mm")

                plt.figure(2)
                plt.rc("font", size=15)
                plt.plot(np.arange(len(self.filepaths_stl)),
                         pcd_expansion_max_y)
                plt.xlabel("Filenumber")
                plt.ylabel("Max y Expansion in mm")

                plt.show()

            expansion_max = np.round(expansion_max, 1)

            self.exp_max_cfg_name = f"calc_max_expansion_cfg_nump_{self.numpoints}"
            np.savez(os.path.join(self.stl_dir,
                                  f"{self.exp_max_cfg_name}.npz"),
                     expansion_max=expansion_max,
                     numpoints=self.numpoints,
                     z_threshold=self.z_threshold,
                     filepaths_stl=self.filepaths_stl)
                    # instantiate an empty dict

            params = {}
            params['expansion_max'] = expansion_max.tolist()
            params['numpoints'] = self.numpoints
            params['z_threshold'] = self.z_threshold
            params['files'] = self.filepaths_stl

            with open(os.path.join(self.stl_dir, f"{self.exp_max_cfg_name}.json"),"w") as f:
                json.dump(params, f)

            print("Done.")

            self.expansion_max = expansion_max
        else:
            print(f"Loading max_expansion_cfg from file: {self.max_exp_cfg_path[0]}")
            self.expansion_max = np.load(
                self.max_exp_cfg_path[0])["expansion_max"]

        if self.keep_xy_ratio:
            self.expansion_max[:2] = self.expansion_max[:2].max() * np.ones(
                shape=(1, 2))

        return self.expansion_max

    def create_param_sha256(self):
        """
        Creates Hash of used parameter-set and returns it
        """
        new_params_list = [
            False if new_param is None else new_param
            for new_param in [self.rot_3d, self.rot_3d_mode, self.rot_2d, self.rot_2d_center, self.invertY, self.keep_xy_ratio, self.conversion_type]
        ]
        if any(new_params_list):
            param_hash = sha256(
                np.concatenate((
                    np.array(self.normbounds).flatten(),
                    np.array(self.frame_size).flatten(),
                    np.array(self.nan_val).flatten(),
                    np.array(self.z_threshold).flatten(),
                    np.array(new_params_list),
                )).tobytes()).hexdigest()
        else:
            print("Use old param_hash function")
            param_hash = sha256(
                np.concatenate((
                    np.array(self.frame_size).flatten(),
                    np.array(self.expansion_max).flatten(),
                    np.array(self.nan_val).flatten(),
                    np.array(self.z_threshold).flatten(),
                )).tobytes()).hexdigest()
        self.param_hash = param_hash[::10]

        return self.param_hash

    def set_paths(self):
        grid_folder = f"{self.grid_size[0]}x{self.grid_size[1]}"

        foldername = self.param_hash

        if self.conversion_type:
            foldername += "-" + self.conversion_type

        if self.keep_xy_ratio:
            foldername += "-keepRatioXY"

        if self.invertY:
            foldername += "-invertY"

        if self.rot_3d:
            foldername += f"-rot_3d-{self.rot_3d_mode}"

        if self.rot_2d:
            foldername += "-rot_2d"
            if self.rot_2d_center:
                foldername += "-centered"



        # Paths
        save_dir_base = os.path.join(self.pcd_dir, f"pcd-{foldername}",
                                     grid_folder)

        self.pcd_grid_save_dir = os.path.join(save_dir_base, "pcd_grid")
        self.npy_grid_save_dir = os.path.join(save_dir_base, "npy_grid")

        image_foldername = foldername

        self.img_dir_grayscale_residual = None
        self.img_dir_rgb_residual = None

        if self.reduced_data_set:
            image_foldername += f"-reduced{self.num_images}"

        self.img_dir_grayscale = os.path.join(self.img_dir_base,
                                            f"images-{image_foldername}",
                                            "grayscale", grid_folder, "img")

        self.img_dir_rgb = self.img_dir_grayscale.replace("grayscale", "rgb")


        if self.reduced_data_set:
            self.img_dir_grayscale_residual = self.img_dir_grayscale + "_residual"
            self.img_dir_rgb_residual = self.img_dir_rgb + "_residual"
            os.makedirs(self.img_dir_grayscale_residual, exist_ok=True)
            os.makedirs(self.img_dir_rgb_residual, exist_ok=True)

        self.np_savepath = os.path.join(
            save_dir_base,
            f"einzelzaehne_train_lb{self.normbounds[0]}_ub{self.normbounds[1]}_{self.param_hash}.npy",
        )

    def create_params_cfg(self, save_as=["json", "npz"]):
        """
        Creates a File containing all parameters.

        name = pcd_to_grid_cfg_{param_hash}.json

        save_dir    : save directory for json File

        save_as     : ["json", "npz"], can be both

        returns all parameters as dict
        """

        print("Creating params cfg..")

        for save_el in save_as:
            if save_el not in ["json", "npz"]:
                raise ValueError('cfg_filetype must be in ["json", "npz"]')

        # Create unique hash for current parameter set
        if "param_hash" not in self.__dict__.keys():
            self.create_param_sha256()

        os.makedirs(self.cfg_dir, exist_ok=True)

        filepath = os.path.join(self.cfg_dir,
                                f"pcd_to_grid_cfg_{self.param_hash}")

        if "json" in save_as:
            # instantiate an empty dict
            params = {}
            params['param_hash'] = self.param_hash
            params['invertY'] = self.invertY
            params['keep_xy_ratio'] = self.keep_xy_ratio
            params['rot_3d'] = self.rot_3d,
            params['rot_3d_mode'] = self.rot_3d_mode,
            params['rot_2d'] = self.rot_2d,
            params['rot_2d_center'] = self.rot_2d_center,
            params['conversion_type'] = self.conversion_type if self.conversion_type is not None else "abs"
            params['frame_size'] = self.frame_size
            params['expansion_max'] = self.expansion_max.tolist()
            params['x_scale'] = np.round(
                self.expansion_max[0] + self.frame_size * 2, 2).tolist()
            params['y_scale'] = np.round(
                self.expansion_max[1] + self.frame_size * 2, 2).tolist()
            params['nan_val'] = self.nan_val
            params['z_threshold'] = self.z_threshold
            params['normbounds'] = self.normbounds

            with open(filepath+".json", "w") as f:
                json.dump(params, f)

        # if "npz" in save_as:
        #     np.savez(
        #         filepath + ".npz",
        #         param_hash=self.param_hash,
        #         invertY=self.invertY,
        #         keep_xy_ratio=self.keep_xy_ratio,
        #         rot_3d = self.rot_3d,
        #         rot_3d_mode = self.rot_3d_mode,
        #         rot_2d = self.rot_2d,
        #         conversion_type=self.conversion_type,
        #         frame_size=self.frame_size,
        #         expansion_max=self.expansion_max,
        #         x_scale=self.expansion_max[0] + self.frame_size * 2,
        #         y_scale=self.expansion_max[1] + self.frame_size * 2,
        #         nan_val=self.nan_val,
        #         z_threshold=self.z_threshold,
        #     )

    def pcd_to_grid(self,
                    filepath_stl,
                    save_path_pcd,
                    save_path_npy,
                    plot_bool=False):
        """
        Imports stl-mesh and
        converts irregular pointcloud to pointcloud with regular grid-size-grid

        grid_size    : tuple with grid size

        expansion_max: np.array with max expansion from calc_max_expansion

        z_threshold : crop pcd to z > (z_max - z_threshold)

        numpoints   : number of initial pointcloud points

        plot_bool   : (true) visualize the pcd

        frame_size  : frame size around tooth

        nan_val     : value for "no points here"

        invertY     : Invert the y_vec -> the cartesian point (0,0,0) will be at img[-1,0]: lower left corner

        conversion_type : ["abs", "rel"] 
        
        "rel": every pointcloud will be scaled according to its own pcd_expansion_max(x,y)

        "abs": all pointclouds will be scaled equally according to the maximal expansion_max(x,y) of all pcds

        """
        import open3d as o3d

        conversion_types = ["abs", "rel"]
        if self.conversion_type is not None and self.conversion_type not in conversion_types:
            raise ValueError(f"conversion type must be in {conversion_types}")

        # Import Mesh (stl) and Create PointCloud from cropped mesh
        pcd = o3d.io.read_triangle_mesh(filepath_stl).sample_points_uniformly(
            number_of_points=self.numpoints)

        # Execute transformations if specified
        if self.rotation_deg_xyz is not None:
            self.rotation_deg_xyz = np.asarray(self.rotation_deg_xyz)
            pcd.rotate(
                pcd.get_rotation_matrix_from_xyz(
                    (self.rotation_deg_xyz / 180 * np.pi)))

        # Convert Open3D.o3d.geometry.PointCloud to numpy array and get boundaries
        pcd_arr = np.asarray(pcd.points)

        # Crop the pcd_arr
        pcd_arr = pcd_arr[pcd_arr[:, 2] > (np.max(pcd_arr, axis=0)[2] -
                                           self.z_threshold)]

        # Max Expansion after crop
        pcd_expansion_max = np.max(pcd_arr, axis=0) - np.min(pcd_arr, axis=0)

        # Swap x and y values if max(x)<max(y)
        if self.rotation_deg_xyz is None and (pcd_expansion_max[0] <
                                              pcd_expansion_max[1]) and not self.rot_3d:
            pcd_expansion_max[:2] = pcd_expansion_max[[1, 0]]
            pcd_arr[:, :2] = pcd_arr[:, [1, 0]]
            # print("Axis swapped!")

        # Normalize the pointcloud to min(pcd) = zeros
        pcd_arr = pcd_arr - np.min(pcd_arr, axis=0)

        # convert to arr, json cant store arrays
        # Store as local variable (important)
        expansion_max = np.array(self.expansion_max)

        if self.conversion_type == "rel":
            expansion_max [:2] = pcd_expansion_max.max() * np.ones(shape=(1, 2))

        # Rigid Body transformation -> put the body in the middle of the xy meshgrid
        pcd_arr[:, :2] += (expansion_max [:2] - pcd_expansion_max[:2]) / 2

        if self.frame_size is not None:
            # Create frame around tooth
            pcd_arr[:, :2] += self.frame_size  # min(pcd) = ones*frame_size now
            expansion_max [:2] += 2 * self.frame_size

        # Create Vectors for mesh creation
        x_vec = np.linspace(0, expansion_max [0], self.grid_size[0])
        y_vec = np.linspace(0, expansion_max [1], self.grid_size[1])

        if self.invertY:
            # Invert the y_vec -> the cartesian point (0,0,0) will be at img[-1,0]: lower left corner
            y_vec = y_vec[::-1]

        # Create meshgrid
        [X, Y] = np.meshgrid(x_vec, y_vec)

        # Create points from meshgrid
        points = np.c_[X.ravel(), Y.ravel()]

        # Create tree with pcd points
        tree = spatial.cKDTree(pcd_arr[:, :2])

        # initialize zvals array
        zvals = np.zeros(np.shape(points[:, 0]))

        # Define ball radies as the max distance between 2 point neighbors in the current pcd
        pcd.points = o3d.utility.Vector3dVector(pcd_arr)
        query_ball_radius = np.max(pcd.compute_nearest_neighbor_distance())

        # Go through mesh grid points and search for nearest points with ballradius
        for ctr, results in enumerate(
                tree.query_ball_point(points, query_ball_radius)):
            if results:
                # Save max z value from results if results is not empty
                zvals[ctr] = pcd_arr[results, 2].max()
            else:
                # Save nan_val if results = empty
                zvals[ctr] = self.nan_val

        # Concatenate the xy meshgrid and the zvals from loop
        new_pcd_arr = np.concatenate([points, zvals[:, np.newaxis]], axis=1)

        # Generate empty Open3D.o3d.geometry.PointCloud
        new_pcd = o3d.geometry.PointCloud()
        new_pcd.points = o3d.utility.Vector3dVector(new_pcd_arr)

        # # Print Relation
        # print(f"zvals_nan/zvals = {zvals[zvals == nan_val].shape[0] / zvals.shape[0]}")

        # Visualize PointCloud
        if plot_bool:
            o3d.visualization.draw_geometries([new_pcd])

        # Save the new pcd
        if save_path_pcd:
            os.makedirs(os.path.dirname(save_path_pcd), exist_ok=True)
            o3d.io.write_point_cloud(save_path_pcd, new_pcd)
        if save_path_npy:
            os.makedirs(os.path.dirname(save_path_npy), exist_ok=True)
            np.save(save_path_npy, new_pcd_arr)

        return new_pcd_arr

    def grid_pcd_to_2D_np(self, pcd_filetype = "pcd" ):
        """
        Load all pcd files in pcd_dirame, create normalized 2D np array and append to training array  

        pcd_dirname     : directory of pcd files  

        np_savepath     : filepath where the training_data file will be saved  

        grid_size,      : grid_size for img  

        normbounds      : [0,1] (for img conversion) or [-1, 0]  

        z_threshold,    : max z expansion for all pcd, used for normalization --> z_threshold  

        nan_val,        : nan val of pcds  

        pcd_filetype    : "npy" or "pcd"

        filename_npy    : the filename of the .npy file that will be generated  

        plot_img_bool   : plot first img if true  
        """
        import open3d as o3d
        
        pcd_filetypes = ["npy", "pcd"]
        if not pcd_filetype in pcd_filetypes:
            raise ValueError(f"pcd_filetype must be in {pcd_filetypes}")  

        os.makedirs(os.path.dirname(self.np_savepath), exist_ok=True)

        print(f"Creating {os.path.basename(self.np_savepath)}-File for Training..")

        if pcd_filetype == "pcd":
            pcd_dir = self.pcd_grid_save_dir
        elif pcd_filetype == "npy":
            pcd_dir = self.npy_grid_save_dir

        # Get all pcd filepaths
        pcd_paths = sorted(glob.glob(os.path.join(pcd_dir, f"*.{pcd_filetype}")))

        # Init the training array
        train_images = np.zeros(
            (len(pcd_paths), self.grid_size[0], self.grid_size[1], 1)).astype("float32")

        # Loop through all files, normalize
        for num, filename in enumerate(pcd_paths):
            if pcd_filetype == "pcd":
                # Read pcd
                pcd = o3d.io.read_point_cloud(filename)
                # Save as np array
                pcd_arr = np.asarray(pcd.points)
            elif pcd_filetype == "npy":
                pcd_arr = np.load(filename)

            # Reshape as 2D img
            pcd_img = pcd_arr[:, 2].reshape([self.grid_size[0], self.grid_size[1], 1])

            # Normalize z values to normbounds
            pcd_img[pcd_img != self.nan_val] = (pcd_img[pcd_img != self.nan_val] +
                                        self.z_threshold * self.normbounds[0] /
                                        (self.normbounds[1] - self.normbounds[0])) / (
                                            self.z_threshold /
                                            (self.normbounds[1] - self.normbounds[0]))

            # Nan values = -1
            pcd_img[pcd_img == self.nan_val] = self.normbounds[0]

            # Add to training array
            train_images[num] = pcd_img.reshape([self.grid_size[0], self.grid_size[1], 1])

        # Save the training array
        np.save(self.np_savepath, train_images)

        self.train_images = train_images
        return self.train_images


    def np_grid_to_grayscale_png(self):
        """
        Load npy Array and create grayscale png images  

        npy_path        # Path to npy File  

        img_dir         # saving directory for images  

        param_sha       # Hashed parameter set  
        """
        import img_tools.image_processing as ip

        print(
            f"Creating greyscale images for current parameter-set {self.param_hash[:10]}.."
        )

        if not "train_images" in self.__dict__.keys():
            images = np.load(self.np_savepath)
        else:
            images = self.train_images

        # images = images[:self.num_images]

        if images.max() > 1 or images.min() < 0:
            raise ValueError(f"Expected values between 0 and 1. Got values between {images.min()} and {images.max()}")

        images = images * 255

        # Get rid of last dimension
        images = images.reshape((
            images.shape[0],
            images.shape[1],
            images.shape[2],
        )).astype(np.uint8)

        # Create the directory with all parents
        os.makedirs(self.img_dir_grayscale, exist_ok=True)

        for img, ctr in zip(
                images,
                tqdm(
                    iterable=range(len(images)),
                    desc="Converting np to grayscale png..",
                    ascii=False,
                    ncols=100,
                ),
        ):
            img_name = f"img_{ctr:04d}_{self.param_hash}.png"

            if ctr < self.num_images:
                img_dir = self.img_dir_grayscale
            else:
                img_dir = self.img_dir_grayscale_residual

            img_path = os.path.join(img_dir, img_name)

            g_img = PIL.Image.fromarray(img, "L")
            if self.rot_2d:
                ImageProps = ip.ImageProps(img=img)
                ImageProps.set_orientation_zero(mode=self.rot_2d_mode, center=self.rot_2d_center, show_img=self.rot_2d_show_img)
                ImageProps.save_images(img_types="current", img_basename=img_name, img_new_dir=img_dir)
                
            else:
                g_img.save(img_path, )

        print("Done.")


    def create_trainingdata(self):
        print("\n")
        print(f"Current grid-size: {self.grid_size}")
        if self.rot_3d:
            self.prepare_stl()
        self.calc_max_expansion()
        self.create_param_sha256()
        print(f"Current Param Hash: {self.param_hash}")
        self.set_paths()
        self.create_params_cfg()
        
        if not os.path.exists(self.pcd_grid_save_dir) or len(os.listdir(self.pcd_grid_save_dir))<self.num_stl \
        or not os.path.exists(self.npy_grid_save_dir) or len(os.listdir(self.npy_grid_save_dir))<self.num_stl:
            
            for filepath_stl, num in zip(
                    self.filepaths_stl,
                    tqdm(
                        range(self.num_stl),
                        desc="Creating pcd-grid files..",
                        ascii=False,
                        ncols=100,
                    ),
            ):
                self.pcd_name = f"einzelzahn_grid_{num:04d}_{self.param_hash}"
                save_path_pcd = os.path.join(
                    self.pcd_grid_save_dir,
                    f"{self.pcd_name}.pcd")  
                save_path_npy = os.path.join(
                    self.npy_grid_save_dir,
                    f"einzelzahn_grid_{num:04d}_{self.param_hash}.npy")
                if not os.path.exists(save_path_pcd) or not os.path.exists(
                        save_path_npy):
                    self.pcd_to_grid(filepath_stl=filepath_stl,
                                     save_path_pcd=save_path_pcd,
                                     save_path_npy=save_path_npy)
        else:
            print(f"Pcds for param-set <{self.param_hash}> already exist.")


        if not os.path.exists(self.np_savepath):
            self.grid_pcd_to_2D_np()

        # Convert the 2D numpy array to grayscale images for nvidia stylegan
        if not os.path.exists(self.img_dir_grayscale) or len(os.listdir(self.img_dir_grayscale))< self.num_images:
            # Creating grayscale images
            self.np_grid_to_grayscale_png()
        else:
            print(
                    f"L-PNGs for param-set <{self.param_hash}> already exist at: \n{self.img_dir_grayscale}"
                )
        
        if not os.path.exists(self.img_dir_rgb) or len(os.listdir(self.img_dir_rgb))< self.num_images:
            # Converting to rgb and save in different folder
            image_conversion_L_RGB(source_img_dir=self.img_dir_grayscale, rgb_dir = self.img_dir_rgb)
            if self.reduced_data_set:
                image_conversion_L_RGB(source_img_dir=self.img_dir_grayscale_residual, rgb_dir = self.img_dir_rgb_residual)

        else:
            print(
            f"RGB-PNGs for param-set <{self.param_hash}> already exist at: \n{self.img_dir_rgb}"
            )

        print("\n")


class ImageConverterParams:

    @classmethod
    def __init__(cls, img_dir = None, param_hash = None, cfg_dir = None):

        cls.img_dir = img_dir
        cls.cfg_dir = cfg_dir

        if param_hash is None:
            cls.get_param_hash_from_img_path()
            if not cls.param_hash:
                raise ValueError(f"Could not find a matching param_hash for {cls.img_dir}")
        else:
            cls.param_hash = param_hash

        cls.search_pcd_cfg()

        print(f"Param Hash: {cls.param_hash}")

        if "invertY" in cls.params:
            cls.invertY = cls.params['invertY']
        else:
            cls.invertY = False

        if "conversion_type" in  cls.params:
            cls.conversion_type = cls.params['conversion_type']
        else:
            cls.conversion_type = "abs"

        if "keep_xy_ratio" in cls.params:
            cls.keep_xy_ratio = cls.params['keep_xy_ratio']
        else:
            cls.keep_xy_ratio =  False

        if "normbounds" in cls.params:
            cls.normbounds = cls.params['normbounds']
        else:
            cls.normbounds = [0,1]

        cls.x_scale = cls.params['x_scale']
        cls.y_scale = cls.params['y_scale']
        cls.nan_val = cls.params['nan_val']
        cls.z_threshold = cls.params['z_threshold']
        


    @classmethod
    def get_param_hash_from_img_path(cls):
        # Finds the matching param_hash in cfg_dir for the specified img_dir
        if cls.cfg_dir is not None:
            param_hashes = [cfg_file.split(".")[0].split("_")[-1] for cfg_file in glob.glob(os.path.join(cls.cfg_dir, "pcd_to_grid_cfg*.npz"))]
            cls.param_hash = [param_hash for param_hash in param_hashes if param_hash in cls.img_dir][0]
        else:
            cls.param_hash = cls.img_dir.split(f"images-")[-1].split("-")[0]

    @classmethod
    def search_pcd_cfg(cls):
        """
        searches for pcd_cfg file in cls.cfg_dir

        uses param_hash if given 

        returns the params as dict  

        """
        if cls.cfg_dir is None:
            if os.name == "nt":
                search_path = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\cfg"
            elif os.name == "posix":
                search_path = "/home/proj_depo/docker/data/einzelzahn/cfg"

        if cls.param_hash:
            # Load pcd_to_grid cfg file
            cls.pcd_to_grid_cfg_path = os.path.join(
                search_path, f"pcd_to_grid_cfg_{cls.param_hash}.json")
            if not os.path.exists(cls.pcd_to_grid_cfg_path):
                print("No cfg for param-hash found. \nReturning [].")
                return []
        else:
            pcd_to_grid_cfg_list = glob.glob(
                os.path.join(search_path, f"pcd_to_grid_cfg*.json"))
            if len(pcd_to_grid_cfg_list) > 1:
                for num, pathname in enumerate(pcd_to_grid_cfg_list):
                    print(f"Index {num}: {os.path.basename(pathname)}")
                cls.pcd_to_grid_cfg_path = pcd_to_grid_cfg_list[int(
                    input(f"Enter Index for preferred cfg-File: \n"))]
            elif len(pcd_to_grid_cfg_list) == 1:
                cls.pcd_to_grid_cfg_path = pcd_to_grid_cfg_list[0]

        with open(cls.pcd_to_grid_cfg_path) as f:
            cls.params = json.load(f)






class ImageConverterSingle(ImageConverterParams):

    # Image
    img = None
    img_path =  None
    channels = None
    grid_size = None

    # NP Array for img_to_2D_np
    np_img = None
    np_savepath = None

    # np_2D_to_grid_pcd
    save_path_pcd = None
    pcd_name = None
    visu_bool = False
    save_pcd = False
    z_crop = 0

    def __init__(self, img_path = None, img = None, crop = True, center = True, rot = True):

        self.crop = crop
        self.center = center
        self.rot = rot

        self.img_path = img_path
        self.img = img

        if img is None and img_path is None:
            raise ValueError("Provide either img or img_path!")


    @property
    def numpoints(self): 
        return self.grid_size[0] * self.grid_size[1]

    def preprocess_image(self):
        import img_tools.image_processing as ip
        # Load the image
        if self.img is None:
            self.img = np.asarray(PIL.Image.open(self.img_path))

        if self.rot or self.center or self.crop:
            ImageProps = ip.ImageProps(img=self.img, rgb_format="RGB")
            if self.crop:
                ImageProps.crop(show_img=False)
            if self.center:
                ImageProps.center()
            if self.rot:
                # Handles: rot + center and only rot
                ImageProps.set_orientation_zero(mode="auto", center=self.center, show_img=False)

            self.img = ImageProps.export(img_type="current")

    def img_to_pcd(self, save_pcd=None, save_path_pcd=None, visu_bool = None, np_savepath = None, z_crop=None):
        """
        Converts 8 Bit RGB or L image into 3D-pcd-arr

        z_crop: crops the pcd from the bottom up // min z_crop to take away the bottom plane 

        """
        if save_pcd is not None:
            self.save_pcd = save_pcd
        if visu_bool is not None:
            self.visu_bool = visu_bool
        if z_crop is not None:
            self.z_crop = z_crop
        if np_savepath is not None:
            self.np_savepath = np_savepath
        if save_path_pcd is not None:
            self.save_path_pcd = save_path_pcd
        if self.img_path is not None:
            self.pcd_name = os.path.basename(self.img_path).split(".")[0]

        self.preprocess_image()
        self.img_to_2D_np()
        self.np_2D_to_grid_pcd()

        return self.pcd_arr


    def img_to_2D_np(self, np_savepath = None):
        """
        Load img and convert to [0,1] normalized stacked np.array 

        save the array if np_savepath is specified  

        returns the img as np.array with shape = (grid_size[0], grid_size[1], 1)  

        img_path    : image path  

        np_savepath : full path for npy file save, leave empty if you dont want to save the file  

        """
        if np_savepath is not None:
            self.np_savepath = np_savepath

        # Load the image
        if self.img is None:
            self.img = np.asarray(PIL.Image.open(self.img_path))

        self.grid_size = self.img.shape[:2]

        # Check for right format (RGB/L)
        self.channels = self.img.shape[2] if self.img.ndim == 3 else 1

        if self.channels not in [1, 3]:
            raise ValueError("Input images must be stored as RGB or grayscale")

        if self.channels == 3:
            # print("RGB-Image")
            # If RGB convert to L and normalize to 0..1
            self.np_img = image_conversion_RGB_L(
                img=self.img, conv_type="luminance_float_exact") / 255
        else:
            # print("L-Image")
            # Normalize to 0..1
            self.np_img = self.img / 255

        # Save the npy File if np_savepath is specified
        if np_savepath is not None:
            np.save(np_savepath, self.np_img)
            print(f"Images saved as: {os.path.basename(np_savepath)}")

        return self.np_img

    def np_2D_to_grid_pcd(self, np_img=None, save_pcd = None, np_filepath = None, visu_bool = None, save_path_pcd = None):
        """
        Load 2D np image (as file or path) and convert to 3D PointCloud with params-data

        visu_bool:              visualize the img and the 3D obj?  

        save_path_pcd:      save_path for the pcd file, create pcd folder in np_filepath directory and save with same name if not specified and save_pcd =  True

        """

        if save_pcd is not None:
            self.save_pcd = save_pcd
        if visu_bool is not None:
            self.visu_bool = visu_bool
        if save_path_pcd is not None:
            self.save_path_pcd = save_path_pcd
        if np_img is not None:
            self.np_img = np_img
        

        if self.save_path_pcd is not None or self.visu_bool:
            import open3d as o3d

        # Load file with path or file-input
        if np_filepath is not None and self.np_img is None:
            self.np_img = np.load(np_filepath)
        elif np_filepath is None and self.np_img is None:
            raise ValueError("Input np_filepath or np_file!")

        if self.np_img.ndim == 2:
            self.np_img = self.np_img[:,:,np.newaxis]

        if self.np_img.shape[0] == 1:
            print("Transpose")
            self.np_img = self.np_img.transpose((1, 2, 0))

        # Flatten the 2D array
        z_img = self.np_img.reshape((-1, 1))

        # Undo the normalization
        z_img = z_img * self.z_threshold / (
            self.normbounds[1] - self.normbounds[0]) - self.z_threshold * self.normbounds[0] / (
                self.normbounds[1] - self.normbounds[0])

        # Create Vectors for mesh creation
        x_vec = np.linspace(0, self.x_scale, self.grid_size[0])
        y_vec = np.linspace(0, self.y_scale, self.grid_size[1])

        if self.invertY:
            # Invert the y_vec -> if the cartesian point (0,0,0) is at img[-1,0]: lower left corner
            y_vec = y_vec[::-1]

        # Create meshgrid
        [X, Y] = np.meshgrid(x_vec, y_vec)

        # Create points from meshgrid
        xy_points = np.c_[X.ravel(), Y.ravel()]

        # Generate a 3D array with z-values from image and meshgrid
        pcd_arr = np.concatenate([xy_points, z_img], axis=1)

        # Crop array to take away the bottom plane from img = 0
        pcd_arr = pcd_arr[pcd_arr[:, 2] >= 1/256]

        # Crop the array if needed
        pcd_arr = pcd_arr[pcd_arr[:, 2] >= self.z_crop]
   
        if self.save_pcd or self.visu_bool:
            # Define empty Pointcloud object and occupy with new points
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(pcd_arr)

            # Save the new pcd
            if self.save_pcd:
                if self.save_path_pcd is None:
                    # pcdpath = np path if not specified
                    if np_filepath is not None:
                        self.save_path_pcd = os.path.join(
                            os.path.dirname(np_filepath), "pcd",
                            self.pcd_name, ".pcd")
                    elif self.np_img is not None:
                        raise ValueError(
                            "Specify save_path_pcd if np_filepath is not specified.")

                os.makedirs(os.path.dirname(self.save_path_pcd), exist_ok=True)
                o3d.io.write_point_cloud(self.save_path_pcd, self.pcd)

            # Visualize the input img and the 3D-obj
            if self.visu_bool:
                plt.figure(self.pcd_name if self.pcd_name is not None else "Image")
                plt.rc("font", size=15)
                plt.imshow(self.np_img, cmap="viridis")
                plt.colorbar()
                plt.show()
                o3d.visualization.draw_geometries([self.pcd],  window_name=self.pcd_name if self.pcd_name is not None else "3D-Model" )


        self.pcd_arr = pcd_arr

        return self.pcd_arr


class ImageConverterMulti(ImageConverterSingle):

    np_img = None
    channels = None
    grid_size = None
    pcd_outdir = None

    def __init__(self, img_dir=None, crop = True, center = True, rot = True):

        self.crop = crop
        self.center = center
        self.rot = rot

        # Take img_dir from ImageConverterParams init
        if img_dir is not None:
            self.img_dir = img_dir

        self.img_paths = glob.glob(os.path.join(self.img_dir, "*.png"))
        self.num_images = len(self.img_paths)


    def img_to_pcd_multi(self, save_pcd=None, pcd_outdir=None, visu_bool = None, z_crop=None):
        """
        Converts all images in self.img_dir to pcd

        z_crop: crops images from the bottom. Example: z_crop = 1 -> pcd = pcd[pcd[:,2]]

        pcd_save: Images will be saved

        visu_bool: Show pcd

        if pcd_outdir is not provided: pcd_outdir = os.path.join(self.img_dir, "pcd")
        """

        if save_pcd is not None:
            self.save_pcd = save_pcd
        if pcd_outdir is not None:
            self.pcd_outdir = pcd_outdir
        if visu_bool is not None:
            self.visu_bool = visu_bool
        if z_crop is not None:
            self.z_crop = z_crop


        if self.pcd_outdir is None and self.save_pcd:
            self.pcd_outdir = os.path.join(self.img_dir, "pcd")
            os.makedirs(self.pcd_outdir, exist_ok=True)

            with open(os.path.join(self.pcd_outdir, os.path.basename(self.pcd_to_grid_cfg_path)), "w") as f:
                json.dump(self.params, f)


        for img_path, num in zip(
                self.img_paths,
                tqdm(iterable=range(self.num_images),
                        desc=f"Converting {self.num_images} images to pcd..",
                        ascii=False,
                        ncols=100)):
            self.pcd_name = os.path.basename(img_path).split(".")[0] 
            if self.save_pcd:
                self.save_path_pcd = os.path.join(self.pcd_outdir, self.pcd_name + ".pcd")

            self.img_path = img_path
            self.img_to_pcd()


