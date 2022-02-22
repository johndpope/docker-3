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
import sys
import copy
import open3d as o3d
import time
import json

import img_tools.image_processing as ip

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
        os.makedirs(rgb_dir)

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



def points_to_pcd(points, color=None):
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

        plane_mat = copy.deepcopy(mat)
        plane_mat[:,2] = mat[:,0]*alpha + mat[:,1]*beta + np.ones(shape=(mat.shape[0]))*gamma

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

def pcd_bounding_rot(pcd, rot_3d_mode, rotY_noask = True, triangle_normal_z_threshold = 0.9):
    """
    Rotate provided 3D-model depending on bounding-box Rotation.

    rotY_noask: rotate around Y after bounding box rotation if True

    """
    # Rotate
    obb = pcd.get_oriented_bounding_box()
    obb.color = (0,1,0)
    # Calculate the inverse Rotation (inv(R)=R.T for rot matrices)
    rot_mat = obb.R.T

    # Choose rotation mode
    if rot_3d_mode == "z":
        euler_rot_rad = rotationMatrixToEulerAngles(rot_mat)
        R = pcd.get_rotation_matrix_from_xyz((np.asarray([0,0,euler_rot_rad[-1]])))
        pcd = pcd.rotate(R, center=obb.center)
    elif rot_3d_mode == "full":
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
        
        # Rotate with inverse Orientation of the calculated plane
        pcd = pcd.rotate(rot_mat_plane.T, center=obb.center)

        visu_list = [pcd.sample_points_uniformly(number_of_points=1000000), frame]

        visu_list.append(plane_pcd)

        # Calculate plane_mat again for the rotated pcd
        # must be parallel to xy plane now
        plane_mat_rot, _ = fit_plane_to_pcd(mat=np.asarray(pcd_new.rotate(rot_mat_plane.T, center=obb.center).points))
        visu_list.append(points_to_pcd(points=plane_mat_rot, color=[0,1,0]))

        o3d.visualization.draw_geometries(visu_list, width=720, height=720, window_name=f"Rotated Tooth", left=1000, top=300)

    return pcd
# ------------------------------- #

## Processing


class DataParams:

    @classmethod
    def __init__(cls, z_threshold, normbounds, frame_size, nan_val,
                 conversion_type,stl_dir, pcd_dir, cfg_dir, img_dir_base, 
                 invertY=None, keep_xy_ratio=None, rot_3d=None, rot_2d=None, rot_3d_mode=None, rot_2d_mode=None):

        cls.z_threshold = z_threshold
        cls.normbounds = normbounds
        cls.frame_size = frame_size
        cls.nan_val = nan_val
        cls.conversion_type = conversion_type
        cls.invertY = invertY
        cls.keep_xy_ratio = keep_xy_ratio
        cls.rot_3d = rot_3d
        cls.rot_3d_mode = rot_3d_mode if rot_3d else None
        cls.rot_2d = rot_2d
        cls.rot_2d_mode = rot_2d_mode if rot_2d else None
        cls.stl_dir = stl_dir
        cls.pcd_dir = pcd_dir
        cls.cfg_dir = cfg_dir
        cls.img_dir_base = img_dir_base




class Dataset(DataParams):

    def __init__(self, grid_size, rotation_deg_xyz = None):
        self.grid_size = grid_size
        self.rotation_deg_xyz = rotation_deg_xyz
        self.numpoints = grid_size[0] * grid_size[1] * 10

    @property
    def num_stl(self): 
        return len(glob.glob(os.path.join(self.stl_dir, "*.stl")))

    @property
    def filepaths_stl(self):
        return sorted(glob.glob(os.path.join(self.stl_dir, "*.stl")))

    def prepare_stl(self, stl_new_dir = None):
        """
        Loads .stl files from directory, rotates them depending on their bounding-box orientation and saves the new files to stl_new_dir.

        stl_dir:        Directory with .stl files

        stl_new_dir:    New Directory for rotated .stl files

        rot_3d_mode:   "full": all axis rotation with user-input,  "z": only z axis

        """
        # Check function input
        rot_3d_modes = ["full", "z"]
        if not self.rot_3d_mode in rot_3d_modes:
            raise ValueError(f"rot_3d_mode must be in {rot_3d_modes}")

        if stl_new_dir is None:
            if self.rot_3d_mode == "full":
                new_folder =  "rot_3d_full"
            elif self.rot_3d_mode == "z":
                new_folder = "rot_3d_z"
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
            for new_param in [self.rot_3d, self.rot_3d_mode, self.rot_2d, self.invertY]
        ]
        if any(new_params_list):
            param_hash = sha256(
                np.concatenate((
                    np.array(self.frame_size).flatten(),
                    np.array(self.expansion_max).flatten(),
                    np.array(self.nan_val).flatten(),
                    np.array(self.z_threshold).flatten(),
                    np.array(new_params_list),
                )).tobytes()).hexdigest()
        else:
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

        # Paths
        save_dir_base = os.path.join(self.pcd_dir, f"pcd-{foldername}",
                                     grid_folder)

        self.pcd_grid_save_dir = os.path.join(save_dir_base, "pcd_grid")
        self.npy_grid_save_dir = os.path.join(save_dir_base, "npy_grid")

        self.img_dir_grayscale = os.path.join(self.img_dir_base,
                                              f"images-{foldername}",
                                              "grayscale", grid_folder, "img")

        self.img_dir_rgb = self.img_dir_grayscale.replace("grayscale", "rgb")

        self.img_dir_rgb_cvRot = self.img_dir_rgb + "-cvRot"

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
            params['conversion_type'] = self.conversion_type
            params['frame_size'] = self.frame_size
            params['expansion_max'] = self.expansion_max.tolist()
            params['x_scale'] = np.round(
                self.expansion_max[0] + self.frame_size * 2, 2).tolist()
            params['y_scale'] = np.round(
                self.expansion_max[1] + self.frame_size * 2, 2).tolist()
            params['nan_val'] = self.nan_val
            params['z_threshold'] = self.z_threshold

            with open(filepath+".json", "w") as f:
                json.dump(params, f)

        if "npz" in save_as:
            np.savez(
                filepath + ".npz",
                param_hash=self.param_hash,
                invertY=self.invertY,
                keep_xy_ratio=self.keep_xy_ratio,
                conversion_type=self.conversion_type,
                frame_size=self.frame_size,
                expansion_max=self.expansion_max,
                x_scale=self.expansion_max[0] + self.frame_size * 2,
                y_scale=self.expansion_max[1] + self.frame_size * 2,
                nan_val=self.nan_val,
                z_threshold=self.z_threshold,
            )

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
        if self.conversion_type not in conversion_types:
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

        # if self.conversion_type == "abs":

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
        print(
            f"Creating greyscale images for current parameter-set {self.param_hash[:10]}.."
        )

        if not "train_images" in self.__dict__.keys():
            images = np.load(self.np_savepath)
        else:
            images = self.train_images

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
        os.makedirs(self.img_dir_grayscale)

        if self.rot_2d:
            ip.ImageProps.set_img_dir(self.img_dir_grayscale)

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
            img_path = os.path.join(self.img_dir_grayscale, img_name)

            g_img = PIL.Image.fromarray(img, "L")
            if self.rot_2d:
                ip.ImagePropsRot(img=img, mode=self.rot_2d_mode, show_img=True).save_images(img_types="current", img_basename=img_name)
            else:
                g_img.save(img_path, )

        print("Done.")


    def create_trainingdata(self):
        print("\n")
        if self.rot_3d:
            self.prepare_stl()
        self.calc_max_expansion()
        self.create_param_sha256()
        print(f"Current Param Hash: {self.param_hash}")
        self.set_paths()
        if not os.path.exists(self.pcd_grid_save_dir) or len(os.listdir(self.pcd_grid_save_dir))<self.num_stl \
        or not os.path.exists(self.npy_grid_save_dir) or len(os.listdir(self.npy_grid_save_dir))<self.num_stl:
            self.create_params_cfg()

            for filepath_stl, num in zip(
                    self.filepaths_stl,
                    tqdm(
                        range(self.num_stl),
                        desc="Creating pcd-grid files..",
                        ascii=False,
                        ncols=100,
                    ),
            ):
                save_path_pcd = os.path.join(
                    self.pcd_grid_save_dir,
                    f"einzelzahn_grid_{num:04d}_{self.param_hash}.pcd")
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
        if not os.path.exists(self.img_dir_grayscale) or len(os.listdir(self.img_dir_grayscale))<self.num_stl:
            # Creating grayscale images
            self.np_grid_to_grayscale_png()
        else:
            print(
                    f"L-PNGs for param-set <{self.param_hash}> already exist at: \n{self.img_dir_grayscale}"
                )

        if not os.path.exists(self.img_dir_rgb) or len(os.listdir(self.img_dir_rgb))<self.num_stl:
            # Converting to rgb and save in different folder
            image_conversion_L_RGB(source_img_dir=self.img_dir_grayscale, rgb_dir = self.img_dir_rgb)
        else:
            print(
            f"RGB-PNGs for param-set <{self.param_hash}> already exist at: \n{self.img_dir_rgb}"
            )

        print("\n")


    

#     @staticmethod
#     def pcd_bounding_rot(pcd, rot_3d_mode, rotY_noask = True):
#         """
#         Rotate provided 3D-model depending on bounding-box Rotation.

#         rotY_noask: rotate around Y after bounding box rotation if True

#         """
#         # Rotate
#         obb = pcd.get_oriented_bounding_box()
#         # Calculate the inverse Rotation (inv(R)=R.T for rot matrices)
#         rot_mat = obb.R.T

#         frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=obb.center)
#         o3d.visualization.draw_geometries([pcd.sample_points_uniformly(number_of_points=100000), obb, frame], width=720, height=720, window_name=f"Rotated Tooth", left=1000, top=300)

#         if rot_3d_mode == "z":
#             euler_rot_rad = rotationMatrixToEulerAngles(rot_mat)
#             R = pcd.get_rotation_matrix_from_xyz((np.asarray([0,0,euler_rot_rad[-1]])))
#             pcd = pcd.rotate(R, center=obb.center)
#         elif rot_3d_mode == "full":
#             pcd = pcd.rotate(rot_mat, center=obb.center)

#             x_rot = np.array( ([1, 0, 0], [0, -1, 0], [0, 0, -1]) )[np.newaxis, :,:]
#             y_rot = np.array( ([-1, 0, 0], [0, 1, 0], [0, 0, -1]) )[np.newaxis, :,:]
#             z_rot = np.array( ([-1, 0, 0], [0, -1, 0], [0, 0, 1]) )[np.newaxis, :,:]
#             rots_xyz = np.concatenate([x_rot, y_rot, z_rot], axis=0)

#             if rotY_noask:
#                 pcd = pcd.rotate(rots_xyz[1], center=obb.center)

#             frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=obb.center)

#             while 1:
#                 obb2 = pcd.get_oriented_bounding_box()
#                 obb2.color = (0,0,1)
#                 aabb = pcd.get_axis_aligned_bounding_box()
#                 aabb.color = (1, 0, 0)
#                 o3d.visualization.draw_geometries([pcd.sample_points_uniformly(number_of_points=100000), frame, aabb, obb2], width=720, height=720, window_name=f"Rotated Tooth", left=1000, top=300)
#                 rot_axis = input("\nTurn 180 degress around axis: \n0: x-red \n1: y-green\n2: z-blue \n3: Show Visu again.\n9: Finish\nUser-Input: ")
                
#                 if rot_axis in ["0","1","2"]:
#                     rot = rots_xyz[int(rot_axis)]
#                     pcd = pcd.rotate(rot, center=obb.center)
#                 elif rot_axis == "3":
#                     continue
#                 elif rot_axis == "9":
#                     break
#                 else:
#                     print("Please insert from list [0, 1, 2, 3, 9].")
#         return pcd

# class DataFile(DataParams):

#     def __init__(self):
#         pass
