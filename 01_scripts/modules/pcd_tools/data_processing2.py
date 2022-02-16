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
import open3d as o3d

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


# ------------------------------- #

## Processing


class DataCreator:

    @classmethod
    def __init__(cls, z_threshold, normbounds, frame_size, nan_val,
                 conversion_type, invertY, keep_xy_ratio, rot_bb, rot_cv,
                 stl_dir, pcd_dir, cfg_dir, img_dir_base):

        cls.z_threshold = z_threshold
        cls.normbounds = normbounds
        cls.frame_size = frame_size
        cls.nan_val = nan_val
        cls.conversion_type = conversion_type
        cls.invertY = invertY
        cls.keep_xy_ratio = keep_xy_ratio
        cls.rot_bb = rot_bb
        cls.rot_cv = rot_cv
        cls.stl_dir = stl_dir
        cls.pcd_dir = pcd_dir
        cls.cfg_dir = cfg_dir
        cls.img_dir_base = img_dir_base


class Dataset(DataCreator):

    def __init__(self, grid_size, rotation_deg_xyz = None):
        self.grid_size = grid_size
        self.rotation_deg_xyz = rotation_deg_xyz
        self.numpoints = grid_size[0] * grid_size[1] * 10
        self.num_stl = len(glob.glob(os.path.join(self.stl_dir, "*.stl")))
        self.filepaths_stl = sorted(
            glob.glob(os.path.join(self.stl_dir, "*.stl")))

    def calc_max_expansion(self, plot_bool: bool = False):
        """
        Imports stl-mesh, converts to pointcloud and outputs the max expansion of the model in xyz as expansion_max

        numpoints   : number of initial pointcloud points

        save_as     : ["json", "npz"], can be both

        plot_bool   : plot data?
        """

        if self.rot_bb:
            cfg_search_hint = f"*{self.numpoints}_rot*"
        else:
            cfg_search_hint = f"*{self.numpoints}*"

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
                if pcd_expansion_max[0] < pcd_expansion_max[1]:
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
            print("Done.")

            self.expansion_max = expansion_max
        else:
            print("Loading max_expansion_cfg from file..")
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
            for new_param in [self.rot_bb, self.rot_cv, self.invertY]
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

        if self.rot_bb:
            foldername += "-rot_bb"

        if self.rot_cv:
            foldername += "-rot_cv"

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
                                f"pcd_to_grid_cfg_{self.param_hash}.filetype")

        if "json" in save_as:
            # instantiate an empty dict
            params = {}
            params['param_hash'] = self.param_hash
            params['invertY'] = self.invertY
            params['keep_xy_ratio'] = self.keep_xy_ratio
            params['rot_bb'] = self.rot_bb,
            params['rot_cv'] = self.rot_cv,
            params['conversion_type'] = self.conversion_type
            params['frame_size'] = self.frame_size
            params['expansion_max'] = self.expansion_max.tolist()
            params['x_scale'] = np.round(
                self.expansion_max[0] + self.frame_size * 2, 2).tolist()
            params['y_scale'] = np.round(
                self.expansion_max[1] + self.frame_size * 2, 2).tolist()
            params['nan_val'] = self.nan_val
            params['z_threshold'] = self.z_threshold

            with open(filepath.replace("filetype", "json"), "w") as f:
                json.dump(params, f)

        if "npz" in save_as:
            np.savez(
                filepath.replace("filetype", "npz"),
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
                                              pcd_expansion_max[1]):
            pcd_expansion_max[:2] = pcd_expansion_max[[1, 0]]
            pcd_arr[:, :2] = pcd_arr[:, [1, 0]]
            # print("Axis swapped!")

        # Normalize the pointcloud to min(pcd) = zeros
        pcd_arr = pcd_arr - np.min(pcd_arr, axis=0)

        if self.conversion_type == "abs":
            # convert to arr, json cant store arrays
            self.expansion_max = np.array(self.expansion_max)
        elif self.conversion_type == "rel":
            self.expansion_max [:2] = pcd_expansion_max.max() * np.ones(shape=(1, 2))

        # Rigid Body transformation -> put the body in the middle of the xy meshgrid
        pcd_arr[:, :2] += (self.expansion_max [:2] - pcd_expansion_max[:2]) / 2

        # Create frame around tooth
        pcd_arr[:, :2] += self.frame_size  # min(pcd) = ones*frame_size now
        self.expansion_max [:2] += 2 * self.frame_size

        # Create Vectors for mesh creation
        x_vec = np.linspace(0, self.expansion_max [0], self.grid_size[0])
        y_vec = np.linspace(0, self.expansion_max [1], self.grid_size[1])

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
        images = images.reshape((
            images.shape[0],
            images.shape[1],
            images.shape[2],
        )).astype(np.uint8)

        # Create the directory with all parents
        if not os.path.exists(self.img_dir_grayscale):
            os.makedirs(self.img_dir_grayscale)

        for img, ctr in zip(
                images,
                tqdm(
                    iterable=range(len(images)),
                    desc="Converting np to grayscale png..",
                    ascii=False,
                    ncols=100,
                ),
        ):
            img_name = os.path.join(self.img_dir_grayscale, f"img_{ctr:04d}_{self.param_hash}.png")
            if not os.path.exists(img_name):
                g_img = PIL.Image.fromarray(img, "L")
                g_img.save(img_name, )

        print("Done.")


    def create_trainingdata(self):
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
                    pcd_arr = self.pcd_to_grid(filepath_stl=filepath_stl,
                                     save_path_pcd=save_path_pcd,
                                     save_path_npy=save_path_npy)

        if not os.path.exists(self.np_savepath):
            self.grid_pcd_to_2D_np()


        # Convert the 2D numpy array to grayscale images for nvidia stylegan
        if not os.path.exists(self.img_dir_grayscale) or len(os.listdir(self.img_dir_grayscale))<self.num_stl:
            # Creating grayscale images
            self.np_grid_to_grayscale_png()
        else:
            print(
                    f"L-PNGs for this parameter-set already exist at: {self.img_dir_grayscale}"
                )

        if not os.path.exists(self.img_dir_rgb) or len(os.listdir(self.img_dir_rgb))<self.num_stl:
            # Converting to rgb and save in different folder
            self.image_conversion_L_RGB(source_img_dir=self.img_dir_grayscale, rgb_dir = self.img_dir_rgb)
        else:
            print(
            f"RGB-PNGs for this parameter-set already exist at: {self.img_dir_rgb}"
            )

        if self.rot_cv: 
            if not os.path.exists(self.img_dir_rgb_cvRot) or len(os.listdir(self.img_dir_rgb_cvRot))<self.num_stl:
                img_paths = sorted(glob.glob(os.path.join(self.img_dir_rgb, "*.png")))
                ip.ImageProps.set_img_dir(self.img_dir_rgb_cvRot)
                for img_path in img_paths:
                    ip.ImagePropsRot(img_path, mode="auto", show_img=True).save_images(img_types="rot")
            else:
                print(
                f"RGB-PNGs with cvRot for this parameter-set already exist at: {self.img_dir_rgb_cvRot}"
                )

    @staticmethod
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



class DataFile(DataCreator):

    def __init__(self, stl_path):
        pass
