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

# ------------------------------- #

## Math 
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

# ------------------------------- #

## Processing

def create_param_sha256(frame_size, expansion_max: np.array, nan_val,
                        z_threshold, rot_bb = None, invertY = None):
    """
    Creates Hash of used parameter-set and returns it
    """
    if rot_bb or invertY:
        param_hash = sha256(
        np.concatenate((
            np.array(frame_size).flatten(),
            np.array(expansion_max).flatten(),
            np.array(nan_val).flatten(),
            np.array(z_threshold).flatten(),
            np.array([rot_bb, invertY]),
        )).tobytes()).hexdigest()   
    else:
        param_hash = sha256(
            np.concatenate((
                np.array(frame_size).flatten(),
                np.array(expansion_max).flatten(),
                np.array(nan_val).flatten(),
                np.array(z_threshold).flatten(),
                np.array([rot_bb, ]),
            )).tobytes()).hexdigest()

    return param_hash[::10]


def create_params_cfg(frame_size,
                      expansion_max,
                      nan_val,
                      z_threshold,
                      invertY,
                      keep_xy_ratio,
                      rot_bb,
                      conversion_type,
                      save_as: str = "npz",
                      save_dir: str = []):
    """
    Creates a File containing all parameters.

    name = pcd_to_grid_cfg_{param_hash}.json

    save_dir    : save directory for json File

    save_as     : ["json", "npz"], can be both

    returns all parameters as dict
    """

    if not save_dir:
        if os.name == "nt":
            save_dir = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\cfg"
        elif os.name == "posix":
            save_dir = "/home/proj_depo/docker/data/einzelzahn/cfg"

    print("Creating params cfg..")

    for save_el in save_as:
        if save_el not in ["json", "npz"]:
            raise ValueError('cfg_filetype must be in ["json", "npz"]')

    # Create unique hash for current parameter set
    param_hash = create_param_sha256(frame_size=frame_size,
                                     expansion_max=expansion_max,
                                     nan_val=nan_val,
                                     z_threshold=z_threshold,
                                     rot_bb = rot_bb,
                                     invertY=invertY)

    os.makedirs(save_dir, exist_ok=True)

    filepath = os.path.join(save_dir, f"pcd_to_grid_cfg_{param_hash}.filetype")

    if "json" in save_as:
        # instantiate an empty dict
        params = {}
        params['param_hash'] = param_hash
        params['invertY'] = invertY                      
        params['keep_xy_ratio'] = keep_xy_ratio      
        params['rot_bb'] = rot_bb,        
        params['conversion_type'] = conversion_type              
        params['frame_size'] = frame_size
        params['expansion_max'] = expansion_max.tolist()
        params['x_scale'] = np.round(expansion_max[0] + frame_size * 2,
                                     2).tolist()
        params['y_scale'] = np.round(expansion_max[1] + frame_size * 2,
                                     2).tolist()
        params['nan_val'] = nan_val
        params['z_threshold'] = z_threshold

        with open(filepath.replace("filetype", "json"), "w") as f:
            json.dump(params, f)

    if "npz" in save_as:
        np.savez(
            filepath.replace("filetype", "npz"),
            param_hash=param_hash,
            invertY=invertY,
            keep_xy_ratio=keep_xy_ratio,
            conversion_type=conversion_type,
            frame_size=frame_size,
            expansion_max=expansion_max,
            x_scale=expansion_max[0] + frame_size * 2,
            y_scale=expansion_max[1] + frame_size * 2,
            nan_val=nan_val,
            z_threshold=z_threshold,
        )


def search_pcd_cfg(search_path: str = [],
                   param_hash: str = [],
                   cfg_filetype: str = "npz"):
    """
    searches for pcd_cfg file in search path  

    uses param_hash if given 

    returns the params as npz-archive or dict  

    cfg_filetype    : ["json", "npz"]
    """
    if not search_path:
        if os.name == "nt":
            search_path = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\cfg"
        elif os.name == "posix":
            search_path = "/home/proj_depo/docker/data/einzelzahn/cfg"

    if cfg_filetype not in ["json", "npz"]:
        raise ValueError('cfg_filetype must be in ["json", "npz"]')

    if param_hash:
        # Load pcd_to_grid cfg file
        pcd_to_grid_cfg_path = os.path.join(
            search_path, f"pcd_to_grid_cfg_{param_hash}.{cfg_filetype}")
        if not os.path.exists(pcd_to_grid_cfg_path):
            print("No cfg for param-hash found. \nReturning [].")
            return []
    else:
        pcd_to_grid_cfg_list = glob.glob(
            os.path.join(search_path, f"pcd_to_grid_cfg*.{cfg_filetype}"))
        if len(pcd_to_grid_cfg_list) > 1:
            for num, pathname in enumerate(pcd_to_grid_cfg_list):
                print(f"Index {num}: {os.path.basename(pathname)}")
            pcd_to_grid_cfg_path = pcd_to_grid_cfg_list[int(
                input(f"Enter Index for preferred cfg-File: \n"))]
        elif len(pcd_to_grid_cfg_list) == 1:
            pcd_to_grid_cfg_path = pcd_to_grid_cfg_list[0]

    if cfg_filetype == "npz":
        params = np.load(pcd_to_grid_cfg_path)
    elif cfg_filetype == "json":
        with open(pcd_to_grid_cfg_path) as f:
            params = json.load(f)

    return params

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
     
def pcd_bounding_rot(pcd, onlyZ = False, rotY_noask = True):
    """
    Rotate provided 3D-model depending on bounding-box Rotation.

    """
    import open3d as o3d
    # Rotate
    obb = pcd.get_oriented_bounding_box()
    rot_mat = obb.R.T

    if onlyZ:
        euler_rot_rad = rotationMatrixToEulerAngles(rot_mat)
        R = pcd.get_rotation_matrix_from_xyz((np.asarray([0,0,euler_rot_rad[-1]])))
        pcd = pcd.rotate(R, center=obb.center)
    else:
        pcd = pcd.rotate(rot_mat, center=obb.center)

        x_rot = np.array( ([1, 0, 0], [0, -1, 0], [0, 0, -1]) )[np.newaxis, :,:]
        y_rot = np.array( ([-1, 0, 0], [0, 1, 0], [0, 0, -1]) )[np.newaxis, :,:]
        z_rot = np.array( ([-1, 0, 0], [0, -1, 0], [0, 0, 1]) )[np.newaxis, :,:]
        rots_xyz = np.concatenate([x_rot, y_rot, z_rot], axis=0)

        if rotY_noask:
            pcd = pcd.rotate(rots_xyz[1], center=obb.center)

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=obb.center)

        while 1:
            o3d.visualization.draw_geometries([pcd.sample_points_uniformly(number_of_points=100000), frame], width=720, height=720, window_name=f"Rotated Tooth", left=1600, top=300)
            rot_axis = int(input("\nTurn 180 degress around axis: \n0: x-red \n1: y-green\n2: z-blue \n3: Show Visu again.\n9: break\nUser-Input: "))
            if rot_axis in [0,1,2]:
                rot = rots_xyz[rot_axis]
                pcd = pcd.rotate(rot, center=obb.center)
            elif rot_axis == 3:
                continue
            elif rot_axis == 9:
                break
            else:
                print("Please insert from list [0, 1, 2, 3].")
    return pcd

def prepare_stl(stl_dir: str, stl_new_dir: str = [], rot_type = "full"):
    """
    Loads .stl files from directory, rotates them depending on their bounding-box orientation and saves the new files to stl_new_dir.

    stl_dir:        Directory with .stl files

    stl_new_dir:    New Directory for rotated .stl files

    rot_type:   "full": all axis rotation with user-input,  "z": only z axis

    """
    import open3d as o3d

    # Check function input
    rot_types = ["full", "z"]
    if not rot_type in rot_types:
        raise ValueError(f"rot_type must be in {rot_types}")

    onlyZ = True if rot_type == "z" else False
    filepaths_stl = sorted(glob.glob(os.path.join(stl_dir, "*.stl")))

    if not stl_new_dir:
        if rot_type == "full":
            new_folder =  "rot_bb_full"
        elif rot_type == "z":
            new_folder = "rot_bb_z"
        stl_new_dir = os.path.join(stl_dir, new_folder)

    os.makedirs(stl_new_dir, exist_ok=True)
    if len(glob.glob(os.path.join(stl_new_dir, "*.stl"))) < len(glob.glob(os.path.join(stl_dir, "*.stl"))):
        for filepath_stl in filepaths_stl:
            filepath_stl_new = os.path.join(filepath_stl).replace(stl_dir, stl_new_dir)
            if not os.path.exists(filepath_stl_new):
                pcd_old=o3d.io.read_triangle_mesh(filepath_stl)
                pcd_old.compute_vertex_normals()
                pcd = pcd_bounding_rot(pcd_old, onlyZ=False)
                o3d.io.write_triangle_mesh(filepath_stl_new, pcd)
            else:
                print(f"{os.path.basename(filepath_stl_new)} skipped.")

        print(f"STL-Rotation finished. \nSaved at: {stl_new_dir}")
    else:
        print("Rotated stls already created.")

    return stl_new_dir

def calc_max_expansion(
    load_dir: str,
    z_threshold: float,
    numpoints: int,
    save_dir: str = [],
    save_as: str = "npz",
    rot_bb = False,
    plot_bool: bool = False,
):
    """
    Imports stl-mesh, converts to pointcloud and outputs the max expansion of the model in xyz as expansion_max

    numpoints   : number of initial pointcloud points

    save_as     : ["json", "npz"], can be both

    plot_bool   : plot data?
    """
    import open3d as o3d

    for save_el in save_as:
        if save_el not in ["json", "npz"]:
            raise ValueError('cfg_filetype must be in ["json", "npz"]')

    filepaths_stl = sorted(glob.glob(os.path.join(load_dir, "*.stl")))

    pcd_expansion_max_x = np.array([])
    pcd_expansion_max_y = np.array([])

    for filepath_stl, ctr in zip(
            filepaths_stl,
            tqdm(
                range(len(filepaths_stl)),
                desc="Calculating max expansion of all pcd files..",
                ascii=False,
                ncols=100,
            ),
    ):

        # Import Mesh (stl) and Create PointCloud from cropped mesh
        pcd = o3d.io.read_triangle_mesh(filepath_stl).sample_points_uniformly(
            number_of_points=numpoints)

        pcd_arr = np.asarray(pcd.points)

        # Crop the pcd_arr
        pcd_arr = pcd_arr[pcd_arr[:, 2] > (np.max(pcd_arr, axis=0)[2] -
                                           z_threshold)]

        # Max Expansion after crop
        pcd_expansion_max = np.max(pcd_arr, axis=0) - np.min(pcd_arr, axis=0)

        # Swap x and y axis if max(x)<max(y)
        if pcd_expansion_max[0] < pcd_expansion_max[1]:
            pcd_expansion_max[:2] = pcd_expansion_max[[1, 0]]
            # print("Axis swapped!")

        # First run
        if ctr == 0:
            expansion_max = pcd_expansion_max
        else:
            expansion_max = np.maximum(expansion_max, pcd_expansion_max)
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
        plt.plot(np.arange(len(filepaths_stl)), pcd_expansion_max_x)
        plt.xlabel("Filenumber")
        plt.ylabel("Max x Expansion in mm")

        plt.figure(2)
        plt.rc("font", size=15)
        plt.plot(np.arange(len(filepaths_stl)), pcd_expansion_max_y)
        plt.xlabel("Filenumber")
        plt.ylabel("Max y Expansion in mm")

        plt.show()

    expansion_max = np.round(expansion_max, 1)

    if save_dir and "json" in save_as:
        cfg_name = f"calc_max_expansion_cfg_nump_{numpoints}"

        if rot_bb:
            cfg_name += "_rot"

        # instantiate an empty dict
        params = {}
        params['expansion_max'] = expansion_max.tolist()
        params['numpoints'] = numpoints
        params['z_threshold'] = z_threshold
        params['files'] = filepaths_stl

        with open(
                os.path.join(save_dir,
                             f"{cfg_name}.json"),
                "w") as f:
            json.dump(params, f)

    if save_dir and "npz" in save_as:
        np.savez(os.path.join(save_dir,
                              f"{cfg_name}.npz"),
                 expansion_max=expansion_max,
                 numpoints=numpoints,
                 z_threshold=z_threshold,
                 files=filepaths_stl)

    print("Done.")

    return expansion_max


def pcd_to_grid(
    filepath_stl: str,
    grid_size: np.array,
    expansion_max: np.array,
    z_threshold: float,
    numpoints: int,
    nan_val: int,
    frame_size: float,
    save_path_pcd: str = [],
    save_path_npy: str = [],
    rotation_deg_xyz: np.array = None,
    plot_bool: bool = False,
    invertY: bool = False,
    conversion_type: str = "abs"
):
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
    if conversion_type not in conversion_types:
        raise ValueError(f"conversion type must be in {conversion_types}")

    # Import Mesh (stl) and Create PointCloud from cropped mesh
    pcd = o3d.io.read_triangle_mesh(filepath_stl).sample_points_uniformly(
        number_of_points=numpoints)

    # Execute transformations if specified
    if rotation_deg_xyz is not None:
        rotation_deg_xyz = np.asarray(rotation_deg_xyz)
        pcd.rotate(
            pcd.get_rotation_matrix_from_xyz((rotation_deg_xyz / 180 * np.pi)))

    # Convert Open3D.o3d.geometry.PointCloud to numpy array and get boundaries
    pcd_arr = np.asarray(pcd.points)

    # Crop the pcd_arr
    pcd_arr = pcd_arr[pcd_arr[:,
                              2] > (np.max(pcd_arr, axis=0)[2] - z_threshold)]

    # Max Expansion after crop
    pcd_expansion_max = np.max(pcd_arr, axis=0) - np.min(pcd_arr, axis=0)

    # Swap x and y values if max(x)<max(y)
    if rotation_deg_xyz is None and (pcd_expansion_max[0] <
                                     pcd_expansion_max[1]):
        pcd_expansion_max[:2] = pcd_expansion_max[[1, 0]]
        pcd_arr[:, :2] = pcd_arr[:, [1, 0]]
        # print("Axis swapped!")

    # Normalize the pointcloud to min(pcd) = zeros
    pcd_arr = pcd_arr - np.min(pcd_arr, axis=0)

    if conversion_type == "abs":
        # convert to arr, json cant store arrays
        expansion_max = np.array(expansion_max)
    elif conversion_type == "rel":
        expansion_max[:2] = pcd_expansion_max.max() * np.ones(shape=(1,2))

    # Rigid Body transformation -> put the body in the middle of the xy meshgrid
    pcd_arr[:, :2] += (expansion_max[:2] - pcd_expansion_max[:2]) / 2

    # Create frame around tooth
    pcd_arr[:, :2] += frame_size  # min(pcd) = ones*frame_size now
    expansion_max[:2] += 2 * frame_size

    # Create Vectors for mesh creation
    x_vec = np.linspace(0, expansion_max[0], grid_size[0])
    y_vec = np.linspace(0, expansion_max[1], grid_size[1])

    if invertY:
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
            zvals[ctr] = nan_val

    # Concatenate the xy meshgrid and the zvals from loop
    new_pcd_arr = np.concatenate(
        [points, zvals[:, np.newaxis]], axis=1)

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

def grid_pcd_to_2D_np(
    pcd_dirname: str,
    np_savepath: str,
    grid_size,
    normbounds,
    z_threshold: float,
    nan_val: int,
    pcd_filetype: str = "pcd",
    plot_img_bool: bool = False,
):
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

    if pcd_filetype == "pcd":
        import open3d as o3d

    os.makedirs(os.path.dirname(np_savepath), exist_ok=True)

    print(f"Creating {os.path.basename(np_savepath)}-File for Training..")

    # Get all pcd filepaths
    files = sorted(glob.glob(os.path.join(pcd_dirname, f"*.{pcd_filetype}")))

    # Init the training array
    train_images = np.zeros(
        (len(files), grid_size[0], grid_size[1], 1)).astype("float32")

    # Loop through all files, normalize
    for num, filename in enumerate(files):
        if pcd_filetype == "pcd":
            # Read pcd
            pcd = o3d.io.read_point_cloud(filename)
            # Save as np array
            pcd_arr = np.asarray(pcd.points)
        elif pcd_filetype == "npy":
            pcd_arr = np.load(filename)

        # Reshape as 2D img
        pcd_img = pcd_arr[:, 2].reshape([grid_size[0], grid_size[1], 1])

        # Normalize z values to normbounds
        pcd_img[pcd_img != nan_val] = (pcd_img[pcd_img != nan_val] +
                                       z_threshold * normbounds[0] /
                                       (normbounds[1] - normbounds[0])) / (
                                           z_threshold /
                                           (normbounds[1] - normbounds[0]))

        # Nan values = -1
        pcd_img[pcd_img == nan_val] = normbounds[0]

        # Add to training array
        train_images[num] = pcd_img.reshape([grid_size[0], grid_size[1], 1])

    # Save the training array
    np.save(np_savepath, train_images)

    while plot_img_bool:
        img_number = int(input("Image-Number (from end): "))

        train_image_select = train_images[-img_number]

        # Plot the first img
        plt.figure()
        plt.rc("font", size=15)
        plt.imshow(
            train_image_select,
            cmap="viridis",
        )
        plt.colorbar()
        plt.show()

        if int(input("Exit? (1/0)")):
            break


def np_2D_to_grid_pcd(normbounds,
                      grid_size,
                      z_threshold: float,
                      expansion_max,
                      np_filepath: str = [],
                      np_file: np.array = [],
                      z_crop: float = 0,
                      invertY: bool = False,
                      visu_bool: bool = False,
                      save_pcd: bool = False,
                      save_path_pcd: str = []) -> np.array:
    """
    Load 2D np image (as file or path) and convert to 3D PointCloud with normalization data

    normbounds:             normalization bounds [lower, upper]  

    grid_size

    z_threshold

    expansion_max

    np_filepath:            path to npy file/img  

    np_file:                np.array of img  

    z_crop:                 value for z-crop (crop starts at z=0)  

    invertY:                Invert the y_vec -> if the cartesian point (0,0,0) is at img[-1,0]: lower left corner

    visu_bool:              visualize the img and the 3D obj?  

    save_pcd:           save the pcd (True/False) (default = False)

    save_path_pcd:      save_path for the pcd file, create pcd folder in np_filepath directory and save with same name if not specified and save_pcd =  True

    """
    if save_pcd or save_path_pcd:
        import open3d as o3d

    # Load file with path or file-input
    if np_filepath:
        gen_image = np.load(np_filepath)
    elif np_file.size:
        gen_image = np_file
    else:
        raise ValueError("Input np_filepath or np_file!")

    if gen_image.ndim == 2:
        gen_image = gen_image[:,:,np.newaxis]

    if gen_image.shape[0] == 1:
        print("Transpose")
        gen_image = gen_image.transpose((1, 2, 0))

    # Flatten the 2D array
    z_img = gen_image.reshape((-1, 1))

    # Undo the normalization
    z_img = z_img * z_threshold / (
        normbounds[1] - normbounds[0]) - z_threshold * normbounds[0] / (
            normbounds[1] - normbounds[0])

    # Create Vectors for mesh creation
    x_vec = np.linspace(0, expansion_max[0], grid_size[0])
    y_vec = np.linspace(0, expansion_max[1], grid_size[1])

    if invertY:
        # Invert the y_vec -> if the cartesian point (0,0,0) is at img[-1,0]: lower left corner
        y_vec = y_vec[::-1]

    # Create meshgrid
    [X, Y] = np.meshgrid(x_vec, y_vec)

    # Create points from meshgrid
    xy_points = np.c_[X.ravel(), Y.ravel()]

    # Generate a 3D array with z-values from image and meshgrid
    pcd_gen_arr = np.concatenate([xy_points, z_img], axis=1)

    # Crop the array if needed
    pcd_gen_arr = pcd_gen_arr[pcd_gen_arr[:, 2] >= z_crop]

    # pcdpath = np path if not specified
    if save_pcd and np_filepath and not save_path_pcd:
        save_path_pcd = os.path.join(
            os.path.dirname(np_filepath), "pcd",
            os.path.basename(np_filepath).split(".")[0], ".pcd")
    elif save_pcd and np_file:
        raise ValueError(
            "Specify save_path_pcd if np_filepath is not specified.")

    # Save the new pcd
    if save_path_pcd:
        # Define empty Pointcloud object and occupy with new points
        pcd_gen = o3d.geometry.PointCloud()
        pcd_gen.points = o3d.utility.Vector3dVector(pcd_gen_arr)
        os.makedirs(os.path.dirname(save_path_pcd), exist_ok=True)
        o3d.io.write_point_cloud(save_path_pcd, pcd_gen)

    # Visualize the input img and the 3D-obj
    if visu_bool:
        plt.figure()
        plt.rc("font", size=15)
        plt.imshow(gen_image, cmap="viridis")
        plt.colorbar()
        plt.show()

        o3d.visualization.draw_geometries([pcd_gen])

    return pcd_gen_arr


def np_grid_to_grayscale_png(npy_path: str, img_dir: str, param_hash: str):
    """
    Load npy Array and create grayscale png images  

    npy_path        # Path to npy File  

    img_dir         # saving directory for images  

    param_sha       # Hashed parameter set  
    """
    print(
        f"Creating greyscale images for current parameter-set {param_hash[:10]}.."
    )

    images = np.load(npy_path)

    if images.max() > 1 or images.min() < 0:
        raise ValueError(f"Expected values between 0 and 1. Got values between {images.min()} and {images.max()}")

    images = images * 255
    images = images.reshape((
        images.shape[0],
        images.shape[1],
        images.shape[2],
    )).astype(np.uint8)

    # Create the directory with all parents
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for img, ctr in zip(
            images,
            tqdm(
                iterable=range(len(images)),
                desc="Converting np to grayscale png..",
                ascii=False,
                ncols=100,
            ),
    ):
        img_name = os.path.join(img_dir, f"img_{ctr:04d}_{param_hash}.png")
        if not os.path.exists(img_name):
            g_img = PIL.Image.fromarray(img, "L")
            g_img.save(img_name, )

    print("Done.")


def image_conversion_L_RGB(img_dir: str, rgb_dir: str):
    """
    Load grayscale pngs and convert to rgb  

    Save files with same name to rgb_dir  
    
    img_dir     : directory with grayscale .png images  

    rgb_dir     : directory to save rgb images  
    """
    print('Loading images from "%s"' % img_dir)
    # Create the directory with all parents
    if not os.path.exists(rgb_dir):
        os.makedirs(rgb_dir)

    # Get all .png files
    image_filenames = sorted(glob.glob(os.path.join(img_dir, "*.png")))
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


def img_to_2D_np_single(img_path: str = None,
                        img: np.array = None,
                        np_savepath: str = []) -> np.array:
    """
    Load img and convert to [0,1] normalized stacked np.array 

    save the array if np_savepath is specified  

    returns the img as np.array with shape = (grid_size[0], grid_size[1], 1)  

    img_path    : image path  

    np_savepath : full path for npy file save, leave empty if you dont want to save the file  

    """
    if img is None and img_path is None:
        raise ValueError("Provide either img or img_path!")

    # Load the image
    if img is None:
        img = np.asarray(PIL.Image.open(img_path))

    # Check for right format (RGB/L)
    channels = img.shape[2] if img.ndim == 3 else 1
    if channels not in [1, 3]:
        raise ValueError("Input images must be stored as RGB or grayscale")

    if channels == 3:
        print("RGB-Image")
        # If RGB convert to L and normalize to 0..1
        np_img = image_conversion_RGB_L(
            img, conv_type="luminance_float_exact") / 255
    else:
        print("L-Image")
        # Normalize to 0..1
        np_img = img / 255

    # Save the npy File if np_savepath is specified
    if np_savepath:
        np.save(np_savepath, np_img)
        print(f"Images saved as: {os.path.basename(np_savepath)}")

    return np_img


def img_to_2D_np_multi(img_dir: str,
                       fileextension: str = "png",
                       np_savepath: str = []) -> np.array:
    """
    Load all images in img_dir and convert to one [0,1] normalized stacked np.array  

    save the array if np_savepath is specified  

    returns the images as one np.array with shape = (num_images, grid_size[0], grid_size[1], 1)  

    img_dir     : directory to the images  

    np_savepath : full path for npy file save, leave empty if you dont want to save the file  

    fileextension: specify file extension (default = "png")
    """
    print('Loading images from "%s"' % img_dir)
    image_filenames = sorted(
        glob.glob(os.path.join(img_dir, f"*.{fileextension}")))

    img_init = np.asarray(PIL.Image.open(image_filenames[0]))
    img_shape = img_init.shape
    channels = img_init.shape[2] if img_init.ndim == 3 else 1

    if channels not in [1, 3]:
        raise ValueError("Input images must be stored as RGB or grayscale")

    # Init the training array
    np_img = np.zeros((len(image_filenames), img_shape[0], img_shape[1],
                       1)).astype("float32")

    for filename, num in zip(
            image_filenames,
            tqdm(iterable=range(len(image_filenames)),
                 desc="Converting images to 2D-np Files..",
                 ascii=False,
                 ncols=100)):
        img = np.asarray(PIL.Image.open(filename))

        if img.shape != img_shape:
            print(
                f"Input images must have the same shape. {os.path.basename(filename)} skipped."
            )
        if channels == 3:
            # np_img[num] = (np.asarray(img.convert("L")).reshape(
            #     (img.shape[0], img.shape[1], 1))) / 255
            np_img[num] = (image_conversion_RGB_L(
                img, conv_type="luminance_float_exact") /
                           255)[np.newaxis, :, :]
        else:
            np_img[num] = (np.asarray(img) / 255)[np.newaxis, :, :]

    if np_savepath:
        np.save(np_savepath, np_img)
        print(f"Images saved as: {os.path.basename(np_savepath)}")

    return np_img


def img_to_pcd_single(img_path=None,
                      img=None,
                      cfg_search_dir: str = [],
                      z_crop: float = 0,
                      number_of_points: int = [],
                      param_hash=[],
                      pcd_save: str = False,
                      pcd_outdir: str = [],
                      pcd_name: str = []) -> np.array:
    """
    Creates and returns pcd from img_path or (PIL) img
    """
    if img is None and img_path is None:
        raise ValueError("Provide either img or img_path!")

    if pcd_save or pcd_outdir or pcd_name:
        if not pcd_outdir:
            pcd_outdir = os.path.dirname(img_path)
            os.makedirs(pcd_outdir)
        if not pcd_name:
            pcd_name = os.path.basename(img_path).split(".")[0]

        save_path_pcd = os.path.join(pcd_outdir, pcd_name, ".pcd")
    else:
        save_path_pcd = []

    params = search_pcd_cfg(search_path=cfg_search_dir,
                            param_hash=param_hash,
                            cfg_filetype="npz")

    if img is not None:
        np_img = img_to_2D_np_single(img=img)
    elif img_path is not None:
        np_img = img_to_2D_np_single(img_path=img_path)

    z_threshold = params["z_threshold"]
    expansion_max = params["expansion_max"]

    if "invertY" in params:
        invertY = params["invertY"]
    else:
        invertY = False

    grid_size = np_img.shape[:2]

    if not number_of_points:
        number_of_points = np_img.shape[0] * np_img.shape[1]

    pcd_arr = np_2D_to_grid_pcd(normbounds = [0, 1],
                                grid_size=grid_size,
                                z_threshold=z_threshold,
                                expansion_max=expansion_max,
                                np_file=np_img,
                                invertY=invertY,
                                save_path_pcd=save_path_pcd,
                                z_crop=z_crop)

    return pcd_arr


def img_to_pcd_multi(img_dir,
                     cfg_search_dir: str = [],
                     z_crop: float = 0,
                     number_of_points: int = [],
                     param_hash=[],
                     pcd_save: str = False,
                     pcd_outdir: str = [],
                     pcd_name: str = []) -> np.array:
    """
    Creates and returns pcd from img
    """

    params = search_pcd_cfg(search_path=cfg_search_dir,
                            param_hash=param_hash,
                            cfg_filetype="npz")

    z_threshold = params["z_threshold"]
    expansion_max = params["expansion_max"]

    image_filenames = glob.glob(os.path.join(img_dir, "*.png"))

    for img_path, num in zip(
            image_filenames,
            tqdm(iterable=range(len(image_filenames)),
                 desc=f"Converting {len(image_filenames)} images to pcd..",
                 ascii=False,
                 ncols=100)):
        if pcd_save or pcd_outdir or pcd_name:
            if not pcd_outdir:
                pcd_outdir = os.path.dirname(img_path)
                os.makedirs(pcd_outdir)
            if not pcd_name:
                pcd_name = os.path.basename(img_path).split(".")[0]

            save_path_pcd = os.path.join(pcd_outdir, pcd_name, ".pcd")
        else:
            save_path_pcd = []

        np_img = img_to_2D_np_single(img_path)

        grid_size = np_img.shape[:2]

        if not number_of_points:
            number_of_points = np_img.shape[0] * np_img.shape[1]

        if not num:
            pcd_arr = np_2D_to_grid_pcd(normbounds=[0, 1],
                                        grid_size=grid_size,
                                        z_threshold=z_threshold,
                                        expansion_max=expansion_max,
                                        np_file=np_img,
                                        save_path_pcd=save_path_pcd)
        else:
            pcd_arr = np.concatenate([
                pcd_arr,
                np_2D_to_grid_pcd(normbounds=[0, 1],
                                  grid_size=grid_size,
                                  z_threshold=z_threshold,
                                  expansion_max=expansion_max,
                                  np_file=np_img,
                                  save_path_pcd=save_path_pcd,
                                  z_crop=z_crop)
            ],
                                     axis=0)

    return pcd_arr

def create_trainingdata_full(
    stl_dir, 
    rotation_deg_xyz, 
    invertY, 
    grid_sizes,
    z_threshold, 
    normbounds, 
    frame_size, 
    nan_val, 
    conversion_type, 
    keep_xy_ratio,
    rot_bb,
    pcd_dir, 
    img_dir_base):

    if rot_bb:
        stl_dir = prepare_stl(stl_dir=stl_dir, rot_type = "full")

    for grid_size in grid_sizes:
        print(f"Current Grid-size: {grid_size}")
        numpoints = grid_size[0] * grid_size[1] * 10
        


        num_stl = len(glob.glob(os.path.join(stl_dir, "*.stl")))

        
        if rot_bb:
            cfg_search_hint = f"*{numpoints}_rot*" 
        else:
            cfg_search_hint = f"*{numpoints}*"

        max_exp_cfg_path = glob.glob(os.path.join(stl_dir, cfg_search_hint))

        # Calculate the max expansion of all teeth for normalization
        if not max_exp_cfg_path:
            expansion_max = calc_max_expansion(
                load_dir=stl_dir,
                z_threshold=z_threshold,
                numpoints=numpoints,
                save_dir=stl_dir,
                rot_bb = rot_bb,
                save_as=["json", "npz"]
            )
        else:
            if max_exp_cfg_path[0].split(".")[-1] == "npz":
                expansion_max = np.load(max_exp_cfg_path[0])["expansion_max"]
            elif max_exp_cfg_path[0].split(".")[-1] == "json":
                with open(max_exp_cfg_path[0]) as f:
                    expansion_max = np.array(json.load(f)["expansion_max"])

        if keep_xy_ratio:
            expansion_max[:2] = expansion_max[:2].max()* np.ones(shape=(1,2)) 

        param_hash = create_param_sha256(frame_size=frame_size,
                                            expansion_max=expansion_max,
                                            nan_val=nan_val,
                                            z_threshold=z_threshold,
                                            rot_bb = rot_bb,
                                            invertY = invertY) 

        print(f"Param-Hash: {param_hash}")

        grid_folder = f"{grid_size[0]}x{grid_size[1]}"

        foldername = param_hash

        if conversion_type:
            foldername += "-" + conversion_type

        if keep_xy_ratio:
            foldername += "-keepRatioXY"

        if invertY:
            foldername += "-invertY"
        
        if rot_bb:
            foldername += "-rot_bb"

        if rotation_deg_xyz is not None:
            rot_folder = f"-rotated_x{rotation_deg_xyz[0]:02d}_y{rotation_deg_xyz[1]:02d}_z{rotation_deg_xyz[2]:02d}"
            foldername += rot_folder
  
        # Paths
        save_dir_base = os.path.join(pcd_dir, f"pcd-{foldername}", grid_folder)

        pcd_grid_save_dir = os.path.join(save_dir_base, "pcd_grid")
        npy_grid_save_dir = os.path.join(save_dir_base, "npy_grid")
 
        img_dir = os.path.join(img_dir_base, f"images-{foldername}", "grayscale",
                            grid_folder, "img")

        img_dir_rgb = img_dir.replace("grayscale", "rgb")

        np_savepath = os.path.join(
            save_dir_base,
            f"einzelzaehne_train_lb{normbounds[0]}_ub{normbounds[1]}_{param_hash}.npy",
        )

        params = search_pcd_cfg(param_hash=param_hash)

        # Load the unregular .pcd files and save them as regularized grid pcds
        if not os.path.exists(pcd_grid_save_dir) or len(os.listdir(pcd_grid_save_dir))<num_stl \
        or not os.path.exists(npy_grid_save_dir) or len(os.listdir(npy_grid_save_dir))<num_stl:
            params = create_params_cfg(frame_size=frame_size,
                                        expansion_max=expansion_max,
                                        nan_val=nan_val,
                                        z_threshold=z_threshold, 
                                        invertY=invertY,
                                        keep_xy_ratio=keep_xy_ratio,
                                        rot_bb = rot_bb,
                                        conversion_type=conversion_type,
                                        save_as=["json", "npz"])

            files = glob.glob(os.path.join(stl_dir, "*.stl"))
            for filename, num in zip(
                    files,
                    tqdm(
                        range(len(files)),
                        desc="Creating pcd-grid files..",
                        ascii=False,
                        ncols=100,
                    ),
            ):
                save_path_pcd = os.path.join(
                        pcd_grid_save_dir, f"einzelzahn_grid_{num:04d}_{param_hash}.pcd")
                save_path_npy = os.path.join(
                        npy_grid_save_dir, f"einzelzahn_grid_{num:04d}_{param_hash}.npy")
                if not os.path.exists(save_path_pcd) or not os.path.exists(save_path_npy):
                    pcd_to_grid(
                        filepath_stl=filename,
                        save_path_pcd=save_path_pcd,
                        save_path_npy=save_path_npy,
                        grid_size=grid_size,
                        expansion_max=expansion_max,
                        frame_size=frame_size,
                        nan_val=nan_val,
                        plot_bool=False,
                        numpoints=numpoints,
                        z_threshold=z_threshold,
                        rotation_deg_xyz=rotation_deg_xyz,
                        invertY= invertY,
                        conversion_type=conversion_type
                    )

        # Convert the 3D regularized grid pcds to one 2D numpy array for training
        if not os.path.exists(np_savepath):
            grid_pcd_to_2D_np(
                pcd_dirname=pcd_grid_save_dir,
                np_savepath=np_savepath,
                grid_size=grid_size,
                z_threshold=z_threshold,
                nan_val=nan_val,
                normbounds=normbounds,
            )

        # Convert the 2D numpy array to grayscale images for nvidia stylegan
        if not os.path.exists(img_dir) or len(os.listdir(img_dir))<num_stl:
            # Creating grayscale images
            np_grid_to_grayscale_png(npy_path=np_savepath,
                                        img_dir=img_dir,
                                        param_hash=param_hash)
        else:
            print(
                    f"L-PNGs for this parameter-set already exist at: {img_dir}"
                )

        if not os.path.exists(img_dir_rgb) or len(os.listdir(img_dir_rgb))<num_stl:
            # Converting to rgb and save in different folder
            image_conversion_L_RGB(img_dir=img_dir, rgb_dir=img_dir_rgb)
        else:
            print(
            f"RGB-PNGs for this parameter-set already exist at: {img_dir_rgb}"
            )
            