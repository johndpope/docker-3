import numpy as np
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
import PIL
import glob
from sklearn.datasets import load_digits
import seaborn as sns
import pandas as pd
import cv2 
import sys

sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))
import img_tools.image_processing as ip
# import pcd_tools.data_processing as dp

tsne_bool = 0
corr_bool = 0
ocv_bool = 1
ocv_multi = 0
ocv_single = 1
plt_bool = 0

grid_size = 256
file_num = 10

np_filepath = os.path.join(os.path.dirname(__file__), "data", os.path.basename(__file__).split(".")[0], "mean_mat.npy")

rot_folders = os.listdir(r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\rotated")

img_real_dir = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\images\7a9a625\rgb\{grid_size}x{grid_size}\img"
img_gen_dir = r"C:\Users\bra45451\Desktop\bra45451_offline\03_Design\01_Python\01_Scripts\0_Archiv\05_streamlit\images"
img_real_rot_dir = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\rotated\{rot_folders[0]}\images\7a9a625\rgb\{grid_size}x{grid_size}\img"


img_real_paths = glob.glob(os.path.join(img_real_dir, "*.png"))[:file_num]
# real_labels = np.ones(len(img_real_paths), dtype=int)
real_labels = ["real"]*len(img_real_paths)

img_real_rot_paths = glob.glob(os.path.join(img_real_rot_dir, "*.png"))[:file_num]
# real_rot_labels = np.ones(len(img_real_rot_paths), dtype=int)*2
real_rot_labels = ["real_rot"]*len(img_real_rot_paths)

img_gen_paths = glob.glob(os.path.join(img_gen_dir, "*.png"))[:file_num]
# gen_labels = np.zeros(len(img_real_rot_paths), dtype=int)
gen_labels = ["gen"]*len(img_gen_paths)

mean_mat = np.empty(shape=(len(rot_folders)+2, len(img_real_paths), len(img_real_paths)))

if 0:
    num_img = 8
    img_r_path =  img_real_paths[num_img]
    img_rr_path = img_real_rot_paths[num_img]
    match_dis, matches= ip.matchfinder_bf(img1_path=img_r_path, img2_path=img_rr_path)
    print(np.mean(match_dis))


if not os.path.exists(np_filepath):

    for r_idx, img_r_path in enumerate(img_real_paths):
        for rr_idx, img_rr_path in enumerate(img_real_paths):
            match_dis, matches = ip.matchfinder_bf(img1_path=img_r_path, img2_path=img_rr_path)
            mean_mat[0, r_idx, rr_idx] = np.min(match_dis)
    
    for r_idx, img_r_path in enumerate(img_real_paths):
        for rr_idx, img_rr_path in enumerate(img_gen_paths):
            match_dis, matches = ip.matchfinder_bf(img1_path=img_r_path, img2_path=img_rr_path)
            mean_mat[1, r_idx, rr_idx] = np.min(match_dis)

    if plt_bool:
        plt.figure()
        plt.imshow(mean_mat[0])
        plt.colorbar()
        # sns.clustermap(mean_mat, cmap='viridis', figsize=(8,8))
        plt.show()


    for folder_idx, rot_folder in enumerate(rot_folders):

        img_real_rot_dir = fr"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\rotated\{rot_folder}\images\7a9a625\rgb\{grid_size}x{grid_size}\img"
        
        img_real_rot_paths = glob.glob(os.path.join(img_real_rot_dir, "*.png"))[:file_num]
        # real_rot_labels = np.ones(len(img_real_rot_paths), dtype=int)*2
        real_rot_labels = ["real_rot"]*len(img_real_rot_paths)

        if ocv_bool:
            print("Starting ocv")
            #img = images[0].reshape(grid_size, grid_size)
            # img = cv2.imread(r"G:\docker\bilder\512.jpg", cv2.IMREAD_COLOR)

            if 1: 
                for r_idx, img_r_path in enumerate(img_real_paths):
                    for rr_idx, img_rr_path in enumerate(img_real_rot_paths):
                        match_dis, matches = ip.matchfinder_bf(img1_path=img_r_path, img2_path=img_rr_path)
                        mean_mat[folder_idx+2, r_idx, rr_idx] = np.min(match_dis)
                if plt_bool:
                    plt.figure()
                    plt.imshow(mean_mat[folder_idx+2])
                    plt.colorbar()
                    # sns.clustermap(mean_mat, cmap='viridis', figsize=(8,8))
                    plt.show()


            # # Create img with the matching results from keypoints of both images, flags=2 -> only show keypoints of displayed matches
            # matching_result = cv2.drawMatches(img_r, kp_r, img_rr, kp_rr, matches, None, flags=2)
            # print(len(kp_r))
            # print(len(kp_rr))
            
            # keypoints, decriptors = orb.detectAndCompute(img, None) 
            # img = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            # img = cv2.drawKeypoints(img, kp, None)

            # cv2.imshow("Real", img_r)
            # cv2.imshow("Real rotated", img_rr)
            # cv2.imshow("Matches", matching_result)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    np.save(arr = mean_mat, file =  np_filepath)
else:
    print(f"{np_filepath} already exists.")
    mean_mat = np.load(np_filepath)


mean_mat_diags = [np.diag(ele) for ele in mean_mat]

print(mean_mat_diags)







if 0:
    # img_paths = img_real_paths + img_gen_paths
    img_paths = img_real_paths + img_real_rot_paths

    labels = real_labels + real_rot_labels

    images = np.asarray(PIL.Image.open(img_paths[0]).convert("L")).reshape(1,-1)

    for img_path in img_paths[1:]:
        images = np.concatenate([images, np.asarray(PIL.Image.open(img_path).convert("L")).reshape(1,-1)] , axis=0)

    print(images.shape)

    if tsne_bool:
        print("Starting tsne")
        plt.imshow(images[0, :].reshape((grid_size, grid_size)))
        # Fit t-SNE to data
        tsne = TSNE()
        images_embedded = tsne.fit_transform(images)

        # Compare shape
        print(images.shape)
        print(images_embedded.shape)
        print(images_embedded[0,:])

        # Plot low-dimensional data
        plt.figure()
        sns.scatterplot(x=images_embedded[:, 0], y=images_embedded[:, 1], hue=labels,  legend='full')
        plt.show()


    if corr_bool:
        # Show correlation structure
        print("Starting corr-print")
        # Calculate correlation matrix
        corr_mat = np.corrcoef(images)

        # Plot correlation matrix
        plt.figure()
        plt.imshow(corr_mat)
        plt.colorbar()
        # Plot cluster map
        sns.clustermap(corr_mat, cmap='viridis', figsize=(8,8))
        plt.show()