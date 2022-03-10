import numpy as np
import os
import PIL
import scipy
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pickle
from hashlib import sha256

def network_hash(stylegan_folder, kimg_str, run_folder, network_pkl):
    """
    Creates unique hash (10 chars) for current config

    kimg_str: f"kimg{kimg_num:04d}"

    network_pkl without .pkl

    """
    return sha256((stylegan_folder+kimg_str+run_folder+network_pkl).encode()).hexdigest()[::10]

def feature_net_calc( img_paths, feature_net, feat_path=None):
    # Load existing datasets or create features for reals, fakes and rots
    if feat_path is not None and os.path.exists(feat_path):
        print(f"Loading from {feat_path} ..")
        features = np.load(feat_path)
    else:
        imgs = np.asarray(PIL.Image.open(img_paths[0]))[np.newaxis, :]
        for img_path in img_paths[1:]:
            img = np.asarray(PIL.Image.open(img_path))[np.newaxis, :]
            imgs = np.concatenate([imgs, img], axis=0)
        # Transpose for feature net
        imgs = imgs.transpose((0, 3, 1, 2))
        # Create batches
        features = np.empty(shape=(0, 2048))
        for img_batch in np.array_split(imgs, np.ceil(imgs.shape[0] / 92)):
            features = np.concatenate(
                [features,
                 feature_net.run(img_batch, assume_frozen=True)],
                axis=0)
        if feat_path is not None:
            # Save the features as npy
            np.save(feat_path, features)
    return features


def fit_tsne(feat1, label1, feat2=None, label2=None, plt_bool=False, fig_path=None, tsne_metric = "correlation" ):
    print("Starting tsne")

    if feat2 is not None:
        features = np.concatenate([feat1, feat2], axis=0)
        labels = label1 + label2
    else:
        features = feat1
        labels = label1

    # tsne_metric =  "correlation"  
    print(f"Tsne Metric: {tsne_metric}")  

    # Fit t-SNE to data
    tsne = TSNE(n_components=2,
                perplexity=10,
                metric=tsne_metric,
                method="barnes_hut",
                random_state=0,
                init='random',
                square_distances=True)

    features_embedded = tsne.fit_transform(features)
    if plt_bool:
        fig_obj = plt.figure()
        sns.scatterplot(x=features_embedded[:, 0],
                        y=features_embedded[:, 1],
                        hue=labels,
                        legend='full',
                        palette='colorblind')
        if fig_path is not None:
            pickle.dump(fig_obj,open(fig_path,'wb'))    

    return features_embedded



def kdtree_query_ball_tree( feat1, 
                            feat2, 
                            img_1_paths, 
                            img_2_paths, 
                            label1,
                            label2, 
                            max_distance, 
                            plt_bool=False):

    features_embedded = fit_tsne(feat1=feat1,
                                 feat2=feat2,
                                 label1=label1,
                                 label2=label2)

    kdtree1 = scipy.spatial.KDTree(features_embedded[:feat1.shape[0]])
    kdtree2 = scipy.spatial.KDTree(features_embedded[feat1.shape[0]:])

    neighbours = kdtree1.query_ball_tree(kdtree2, r=max_distance)
    feat1_feat2_pairs = []

    for ctr, neighbour in enumerate(neighbours):
        if neighbour:
            for neighbour_sub in neighbour:
                feat1_feat2_pairs.append([ctr, neighbour_sub])

    print(f"Number of nearest neighbours: {len(feat1_feat2_pairs)}")
    if plt_bool:
        for feat1_feat2_pair in feat1_feat2_pairs[:100]:
            img1 = PIL.Image.open(img_1_paths[feat1_feat2_pair[0]])
            img2 = PIL.Image.open(img_2_paths[feat1_feat2_pair[1]])

            fig, (ax1, ax2) = plt.subplots(1, 2)
            print([label1[feat1_feat2_pair[0]], label2[feat1_feat2_pair[1]]])
            print([
                os.path.basename(
                    img_1_paths[feat1_feat2_pair[0]]).split(".")[0],
                os.path.basename(
                    img_2_paths[feat1_feat2_pair[1]]).split(".")[0]
            ])
            ax1.imshow(img1)
            ax2.imshow(img2)
            plt.show()

    return feat1_feat2_pairs


## Error metrics
def compute_kid(feat_real, feat_fake, num_subsets=1000, max_subset_size=1000):
    n = feat_real.shape[1]
    m = min(min(feat_real.shape[0], feat_fake.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = feat_fake[np.random.choice(feat_fake.shape[0], m, replace=False)]
        y = feat_real[np.random.choice(feat_real.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1)**3 + (y @ y.T / n + 1)**3
        b = (x @ y.T / n + 1)**3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    return t / num_subsets / m


def compute_fid(feat_real, feat_fake):
    mu_fake = np.mean(feat_fake, axis=0)
    sigma_fake = np.cov(feat_fake, rowvar=False)
    mu_real = np.mean(feat_real, axis=0)
    sigma_real = np.cov(feat_real, rowvar=False)

    # Calculate FID.
    m = np.square(mu_fake - mu_real).sum()
    s, _ = scipy.linalg.sqrtm(np.dot(sigma_fake, sigma_real), disp=False)  # pylint: disable=no-member
    dist = m + np.trace(sigma_fake + sigma_real - 2 * s)

    return np.real(dist)