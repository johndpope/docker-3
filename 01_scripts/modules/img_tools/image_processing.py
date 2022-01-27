import cv2
import numpy as np
from tqdm import tqdm


def matchfinder_bf(img1_path, img2_path):
    img_r = cv2.imread(img1_path, cv2.IMREAD_COLOR)
    img_rr = cv2.imread(img2_path, cv2.IMREAD_COLOR)

    orb = cv2.ORB_create()
    orb.setEdgeThreshold(0)
    orb.setMaxFeatures(10000)
    orb.setNLevels(20)

    kp_r, des_r = orb.detectAndCompute(img_r, None)
    kp_rr, des_rr = orb.detectAndCompute(img_rr, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des_r, des_rr)
    matches = sorted(matches, key=lambda x: x.distance)

    # matches = [m for m in matches if m.distance <= 10]
    match_dis = [m.distance for m in matches]

    if not match_dis:
        match_dis = 100

    return match_dis, matches


# # From stylegan2-ada metrics/kernel_inception_distance.py
# def compute_kid(feat_real, feat_fake, num_subsets=100, max_subset_size=1000):
#     n = feat_real.shape[1]
#     m = min(min(feat_real.shape[0], feat_fake.shape[0]), max_subset_size)
#     t = 0
#     for _subset_idx in range(num_subsets):
#         x = feat_fake[np.random.choice(feat_fake.shape[0], m, replace=False)]
#         y = feat_real[np.random.choice(feat_real.shape[0], m, replace=False)]
#         a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
#         b = (x @ y.T / n + 1) ** 3
#         t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
#     return t / num_subsets / m


# From stylegan2-ada metrics/kernel_inception_distance.py
def compute_kid(feat_real, feat_fake, num_subsets=100, max_subset_size=1000):
    n = feat_real.shape[1]
    m = min(min(feat_real.shape[0], feat_fake.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in tqdm(
            range(num_subsets),
            desc="Calculating KID",
            ascii=False,
            ncols=num_subsets,
    ):
        x = feat_fake[np.random.choice(feat_fake.shape[0], m, replace=False)]
        y = feat_real[np.random.choice(feat_real.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1)**3 + (y @ y.T / n + 1)**3
        b = (x @ y.T / n + 1)**3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    return t / num_subsets / m
