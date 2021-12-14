# import yfinance as yf
import streamlit as st
import numpy as np
from PIL import Image
import glob
import os
import open3d as o3d


stl_path = "/home/home_bra"
# npy_file = glob.glob(os.path.join(pcd_path, "*.npy"))[0]

stl_files = glob.glob(os.path.join(stl_path, "*.stl"))

pcd = o3d.io.read_triangle_mesh(stl_files[0]).sample_points_uniformly(
    number_of_points=10000
)

o3d.visualization.draw_geometries([pcd])

pcd_arr = np.asarray(pcd.points)

# # import dnnlib
# # import dnnlib.tflib as tflib


# seeds = range(10)
# print(seeds)
# network_pkl = "/home/proj_depo/docker/models/stylegan2/211210_brecahad-mirror-paper512-ada/results/00002-img_prep-mirror-auto8-kimg1000-ada-resumecustom-freezed0/network-snapshot-000491.pkl"
# truncation_psi = 0.5
# class_idx = None
# dlatents_npz = None

# tflib.init_tf()

# print('Loading networks from "%s"...' % network_pkl)
# with dnnlib.util.open_url(network_pkl) as fp:
#     _G, _D, Gs = pickle.load(fp)

# # # Render images for a given dlatent vector.
# # if dlatents_npz is not None:
# #     print(f'Generating images from dlatents file "{dlatents_npz}"')
# #     dlatents = np.load(dlatents_npz)["dlatents"]
# #     assert dlatents.shape[1:] == (18, 512)  # [N, 18, 512]
# #     imgs = Gs.components.synthesis.run(
# #         dlatents,
# #         output_transform=dict(
# #             func=tflib.convert_images_to_uint8, nchw_to_nhwc=True
# #         ),
# #     )
# #     for i, img in enumerate(imgs):
# #         fname = f"{outdir}/dlatent{i:02d}.png"
# #         print(f"Saved {fname}")
# #         PIL.Image.fromarray(img, "RGB").save(fname)

# # Render images for dlatents initialized from random seeds.
# Gs_kwargs = {
#     "output_transform": dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
#     "randomize_noise": False,
# }
# # if truncation_psi is not None:
# #     Gs_kwargs["truncation_psi"] = truncation_psi

# noise_vars = [
#     var
#     for name, var in Gs.components.synthesis.vars.items()
#     if name.startswith("noise")
# ]

# label = np.zeros([1] + Gs.input_shapes[1][1:])

# # if class_idx is not None:
# #     label[:, class_idx] = 1


# for seed_idx, seed in enumerate(seeds):
#     print("Generating image for seed %d (%d/%d) ..." % (seed, seed_idx, len(seeds)))
#     # seed = 0
#     rnd = np.random.RandomState(seed)
#     z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
#     tflib.set_vars(
#         {var: rnd.randn(*var.shape.as_list()) for var in noise_vars}
#     )  # [height, width]
#     images = Gs.run(z, label, **Gs_kwargs)  # [minibatch, height, width, channel]
#     # bra45451: added squeeze
#     img = PIL.Image.fromarray(images[0].squeeze())
#     st.image(img)
