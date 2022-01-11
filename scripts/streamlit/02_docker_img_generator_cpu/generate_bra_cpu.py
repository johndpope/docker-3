# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import argparse
import os
import pickle
import re

import numpy as np
import PIL.Image

import dnnlib
import dnnlib.tflib as tflib

#----------------------------------------------------------------------------

def init_network(network_pkl, cache_dir=None, truncation_psi=0.5, noise_var_seed = 0, outdir=None):
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    tflib.init_tf()
    res = resolution = 512
    cfg="paper512"
    num_channels=3
    label_size=0
    
    cfg_specs = {
        'auto':          dict(ref_gpus=-1, kimg=25000,  mb=-1, mbstd=-1, fmaps=-1,  lrate=-1,     gamma=-1,   ema=-1,  ramp=0.05, map=2), # populated dynamically based on 'gpus' and 'res'
        'stylegan2':     dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=10,   ema=10,  ramp=None, map=8), # uses mixed-precision, unlike original StyleGAN2
        'paper256':      dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=0.5, lrate=0.0025, gamma=1,    ema=20,  ramp=None, map=8),
        'paper512':      dict(ref_gpus=8,  kimg=25000,  mb=64, mbstd=8,  fmaps=1,   lrate=0.0025, gamma=0.5,  ema=20,  ramp=None, map=8),
        'paper1024':     dict(ref_gpus=8,  kimg=25000,  mb=32, mbstd=4,  fmaps=1,   lrate=0.002,  gamma=2,    ema=10,  ramp=None, map=8),
        'cifar':         dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=0.5, lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=2),
        'cifarbaseline': dict(ref_gpus=2,  kimg=100000, mb=64, mbstd=32, fmaps=0.5, lrate=0.0025, gamma=0.01, ema=500, ramp=0.05, map=8),
    }
    spec = dnnlib.EasyDict(cfg_specs[cfg])
    if cfg == 'auto':
        spec.fmaps = 1 if res >= 512 else 0.5
    G_args = dnnlib.EasyDict(func_name='training.networks.G_main')
    G_args.fmap_base = int(spec.fmaps * 16384)
    G_args.fmap_max = 512
    G_args.mapping_layers = spec.map
    G_args.num_fp16_res = 4
    G_args.conv_clamp = 256
    G = tflib.Network('G', num_channels=num_channels, resolution=resolution, label_size=label_size, **G_args)
    Gs = G.clone('Gs')
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl, cache_dir) as fp:
        _G, _D, rGs = pickle.load(fp)
    Gs.copy_vars_from(rGs)   
     
    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    # Set the noise_vars
    # Changing these values will not substantially change a particular vector’s output image
    rnd = np.random.RandomState(noise_var_seed)
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]

    label = np.zeros([1] + Gs.input_shapes[1][1:])

    return Gs, Gs_kwargs, label

#----------------------------------------------------------------------------

def generate_image(Gs, Gs_kwargs, label, seed, outdir=None, dlatents_npz=None):

    # Render images for a given dlatent vector.
    if dlatents_npz is not None:
        print(f'Generating images from dlatents file "{dlatents_npz}"')
        dlatents = np.load(dlatents_npz)['dlatents']
        # bra // concat first 4 elements again, projector output: dlatents.shape = (1, 14, 512): dlatents_npz contains 14x the same 1,512 vector
        if dlatents.shape[1] != 18:      
            print(f"dlatents.shape[1] changed from {dlatents.shape[1]} to 18.")
            dlatents = np.concatenate([dlatents, dlatents[:,:4,:]], axis=1) 
        #    
        assert dlatents.shape[1:] == (18, 512) # [N, 18, 512]
        img = Gs.components.synthesis.run(dlatents, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
        if outdir is not None:
            fname = f'{outdir}/dlatent{0:02d}.png'
            print (f'Saved {fname}')
            PIL.Image.fromarray(img, 'RGB').save(fname)
        return img

    print('Generating image for seed %d...' % (seed))
    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
    
    image = Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel]
    if outdir is not None:
        PIL.Image.fromarray(image[0], 'RGB').save(f'{outdir}/seed{seed:04d}.png')
    return image[0]

#----------------------------------------------------------------------------