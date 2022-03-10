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

def init_network(network_pkl):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as fp:
        _G, _D, Gs = pickle.load(fp)

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    return Gs, noise_vars


#----------------------------------------------------------------------------
#def generate_image(Gs, Gs_kwargs, label, seed=None, outdir=None, dlatents_npz=None, dlatents = None, img_as_pil=False, truncation_psi=0.5, noise_var_seed = 0):
def generate_image(Gs, noise_vars, seed=None, outdir=None, dlatents_npz=None, dlatents = None, img_as_pil=False, truncation_psi=0.5, noise_var_seed = 0):
    """
    dlatents shapes:

    1024x1024:   (1, 18, 512)
    512x512:     (1, 16, 512)
    256x256:     (1, 14, 512)        
    """
    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)

    # Render images for a given dlatent vector.
    if dlatents_npz is not None or dlatents is not None:
        
        if dlatents_npz is not None:
            print(f'Generating images from dlatents file "{dlatents_npz}"')
            dlatents = np.load(dlatents_npz)['dlatents']
        else:
            print('Generating images from dlatents.')

        assert dlatents.shape[-1] == 512 
        assert dlatents.shape[1] == 18 or dlatents.shape[1] == 16 or dlatents.shape[1] == 14

        img = Gs.components.synthesis.run(dlatents, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
        if outdir is not None:
            fname = f'{outdir}/dlatent{0:02d}.png'
            print (f'Saved {fname}')
            PIL.Image.fromarray(img[0], 'RGB').save(fname)

        if img_as_pil:
            return PIL.Image.fromarray(img[0], 'RGB')    
        else:
            return img[0]

    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    if truncation_psi is not None:
        Gs_kwargs['truncation_psi'] = truncation_psi

    # Set the noise_vars
    # Changing these values will not substantially change a particular vector’s output image
    rnd = np.random.RandomState(noise_var_seed)
    
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
    label = np.zeros([1] + Gs.input_shapes[1][1:])

    print('Generating image for seed %d...' % (seed))
    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
    image = Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel]
    if outdir is not None:
        PIL.Image.fromarray(image[0], 'RGB').save(os.path.join(outdir, f"seed{seed:04d}.png"))
    if img_as_pil:
        return PIL.Image.fromarray(image[0], 'RGB')    
    else:
        return image[0]

#----------------------------------------------------------------------------