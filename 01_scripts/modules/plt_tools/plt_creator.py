# Normalize with own fun
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import pickle

def plot_init():
    plt.rcParams['text.usetex'] = True
    plt.rcParams.update({'font.size': 12})

## compare_latent_residual_images.py
def normalize_01(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

def plot_metrics(x, y, xlabel, ylabel,  stylegan_folder, image_folder, kimg, run_folder, grid_size, fig_base_dir, network_pkl = None, vline_value=None, legend_name=None, normalize_y = True, fig_folder=None, master_filepath = None, save_fig=True):
    """
    Creates Plot with static x axis and an arbitrary number of y-values

    normalize_y = True: y-values can all be normalized to [0,1]
    save_fig = True:    saves the fig as pdf, the pickled fig obj, the 

    """
    plot_init()

    if fig_folder is None and save_fig:
        raise ValueError("Provide fig_folder in order to save the figures.")
    if master_filepath is None and save_fig:
        raise ValueError("Provide master_filepath (__file__) in order to save the figures.")

    if not isinstance(y, list):
        y = [y]

    if legend_name is not None:
        if len(y) != len(legend_name):
            raise ValueError("Legend entries have to match the y-values.")

    fig_obj = plt.figure()

    if vline_value is not None:
        plt.axvline(x=vline_value, color='r', linestyle='--')

    for y_item in y: 
        if normalize_y:
            y_item = normalize_01(y_item[:, np.newaxis])
        plt.plot(x, y_item)

    if legend_name is not None:
        plt.legend(legend_name)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    if save_fig:
        fig_name = f"{stylegan_folder.split('_')[0]}_im-{image_folder.split('-')[1]}_{kimg}_{run_folder.split('-')[0]}"
        if network_pkl is not None:
            fig_name += f"_{network_pkl}"

        fig_dir = os.path.join(fig_base_dir, fig_folder, fig_name)
        os.makedirs(fig_dir, exist_ok=True)
        fig_path_base = os.path.join(fig_dir, fig_name)

        shutil.copy(master_filepath, fig_path_base+".py")
        shutil.copy(__file__, fig_path_base+"_module.py")

        with open(fig_path_base + ".txt", "w") as f:
            f.write(f"legend_name: {legend_name}\n")
            f.write(f"grid_size: {grid_size}\n")
            f.write(f"image_folder: {image_folder}\n")
            f.write(f"stylegan_folder: {stylegan_folder}\n")
            f.write(f"run_folder: {run_folder}\n")
            if network_pkl is not None:
                f.write(f"network_pkl: {network_pkl}\n")

        
        with open(fig_path_base + ".pickle",'wb') as f:
            pickle.dump(fig_obj, f)  

        plt.savefig(fig_path_base + ".pdf")

    plt.show()

