import numpy as np
import matplotlib.pyplot as plt

filepath = r"P:\MB\Labore\Robotics\019_ChewRob\99_Homes\bra45451\depo\docker\data\einzelzahn\latents\images-4e742fa-abs-keepRatioXY-invertY-rot_3d-full-rot_2d-centered-reduced89\256x256\220312_celebahq-res256-mirror-paper256-kimg100000-ada-target0_5_pyt\00000-img_prep-mirror-paper256-kimg10000-ada-target0.5-bg-resumecustom-freezed0\network-snapshot-000000\latent\img_0089_4e742fa_residual\loss.npz"
loss_file = np.load(filepath)
loss_list = loss_file["loss_list"]
loss = loss_file["loss"]

print(loss)
plt.figure()
plt.plot(np.arange(len(loss_list)), loss_list)
plt.show()