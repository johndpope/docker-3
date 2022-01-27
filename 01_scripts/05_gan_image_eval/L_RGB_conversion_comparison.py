from pickletools import uint8
import numpy as np
import os
import PIL
import glob
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules"))
import pcd_tools.data_processing as dp

# Paths to images in the same directory as this file
img_real_paths = glob.glob(os.path.join(os.path.dirname(__file__), "real_same", "*.png"))
img_gen_paths =  glob.glob(os.path.join(os.path.dirname(__file__), "gen_same", "*.png"))


# Load the first of the generated images and start conversions
# Attention: Use arr.reshape(-1,3) OR  arr.T.reshape(3,-1)
img = PIL.Image.open(img_gen_paths[0])

images_rgb_no_reshape = np.asarray(img)
images_rgb = np.asarray(img).reshape(-1, 3)
images = np.asarray(img.convert("L")).reshape(-1, 1)
images_L_dp = dp.image_conversion_RGB_L(np.asarray(img), conv_type="luminance_float_exact").reshape(-1, 1)
images_rgb_convert = np.asarray(
    PIL.Image.open(img_gen_paths[0]).convert("L").convert("RGB")).reshape(-1, 3)

images_L_tile = np.tile(images, (1,3))
image_bool_tile = np.any((images_rgb != images_L_tile), axis=1)
image_bool_conv_tile = np.any((images_L_tile != images_rgb_convert), axis=1)

img1 = PIL.Image.open(img_gen_paths[1])

images_rgb_no_reshape_1 = np.asarray(img1)
images_rgb_1 = np.asarray(img1).reshape(-1, 3)
images_1 = np.asarray(img1.convert("L")).reshape(-1, 1)
images_L_dp_1 = dp.image_conversion_RGB_L(np.asarray(img1), conv_type="luminance_float_exact").reshape(-1, 1)
images_rgb_convert_1 = np.asarray(
    img1.convert("L").convert("RGB")).reshape(-1, 3)

images_L_tile_1 = np.tile(images_1, (1,3))
image_bool_tile_1 = np.any((images_rgb_1 != images_L_tile_1), axis=1)
image_bool_conv_tile_1 = np.any((images_L_tile_1 != images_rgb_convert_1), axis=1)

image_bool_01 = np.any((images_rgb != images_rgb_1), axis=1)

where_not_equal_01 = np.where(image_bool_01 == True)[0]

where_not_equal_0 = np.intersect1d(where_not_equal_01, np.where(image_bool_tile == True)[0])
where_not_equal_1 = np.intersect1d(where_not_equal_01, np.where(image_bool_tile_1 == True)[0])

where_not_equal = np.intersect1d(where_not_equal_0, where_not_equal_1)

pixel_nums = where_not_equal

pixel_nums = np.where(image_bool_tile==True)[0]
std_arr = []
lumi_diff_arr = []

for pixel_num in  np.where(image_bool_tile==True)[0]:
    std_arr.append(np.std(images_rgb[pixel_num, :]))
    lumi_diff_arr.append(np.abs((images_rgb[pixel_num, 0]*299/1000 + images_rgb[pixel_num, 1]*587/1000 + images_rgb[pixel_num, 2]*114/1000)-images_rgb[pixel_num, :].max()))

for pixel_num in  np.where(image_bool_tile_1==True)[0]:
    std_arr.append(np.std(images_rgb_1[pixel_num, :]))
    lumi_diff_arr.append(np.abs((images_rgb_1[pixel_num, 0]*299/1000 + images_rgb_1[pixel_num, 1]*587/1000 + images_rgb_1[pixel_num, 2]*114/1000)-images_rgb_1[pixel_num, :].max()))

std_arr = np.asarray(std_arr)
lumi_diff_arr = np.asarray(lumi_diff_arr)

print(lumi_diff_arr.max())
print(std_arr.max())

pixel_num = where_not_equal[1]

print("\n+ ----------------- +\n")
print("Conversion comparison for L to RGB and vice versa:\n")

print(f"(1) Pixel from GAN (RGB): \n{images_rgb[pixel_num, :]}\n")
print(
    f"Calculated grayscale value from RGB with L = R*299/1000 + G*587/1000 + B*114/1000: \
\nfloat: {images_L_dp[pixel_num, 0]} \
\nvalue_L = round(calc_value) NOT int(calc_value)\n"
)
print(f"(2) Pixel (RGB) converted to L: \n{images[pixel_num, :]}\n")

print(f"(3) Pixel (RGB) converted to L and back to RGB: \n{images_rgb_convert[pixel_num, :]}\n")

print(f"Number of Occurences where RGB != [L, L, L]: \n{np.count_nonzero(image_bool_tile)} = {np.count_nonzero(image_bool_tile)/images.shape[0]*100:.02f} %\n")

print(f"Number of Occurences where (3) != [L, L, L]: \n{np.count_nonzero(image_bool_conv_tile)} = {np.count_nonzero(image_bool_conv_tile)/images.shape[0]*100:.02f} %\n")

print("\n+ ----------------- +\n")



print("\n+ ----------------- +\n")
print("Conversion comparison for L to RGB and vice versa:\n")

print(f"(1) Pixel from GAN (RGB): \n{images_rgb_1[pixel_num, :]}\n")
print(
    f"Calculated grayscale value from RGB with L = R*299/1000 + G*587/1000 + B*114/1000: \
\nfloat: {images_L_dp_1[pixel_num, 0]} \
\nvalue_L = round(calc_value) NOT int(calc_value)\n"
)
print(f"(2) Pixel (RGB) converted to L: \n{images_1[pixel_num, :]}\n")

print(f"(3) Pixel (RGB) converted to L and back to RGB: \n{images_rgb_convert_1[pixel_num, :]}\n")

print(f"Number of Occurences where RGB != [L, L, L]: \n{np.count_nonzero(image_bool_tile_1)} = {np.count_nonzero(image_bool_tile_1)/images_1.shape[0]*100:.02f} %\n")

print(f"Number of Occurences where (3) != [L, L, L]: \n{np.count_nonzero(image_bool_conv_tile_1)} = {np.count_nonzero(image_bool_conv_tile_1)/images_1.shape[0]*100:.02f} %\n")

print("\n+ ----------------- +\n")

