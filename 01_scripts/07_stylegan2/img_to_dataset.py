import os

img_path = "/home/proj_depo/docker/data/einzelzahn/images"
stylegan_path = "/home/stylegan2-ada"

# Create tfRecord files for every image set: resolution, rgb/L, param_set
for imgdir in [x[0] for x in os.walk(img_path)]:
    if os.path.basename(imgdir) == "img":
        prepdir = os.path.join(os.path.dirname(imgdir), "img_prep")
        if not os.path.exists(prepdir):
            print(f"img_prep-directory: {prepdir}")
            os.system(
                f"python {os.path.join(stylegan_path, 'dataset_tool.py')} create_from_images {prepdir} {imgdir}"
            )
        else:
            print(f"Data for {os.path.dirname(imgdir)} already exists..")
