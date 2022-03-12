import os

stylegan_version = 1
stylegan_versions = ["stylegan2-ada", "stylegan2-ada-pytorch", "stylegan3",]

img_dir_base = "/home/proj_depo/docker/data/einzelzahn/images"
img_folder = "images-4e742fa-abs-keepRatioXY-invertY-rot_3d-full-rot_2d-centered-reduced89"
if img_folder is not None:
    img_dir_base = os.path.join(img_dir_base, img_folder)

stylegan_path = f"/home/{stylegan_versions[stylegan_version]}"

run_mode = "create" # ["create", "remove"]
remove_string = "rotated"
dry_run = False

if dry_run:
    print("\n*-----------*")
    print("DRY RUN")
    print("*-----------*\n")

if run_mode == "create":

    # Create tfRecord files for every image set: resolution, rgb/L, param_set
    for img_dir in [x[0] for x in os.walk(img_dir_base)]:
        if os.path.basename(img_dir) == "img" and not ("generated" in img_dir) and not ("rotated" in img_dir):
                prepdir = os.path.join(os.path.dirname(img_dir), "img_prep")
                if stylegan_version in [1, 2]:
                    prepdir += ".zip"
                    
                if not os.path.exists(prepdir):
                    print(f"\nCreating dataset for images in {img_dir} .. \n")
                    if not dry_run:
                        print(f"img_prep-directory: {prepdir}")
                        if stylegan_version == 0:
                            os.system(
                                f"python {os.path.join(stylegan_path, 'dataset_tool.py')} create_from_images {prepdir} {img_dir}"
                            )
                        elif stylegan_version in [1,2]:
                            os.system(
                            f"python {os.path.join(stylegan_path, 'dataset_tool.py')}  --source={img_dir} --dest={prepdir} "
                            )
                else:
                    print(f"Data for {os.path.dirname(img_dir)} already exists..")

elif run_mode == "remove":

    for img_dir in [x[0] for x in os.walk(img_dir_base)]:
        if os.path.basename(img_dir) == "img_prep" and remove_string in img_dir:    
            print(f"{img_dir} removed. \n")
            if not dry_run:
                os.system(f"rm -rf {img_dir}")

if dry_run:
    print("Dry run finished. No errors.\n")