import os
import glob
import shutil

foldername_list = ["Bruecken", "Einzelzaehne"]

for foldername in foldername_list:

    save_pathname = f"G:\\ukr_data\\{foldername}_sorted\\"

    if not os.path.exists(save_pathname):
        os.mkdir(save_pathname)
    else:
        if not int(input(f"Overwrite {foldername}? (1/0)")):
            continue

    list_files = sorted(glob.glob(f"E:\\ukr_data\\{foldername}_unsorted\\*\\*v.stl"))

    # Loop through the dir and copy/rename the files to new path
    for count, filename in enumerate(list_files):
        new = f"{foldername}_{count}.stl"  # new file name
        src = filename  # file source
        dst = os.path.join(save_pathname, new)  # file destination
        shutil.copy(src, dst)

    print(f"Done: {foldername}")
