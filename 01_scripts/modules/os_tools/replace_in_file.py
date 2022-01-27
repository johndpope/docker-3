import os
import glob
from distutils.util import strtobool

def _ask_yes_no(question: str) -> bool:
    """Ask the user the question until the user inputs a valid answer."""
    while True:
        try:
            print("{0} [y/n]".format(question))
            return strtobool(input().lower())
        except ValueError:
            pass

def find_and_replace(old_string: str, new_string: str, search_dir: str, search_files: str):
    """
    Recursively finds and replaces all occurences of old_string with new_string in all specified files in the search_dir

    Examples for search_files:

    All python files: "*.py"

    All files that contain the word "test": "*test*"


    """
    files = glob.glob(os.path.join(search_dir, "**", search_files), recursive=True)
    ctr = 0
    replace_file_list = []

    for file in files:
        if file != __file__ and os.path.isfile(file):
            with open(file) as r:
                text = r.read()
                if old_string in text:
                    ctr += 1
                    replace_file_list.append(file)

    print(f"\n{ctr} occurences of {old_string} were found in the following files: \n")
    [print(f"{ctr}: {file}") for ctr, file in enumerate(replace_file_list)]

    if _ask_yes_no(f"\nContinue?\n"):
        while replace_file_list:
            if _ask_yes_no("\nExclude files? See indices above."):
                replace_file_list.pop(int(input("Skip Index: \n")))
                [print(f"{ctr}: {file}") for ctr, file in enumerate(replace_file_list)]
            else:
                break  

        for file in replace_file_list:
            with open(file) as r:
                text = r.read()
            text_new = text.replace(old_string, new_string)
            with open(file, "w") as w:
                w.write(text_new)

        print(f"{len(replace_file_list)} files edited.")

    else:
        print("Aborted.")


if __name__ == "__main__":
    new_string = 'sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "modules"))'
    old_string = 'sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))'
    search_dir = r"C:\Users\bra45451\Desktop\Neuer Ordner (3)"
    search_files = "*.py"
    find_and_replace(old_string, new_string, search_dir, search_files)