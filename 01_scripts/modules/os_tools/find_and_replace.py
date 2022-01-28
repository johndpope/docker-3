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


def find_and_replace(old_string: str,
                     new_string: str,
                     search_dir: str,
                     search_condition: str,
                     replace_count: int = -1,
                     recursive: bool = True):
    """
    Recursively (default) finds and replaces all occurences (default) of old_string with new_string in all files that match the search_condition in the search_dir

    if recursive == False: os.path.join(search_dir, "folder", "file.py") won't be found.

    if replace_count == 1: Only the first (etc.) occurence of old_string in every file will be replaced, -1 (default) replaces all occurences

    Examples for search_condition:

    All python files: "*.py"

    All files that contain the word "test": "*test*"


    """
    # Get a list of all files which match the search_condition criteria
    files = glob.glob(os.path.join(search_dir, "**", search_condition),
                      recursive=recursive)

    # Init empty list for relevant files
    replace_file_list = []

    for file in files:
        # Exclude this file and all folders. Only open files
        if not os.path.samefile(file, __file__) and os.path.isfile(file):
            with open(file) as r:
                # Get all text
                text = r.read()
                if old_string in text:
                    # Append the file to list if it contains old_string
                    replace_file_list.append(file)
                    
    if not replace_file_list:
        print(f"No files containing <{old_string}> were found.")
    else:
        print(
            f"{old_string} was found in the following {len(replace_file_list)} files: \n"
        )
        [print(f"{ctr}: {file}") for ctr, file in enumerate(replace_file_list)]

        if _ask_yes_no(f"\nContinue?\n"):
            # Possibility to skip indices as long as replace_file_list is not empty
            while replace_file_list:
                if _ask_yes_no("\nExclude files? See indices above."):
                    replace_file_list.pop(int(input("Skip Index: \n")))
                    [
                        print(f"{ctr}: {file}")
                        for ctr, file in enumerate(replace_file_list)
                    ]
                else:
                    break

            # Open all the remaining files in replace_file_list
            for file in replace_file_list:
                with open(file) as r:
                    text = r.read()
                # Replace all replace_count occurences of old_string with new_string
                text_new = text.replace(old_string,
                                        new_string,
                                        replace_count)
                with open(file, "w") as w:
                    w.write(text_new)

            print(f"{len(replace_file_list)} files edited.")
        else:
            print("Aborted.")



if __name__ == "__main__":
    old_string = 'sys.path.append(os.path.abspath(".\modules"))'
    new_string = 'sys.path.append(os.path.join(os.path.dirname(__file__).split("01_scripts")[0], "01_scripts", "modules"))'
    search_dir = r"G:\docker\01_scripts"
    search_condition = "*.py"
    find_and_replace(old_string=old_string,
                     new_string=new_string,
                     search_dir=search_dir,
                     search_condition=search_condition,
                     replace_count=-1,
                     recursive=True)
