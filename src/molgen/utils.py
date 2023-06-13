import os
import re

from src.molgen.constants import PROJECT_NAME


def get_root_directory() -> str:
    """
    Returns the root directory of the project.
    """
    current_dir = os.path.dirname(__file__)
    current_dir_list = re.split(r"[\\/]", current_dir)
    find_project_folder = [bool(i.lower() == PROJECT_NAME.lower()) for i in current_dir_list]
    # Find the first instance of the project name in the directory where the directory has a docs folder
    for i, x in enumerate(find_project_folder):
        if x:
            path = "/".join(current_dir_list[: i + 1])
            if os.path.exists(os.path.join(path, "docs")):
                root_dir = path
                break

    return root_dir
