# Adding this directory to PATH
# =============================
# This is a small support script to make importing modules of this repository more elegant.


import os
import sys


def makedir(dir_path:str) -> None:
    """
    os.makedirs() is too annoying to use in scripts.
    This small function doesn't raise an error if the path already exists.
    """

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


file_directory    = os.path.dirname(os.path.realpath(__file__))
package_directory = os.path.realpath(file_directory + "/..")

sys.path.append(package_directory)
