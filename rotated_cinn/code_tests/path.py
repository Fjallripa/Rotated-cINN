# Adding this directory to PATH
# =============================
# This is a small support script to make importing modules of this repository more elegant.


import os
import sys


file_directory    = os.path.dirname(os.path.realpath(__file__))
package_directory = os.path.realpath(file_directory + "../../")
sys.path.append(package_directory)
