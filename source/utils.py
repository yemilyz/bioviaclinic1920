import os
import glob

def get_filepaths(pattern = ""):
    repo_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..',))
    files = []
    for root, dirs, _ in os.walk(os.path.join(repo_dir, 'sabdab_dataset'), topdown=False):
        for name in dirs:
            files += glob.glob(os.path.join(root, name, pattern))
    return files
