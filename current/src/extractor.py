import os
import fnmatch
import pandas as pd
import numpy as np
import shutil
import gzip
from tqdm import tqdm

dir = os.getcwd()
filter_list = pd.read_csv(dir + "/../results/pruned_data.csv")
## IMPORTANT COMMENT: PLEASE READ BEFORE RUNNING EXTRACTOR!!!
## STEPS TO ENSURE SUCCESSFUL RUN OF extractor.py
# 1 - Download Tommi's dataset stored in Google Drive, place it in your repo in current/data/
# 2 - Rename Tommi's databank folder as "mri". This is important, else 'mri_path' becomes invalid
# 3 - The .gitignore has been updated to ignore Tommi's databank, given you named it properly by following step 2
# 4 - Check imports list, make sure you have all the dependencies installed
# 5 - Now run the program
# P.S. the directories are named to follow Unix/Mac directory naming. If you're using Windows, too bad. :shrug:
mri_path = dir + "/../data/mri/"
subfolders = [f.path for f in os.scandir(mri_path) if f.is_dir()]


# get desired sub dirs from total list of subfolders
def extract_required_paths(subfolders):
    print('Pruning path lists')
    nifti_paths = []
    size = len(subfolders)
    for index, row in filter_list.iterrows():
        dir_check = mri_path + row['MR ID']
        for i in range(size):
            if fnmatch.fnmatch(subfolders[i], dir_check):
                nifti_paths.append(subfolders[i])
                break
    return nifti_paths


# extract onii-sama from .gz into subdir properly UwU
def extract_onii_chan(subfolders):
    print('Extracting nii-sama')
    for i in tqdm(range(len(subfolders))):
        more_subfolders = [f.path for f in os.scandir(subfolders[i]) if f.is_dir()]
        for j in tqdm(range(len(more_subfolders))):
            gz_file = os.listdir(more_subfolders[j] + "/NIFTI")  # this is a list
            gz_filepath = more_subfolders[j] + "/NIFTI/" + gz_file[0]
            # Python 3.9 required for .removesuffix - alternative provided in lines 47 - 49
            # target_path = subfolders[i] + "/" + gz_file[0].removesuffix('.gz')
            target_path = subfolders[i] + "/" + gz_file[0]
            """
            - Unzip was creating errors with Nibabel, use compressed nifti files instead
            if target_path.endswith('.gz'):
                target_path = target_path[:-3]
            unzip(gz_filepath, target_path)
            """
            move(gz_filepath, target_path)
    return


# helper unzip function
def unzip(source_filepath, dest_filepath):
    block_size = 65536
    f = open(source_filepath, 'rb')
    if f.read(2) == '\x1f\x8b':
        f.seek(0)
        with gzip.open(source_filepath, 'rb') as s_file, \
                open(dest_filepath, 'wb') as d_file:
            shutil.copyfileobj(s_file, d_file, block_size)
    else:
        f.seek(0)
        with open(source_filepath, 'rb') as s_file, \
                open(dest_filepath, 'wb') as d_file:
            shutil.copyfileobj(s_file, d_file, block_size)
    f.close()


# helper unzip function
def move(source_filepath, dest_filepath):
    block_size = 65536
    f = open(source_filepath, 'rb')
    f.seek(0)
    with open(source_filepath, 'rb') as s_file, \
            open(dest_filepath, 'wb') as d_file:
        shutil.copyfileobj(s_file, d_file, block_size)
    f.close()


def main():
    pruned_paths = extract_required_paths(subfolders)
    extract_onii_chan(pruned_paths)
    print('Program complete')

    return


if __name__ == '__main__':
    main()
