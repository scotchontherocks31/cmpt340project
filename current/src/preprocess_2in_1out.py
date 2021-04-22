"""
preprocess.py
Takes nifti type input and resamples it to output dimensions
"""
import os
from distutils.dir_util import copy_tree
from tqdm import tqdm
import shutil
import subprocess
import math
import numpy as np
import numpy.linalg as npl
import nibabel as nib
from nibabel.processing import resample_from_to, resample_to_output, conform
from nibabel.affines import apply_affine
from PIL import Image
from preprocess import import_images, get_MRI_dim, slice_nifti, flairVox_to_mriVox, match_z_len_mri2FLAIR, \
    test_equal_vox


TESTING = True


def concat_png_2in1out(t1w_png, t2w_png):
    """
    Concatenates png images horizontally into a single image to match required input for In2I
    :param t1w_png, t2w_png: string: path to png images to concatenate
    :return: concatenated image as PIL.Image class
    """
    im1 = Image.open(t1w_png)
    im2 = Image.open(t2w_png)
    # TODO: Check image mode (should it be RGB?)
    output = Image.new('RGB', (im1.width + im2.width, im1.height))
    output.paste(im1, (0, 0))
    output.paste(im2, (im1.width, 0))
    return output


# Concatenate all png images in a directory
def concat_patient_imgs_2in1out(t1w_slice_dir, t2w_slice_dir, swi_slice_dir, patient_num):
    """
    Calls concat_png on all slices for a given patient, and moves the resulting files into proper folder structure for
    use in In2I

    :param t1w_slice_dir: string: path to directory of t1w slice png images
    :param t2w_slice_dir: string: path to directory of t2w slice png images
    :param flair_slice_dir: string: path to directory of flair slice png images
    :param swi_slice_dir: string: path to directory of swi slice png images
    :param patient_num: string: patient number
    """
    t1w_slices = sorted(os.listdir(t1w_slice_dir))
    t2w_slices = sorted(os.listdir(t2w_slice_dir))
    swi_slices = sorted(os.listdir(swi_slice_dir))

    if len(t1w_slices) == len(t2w_slices) and len(t1w_slices) == len(swi_slices):
        for i in range(len(t1w_slices)):
            # Concatenate t1w + t2w
            # TODO Only concat sets of images if none are black
            concat_img = concat_png_2in1out(t1w_slice_dir + t1w_slices[i], t2w_slice_dir + t2w_slices[i])
            concat_dest = '../data/processed/2in_1out/trainA/' + patient_num + \
                          '_concat_t1_t2' + str(i) + '.png'

        # Move SWI into the trainB folder
        swi_dest = '../data/processed/2in_1out/trainB/'
        copy_tree(swi_slice_dir, swi_dest)
    else:
        print("Error: directories do not contain equal number of slices!")
        print("t1 slice len = ", len(t1w_slices), "t2 slice len = ", len(t2w_slices), "swi slice len = ", len(swi_slices))
        return


def preprocess_dir_2in1out(directory):
    """
    Preprocesses T1w, T2w, FLAIR, and SWI images for a single patient. Preprocessing includes resizing (downsampling and
    interpolation), creating 2D horizontal slices from 3D images in .nii format, and concatenation of T1w, T2w, and
    FLAIR images into a single image for input into the In2I model
    :param directory: string: Path to directory to preprocess
    """
    images = import_images(directory)
    # Get paths to each individual image
    t1w = list(filter(lambda x: 'T1w' in x, images))[0]
    t2w = list(filter(lambda x: 'T2w' in x, images))[0]
    flair = list(filter(lambda x: 'FLAIR' in x, images))[0]
    swi = list(filter(lambda x: 'swi' in x, images))[0]

    # Unzip flair
    flair_load = nib.load(directory + '/' + flair)
    flair_unzipped = flair[:-7] + '_unzipped.nii'
    nib.save(flair_load, directory + '/' + flair_unzipped)

    # Crop and resize t1w, t2w, and swi Z dimension to match FLAIR
    # Variables are assigned the paths to the new images
    t1w_resized = match_z_len_mri2FLAIR(directory + '/' + t1w, directory + '/' + flair, "t1")
    # test_equal_vox(t1w_resized, directory + '/' + flair)

    t2w_resized = match_z_len_mri2FLAIR(directory + '/' + t2w, directory + '/' + flair, "t1")
    # test_equal_vox(t2w_resized, directory + '/' + flair)

    swi_resized = match_z_len_mri2FLAIR(directory + '/' + swi, directory + '/' + flair)
    # test_equal_vox(swi_resized, directory + '/' + flair)

    # Get patient number from directory
    patient_num = directory[-14:]

    # slice each MRI image into 2d png images along the horizontal plane
    t1w_sl_dir = slice_nifti(t1w_resized, 'T1w', patient_num)
    t2w_sl_dir = slice_nifti(t2w_resized, 'T2w', patient_num)
    swi_sl_dir = slice_nifti(swi_resized, 'swi', patient_num)
    flair_sl_dir = slice_nifti(directory + '/' + flair_unzipped, 'FLAIR', patient_num)

    # Concatenate t1w + t2w horizontally to prepare data for in2i model
    concat_patient_imgs_2in1out(t1w_sl_dir, t2w_sl_dir, swi_sl_dir, patient_num)


# Preprocess full /mri directory
def main():
    failed_directories = []
    all_patients = os.listdir('../../current/data/mri/')
    os.makedirs('../data/processed/2in_1out/trainA')
    os.makedirs('../data/processed/2in_1out/trainB')
    for patient_dir in tqdm(all_patients):
        print("Processing: ", patient_dir)
        try:
            preprocess_dir_2in1out('../../current/data/mri/' + patient_dir)
        except Exception:
            failed_directories.append([patient_dir])
            pass
    print("These directories raised exceptions: ")
    for f in failed_directories:
        print(f[0])

    # Use to preprocess single patient
    # preprocess_dir('../data/mri/OAS30003_MR_d1631')


def preprocess_2in_1out_main_test():
    test_dir = '../data/mri/2in_1out/unit_test'
    patients_to_test = ['../data/mri/OAS30004_MR_d2229',
                        '../data/mri/OAS30005_MR_d1274',
                        '../data/mri/OAS30009_MR_d1210']
    os.makedirs(test_dir)
    os.makedirs('../data/processed/2in_1out/trainA')
    os.makedirs('../data/processed/2in_1out/trainB')
    for patient in patients_to_test:
        patient_num = patient[-14:]
        copy_tree(patient, test_dir + '/' + patient_num)

    # Call main function on test subset
    failed_directories = []
    all_patients = os.listdir(test_dir)
    for patient_dir in tqdm(all_patients):
        print("Processing: ", patient_dir)
        try:
            preprocess_dir_2in1out(test_dir + '/' + patient_dir)
        except Exception:
            failed_directories.append([patient_dir])
            pass
    print("These directories raised exceptions: ")
    for f in failed_directories:
        print(f[0])

    input("Press any key to complete test, and destroy created directories. \n"
          "Warning - this will delete directories made by prior runs on preprocess (nii2png, processed)")

    shutil.rmtree('../data/mri/2in_1out')
    shutil.rmtree('../data/nii2png')
    shutil.rmtree('../data/processed/2in_1out')

    print("testing complete")


if __name__ == "__main__":
    if TESTING is True:
        preprocess_2in_1out_main_test()
    else:
        main()

