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

TESTING = True
CROP = False


# Import images tool
def import_images(directory):
    """
    Returns a list of paths to T1w, T2w, FLAIR, and SWI Images present in a directory
    :param directory: string: path to directory of NIFTI images
    :return: list of string: returns the paths of the desired NIFTI image types
    """
    full_list = os.listdir(directory)
    exclude_substrings = ['anat', 'acq-TSE', 'run-02', 'swi4', 'swi2', 'DS_Store']
    final_list = [f for f in full_list if not [ss for ss in exclude_substrings if ss in f]]
    return final_list


# get_MRI_dim
def get_MRI_dim(MRI_nifti):
    """
    Gets the dimensions of a 3D nifti image
    :param MRI_nifti: string: path to 3D image in .nii format
    :return Tuple of int: representing the number of points along the x, y, and z axes of the 3D image
    """
    mri = nib.load(MRI_nifti)
    return mri.shape


def slice_nifti(nifti_image, image_type, patient_num):
    """
    Slices a NIFTI image into horizontal slices in .png images
    :param nifti_image: string: path to 3D image in .nii format
    :param image_type: string: Name of image type being sliced. Used to name directory
    :param patient_num: string: patient number
    :return: string: Path to folder of saved slices
    """
    save_to = '../data/nii2png/' + patient_num + '_' + image_type + '/'
    subprocess.run(["python3", "nii2png.py", "-i", nifti_image, "-o", save_to],
                   input=b'y\n90', stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return save_to


def concat_png(t1w_png, t2w_png, flair_png):
    """
    Concatenates png images horizontally into a single image to match required input for In2I
    :param t1w_png, t2w_png, swi_png: string: path to png images to concatenate
    :return: concatenated image as PIL.Image class
    """
    im1 = Image.open(t1w_png)
    im2 = Image.open(t2w_png)
    im3 = Image.open(flair_png)
    # TODO: Check image mode (should it be RGB?)
    output = Image.new('RGB', (im1.width + im2.width + im3.width, im1.height))
    output.paste(im1, (0, 0))
    output.paste(im2, (im1.width, 0))
    output.paste(im3, (im1.width + im2.width, 0))
    return output


# Concatenate all png images in a directory
def concat_patient_imgs(t1w_slice_dir, t2w_slice_dir, flair_slice_dir, swi_slice_dir, patient_num):
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
    flair_slices = sorted(os.listdir(flair_slice_dir))
    swi_slices = sorted(os.listdir(swi_slice_dir))

    if len(t1w_slices) == len(t2w_slices) and len(t1w_slices) == len(flair_slices) \
            and len(t1w_slices) == len(swi_slices):
        for i in range(len(t1w_slices)):
            # Concatenate t1w + t2w + flair
            # Remove first and last slice as they are often black
            if i == 0 or i == len(t1w_slices) - 1:
                continue
            concat_img = concat_png(t1w_slice_dir + t1w_slices[i], t2w_slice_dir + t2w_slices[i],
                                    flair_slice_dir + flair_slices[i])
            concat_dest = '../data/processed/trainA/' + patient_num + \
                          '_concat_t1_t2_flair_' + str(i-1) + '.png'
            concat_img.save(concat_dest)

            # Rename swi images to match convention
            dst = swi_slice_dir + patient_num + '_swi' + str(i-1) + '.png'
            src = swi_slice_dir + swi_slices[i]
            os.rename(src, dst)

        # Move SWI into the trainB folder
        swi_dest = '../data/processed/trainB/'
        # Remove first and last slice
        os.remove(swi_slice_dir + swi_slices[0])
        os.remove(swi_slice_dir + swi_slices[len(swi_slices) - 1])
        copy_tree(swi_slice_dir, swi_dest)
    else:
        print("Error: directories do not contain equal number of slices!")
        print("t1 slice len = ", len(t1w_slices), "t2 slice len = ", len(t2w_slices), "flair slice len = ",
              len(flair_slices), "swi slice len = ", len(swi_slices))
        return


def crop(input_image, cropped_name='cropped.nii', tol=150, nose_reserve=8, ear_reserve=6):
    """
    Crops off the border of a 3D nifti image, where the border consists of pixels whose values are lower than a given tolerance level
    :param input_image: string: path to 3D image in .nii format which will be cropped
    :param cropped_name: path to 3D image in .nii format which was cropped
    :param tol: Int: pixel value tolerance level
    :param nose_reserve: Int: amount of pixels to reserve for nose
    :param ear_reserve: Int: amount of pixels to reserve for ears
    :return cropped_name: path to 3D image in .nii format which was cropped
    """
    nim = nib.load(input_image)
    img = np.array(nim.dataobj)
    if cropped_name == 'cropped.nii':
        cropped_name = input_image[:-4] + '_cropped.nii'

    idx = np.nonzero(img > tol)
    x1 = max(0, idx[0].min() - ear_reserve)  # right-ear
    y1 = max(0, idx[1].min())  # back-neck
    z1 = max(0, idx[2].min())  # top of head
    x2 = min(img.shape[0], idx[0].max() + 1 + ear_reserve)  # left-ear
    y2 = min(img.shape[1], idx[1].max() + 1 + nose_reserve)  # front-face
    z2 = min(img.shape[2], idx[2].max() + 1)  # bottom-jaw
    img = img[x1:x2, y1:y2, z1:z2]

    affine = nim.affine
    affine[:3, 3] = np.dot(affine, np.array([x1, y1, z1, 1]))[:3]
    cropped = nib.Nifti1Image(img, affine)
    nib.save(cropped, cropped_name)
    return cropped_name


def flairVox_to_mriVox(mri_img, flair_img):
    return npl.inv(mri_img.affine).dot(flair_img.affine)


def match_mri_to_FLAIR(mri, flair, img_type=None):
    mri_img = nib.load(mri)
    flair_img = nib.load(flair)

    flair_dim = get_MRI_dim(flair)

    flair_vox_center = (np.array(flair_img.shape) - 1) / 2.
    flair_vox_center_low = np.copy(flair_vox_center)
    flair_vox_center_low[2] = 0
    flair_vox_center_high = np.copy(flair_vox_center)
    flair_vox_center_high[2] = int(flair_dim[2])

    mri_vox_at_flair_low = apply_affine(flairVox_to_mriVox(mri_img, flair_img), flair_vox_center_low)
    mri_vox_at_flair_high = apply_affine(flairVox_to_mriVox(mri_img, flair_img), flair_vox_center_high)

    # Crop the image
    # print("orig dimensions: ", mri_img.shape)
    if img_type == "t1":
        mri_img_cropped = mri_img.slicer[:, :, round(mri_vox_at_flair_low[2]):round(mri_vox_at_flair_high[2]):6]
    else:
        mri_img_cropped = mri_img.slicer[:, :, round(mri_vox_at_flair_low[2]):round(mri_vox_at_flair_high[2]):3]
    mri_img_cropped = conform(mri_img_cropped, flair_img.shape, [0.82, 0.82, 6])
    # print("cropped dimensions: ", mri_img_cropped.shape)

    output_name = mri[:-7] + '_resized.nii'
    nib.save(mri_img_cropped, output_name)
    return output_name


def test_equal_vox(img1_path, img2_path):
    # Load images
    img1 = nib.load(img1_path)
    img2 = nib.load(img2_path)

    # Get coordinates for the center voxel of img2
    img2_vox_center = (np.array(img2.shape) - 1) / 2.
    img2_x = img2_vox_center[0]
    img2_y = img2_vox_center[1]

    # Find the voxel in img1 which matches the real world coordinates of the center of img 2
    img1_vox_at_img2_centre = apply_affine(flairVox_to_mriVox(img1, img2), img2_vox_center)
    img1_x = img1_vox_at_img2_centre[0]
    img1_y = img1_vox_at_img2_centre[1]

    # Check each slice for equality
    for z in range(24):
        print(z, ": ", apply_affine(img1.affine, [img1_x, img1_y, z]), "==",
              apply_affine(img2.affine, [img2_x, img2_y, z]))


def preprocess_dir(directory):
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
    swi = list(filter(lambda x: 'swi' in x, images))[0]
    flair = list(filter(lambda x: 'FLAIR' in x, images))[0]

    t1w_path = directory + '/' + t1w
    t2w_path = directory + '/' + t2w
    swi_path = directory + '/' + swi
    flair_path = directory + '/' + flair

    # Unzip flair
    flair_load = nib.load(flair_path)
    flair_unzipped = '{0}/{1}_unzipped.nii'.format(directory, flair[:-7])
    nib.save(flair_load, flair_unzipped)

    # Cropping images (comment this out if not using crop)
    if CROP is True:
        t1w_cropped = crop(t1w_path)
        t2w_cropped = crop(t2w_path)
        swi_cropped = crop(swi_path)
        flair_cropped = crop(flair_unzipped)

        # Crop and resize t1w, t2w, and swi Z dimension to match FLAIR
        # Variables are assigned the paths to the new images
        t1w_resized = match_mri_to_FLAIR(t1w_cropped, flair_cropped, "t1")
        t2w_resized = match_mri_to_FLAIR(t2w_cropped, flair_cropped, "t1")
        swi_resized = match_mri_to_FLAIR(swi_cropped, flair_cropped)

    else:
        # Crop and resize t1w, t2w, and swi Z dimension to match FLAIR
        # Variables are assigned the paths to the new images
        t1w_resized = match_mri_to_FLAIR(t1w_path, flair_unzipped, "t1")
        t2w_resized = match_mri_to_FLAIR(t1w_path, flair_unzipped, "t1")
        swi_resized = match_mri_to_FLAIR(t2w_path, flair_unzipped)

    # Use to test if voxels are correctly lined up between FLAIR and other image types
    # print("t1w : FLAIR voxel comparison in real space")
    # test_equal_vox(t1w_resized, directory + '/' + flair)
    # print("t2w : FLAIR voxel comparison in real space")
    # test_equal_vox(t2w_resized, directory + '/' + flair)
    # print("swi : FLAIR voxel comparison in real space")
    # test_equal_vox(swi_resized, directory + '/' + flair)

    # Get patient number from directory
    patient_num = directory[-14:]

    # slice each MRI image into 2d png images along the horizontal plane
    t1w_sl_dir = slice_nifti(t1w_resized, 'T1w', patient_num)
    t2w_sl_dir = slice_nifti(t2w_resized, 'T2w', patient_num)
    swi_sl_dir = slice_nifti(swi_resized, 'swi', patient_num)
    flair_sl_dir = slice_nifti(flair_unzipped, 'FLAIR', patient_num)

    # Concatenate t1w + t2w + FLAIR horizontally to prepare data for in2i model
    concat_patient_imgs(t1w_sl_dir, t2w_sl_dir, flair_sl_dir, swi_sl_dir, patient_num)


# Preprocess full /mri directory
def main():
    failed_directories = []
    all_patients = os.listdir('../../current/data/mri/')
    os.makedirs('../data/processed/trainA')
    os.makedirs('../data/processed/trainB')
    for patient_dir in tqdm(all_patients):
        print("Processing: ", patient_dir)
        try:
            preprocess_dir('../../current/data/mri/' + patient_dir)
        except Exception:
            failed_directories.append([patient_dir])
            pass
    print("These directories raised exceptions: ")
    for f in failed_directories:
        print(f[0])

    # Use to preprocess single patient
    # preprocess_dir('../data/mri/OAS30003_MR_d1631')


def preprocess_main_test():
    test_dir = '../data/mri/unit_test'
    patients_to_test = ['../data/mri/OAS30004_MR_d2229',
                        '../data/mri/OAS30005_MR_d1274',
                        '../data/mri/OAS30009_MR_d1210']
    os.mkdir(test_dir)
    os.makedirs('../data/processed/trainA')
    os.makedirs('../data/processed/trainB')
    for patient in patients_to_test:
        patient_num = patient[-14:]
        copy_tree(patient, test_dir + '/' + patient_num)

    # Call main function on test subset
    failed_directories = []
    all_patients = os.listdir(test_dir)
    for patient_dir in tqdm(all_patients):
        print("Processing: ", patient_dir)
        try:
            preprocess_dir(test_dir + '/' + patient_dir)
        except Exception:
            failed_directories.append([patient_dir])
            pass
    print("These directories raised exceptions: ")
    for f in failed_directories:
        print(f[0])

    input("Press any key to complete test, and destroy created directories. \n"
          "Warning - this will delete directories made by prior runs on preprocess (nii2png, processed)")

    shutil.rmtree('../data/mri/unit_test')
    shutil.rmtree('../data/nii2png')
    shutil.rmtree('../data/processed')

    print("testing complete")


if __name__ == "__main__":
    if TESTING is True:
        preprocess_main_test()
    else:
        main()

