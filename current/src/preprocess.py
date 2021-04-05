"""
preprocess.py
Takes nifti type input and resamples it to output dimensions
"""
import os
from distutils.dir_util import copy_tree
from tqdm import tqdm
import shutil
import subprocess
import nibabel as nib
import numpy as np
from nibabel.processing import resample_to_output, conform
from pathlib import Path
from PIL import Image


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


def resize(input_image, output_dim):
    """
    Applies downsampling and interpolation to a NIFTI file, reducing it's spatial resolution
    to match specified dimensions
    :param input_image: string: path to 3D image in .nii format which will be resized
    :param output_dim: tuple: takes a triple of int values, in the form (x, y, z),
    which input_image will be resized match
    :return output_name: string: path to resized 3D image in .nii
    """
    input_dim = get_MRI_dim(input_image)
    input_mri = nib.load(input_image)
    output_name = input_image[:-7] + '_resized.nii'
    voxel_size = [input_dim[0] / output_dim[0], input_dim[1] / output_dim[1], input_dim[2] / output_dim[2]]
    # Downsample with interpolation
    interp = resample_to_output(input_mri, voxel_size)
    interp = conform(interp, output_dim, voxel_size)
    # Save interpolated image
    nib.save(interp, output_name)
    return output_name


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
    img = nim.get_data()
    if cropped_name == 'cropped.nii':
        cropped_name = input_image[:-4] + '_cropped.nii'

    idx = np.nonzero(img > tol)
    x1 = max(0, idx[0].min() - ear_reserve)                 #right-ear
    y1 = max(0, idx[1].min())                               #back-neck
    z1 = max(0, idx[2].min())                               #top of head
    x2 = min(img.shape[0], idx[0].max() + 1 + ear_reserve)  #left-ear
    y2 = min(img.shape[1], idx[1].max() + 1 + nose_reserve) #front-face
    z2 = min(img.shape[2], idx[2].max() + 1)                #bottom-jaw
    img = img[x1:x2, y1:y2, z1:z2]
    
    affine = nim.affine
    affine[:3, 3] = np.dot(affine, np.array([x1, y1, z1, 1]))[:3]
    cropped = nib.Nifti1Image(img, affine)
    nib.save(cropped, cropped_name)
    return cropped_name


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
                   input=b'n', stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return save_to


def concat_png(t1w_png, t2w_png, swi_png):
    """
    Concatenates png images horizontally into a single image to match required input for In2I
    :param t1w_png, t2w_png, swi_png: string: path to png images to concatenate
    :return: concatenated image as PIL.Image class
    """
    im1 = Image.open(t1w_png)
    im2 = Image.open(t2w_png)
    im3 = Image.open(swi_png)
    # TODO: Check image mode (should it be RGB?)
    output = Image.new('RGB', (im1.width, im1.height + im2.height + im3.height))
    output.paste(im1, (0, 0))
    output.paste(im2, (0, im1.height))
    output.paste(im3, (0, im1.height + im2.height))
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
        os.makedirs('../data/processed/patient' + patient_num + '/trainA')
        os.makedirs('../data/processed/patient' + patient_num + '/trainB')
        for i in range(len(t1w_slices)):
            # Concatenate t1w + t2w + flair
            concat_img = concat_png(t1w_slice_dir + t1w_slices[i], t2w_slice_dir + t2w_slices[i],
                                    flair_slice_dir + flair_slices[i])

            concat_dest = '../data/processed/patient' + patient_num + '/trainA/concat_t1_t2_flair' + str(i) + '.png'
            concat_img.save(concat_dest)

        # Move SWI into the trainB folder
        swi_dest = '../data/processed/patient' + patient_num + '/trainB/'
        copy_tree(swi_slice_dir, swi_dest)
    else:
        print("Error: directories do not contain equal number of slices!")
        print("t1 slice len = ", len(t1w_slices), "t2 slice len = ", len(t2w_slices), "flair slice len = ",
              len(flair_slices), "swi slice len = ", len(swi_slices), )
        return


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
    flair = list(filter(lambda x: 'FLAIR' in x, images))[0]
    swi = list(filter(lambda x: 'swi' in x, images))[0]

    # Get output dimensions from flair image
    target_dim = get_MRI_dim(directory + '/' + flair)

   # Cropping images (comment this out if not using crop)
    t1w_cropped = crop(directory + '/' + t1w)
    t2w_cropped = crop(directory + '/' + t2w)
    swi_cropped = crop(directory + '/' + swi)



    # Resize t1w, t2w, and swi to have the same dimensions as flair image
    # Variables are assigned the paths to the new images

    #Comment out these if using crop
    # t1w_resized = resize(directory + '/' + t1w, target_dim)
    # t2w_resized = resize(directory + '/' + t2w, target_dim)
    # swi_resized = resize(directory + '/' + swi, target_dim)

    #Comment out these below if not using crop
    t1w_resized = resize(t1w_cropped, target_dim)
    t2w_resized = resize(t2w_cropped, target_dim)
    swi_resized = resize(swi_cropped, target_dim)



    # Unzip flair
    flair_load = nib.load(directory + '/' + flair)
    flair_unzipped = flair[:-7] + '_unzipped.nii'
    nib.save(flair_load, directory + '/' + flair_unzipped)

    # Get patient number from directory
    patient_num = directory[-14:]


    
    # slice each MRI image into 2d png images along the horizontal plane
    t1w_sl_dir = slice_nifti(t1w_resized, 'T1w', patient_num)
    t2w_sl_dir = slice_nifti(t2w_resized, 'T2w', patient_num)
    swi_sl_dir = slice_nifti(swi_resized, 'swi', patient_num)
    flair_sl_dir = slice_nifti(directory + '/' + flair_unzipped, 'FLAIR', patient_num)

    # Concatenate t1w + t2w + swi horizontally to prepare data for in2i model
    concat_patient_imgs(t1w_sl_dir, t2w_sl_dir, flair_sl_dir, swi_sl_dir, patient_num)


# Preprocess full /mri directory
def main():
    failed_directories = []
    all_patients = os.listdir('../../current/data/mri/')
    for patient_dir in tqdm(all_patients):
        print("Processing: ", patient_dir)
        try:
            preprocess_dir('../../current/data/mri/' + patient_dir)
        except Exception as e:
            print(e)
            failed_directories.append([patient_dir])
            pass
    print("These directories raised exceptions: ")
    for f in failed_directories:
        print(f[0])

    # Use to preprocess single patient
    # preprocess_dir('../data/mri/OAS30003_MR_d1631')


if __name__ == "__main__":
    main()