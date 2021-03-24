"""
preprocess.py
Takes nifti type input and resamples it to output dimensions
"""
import os
import subprocess
import nibabel as nib
from nibabel.processing import resample_to_output, conform

# Variables for file to be interpolated (input_type_file) and reference file w/ wanted dimensions (output_type_file)
# (change these for testing – make sure files are in the same directory as .py file)
input_type_file = './sub-OAS30001_ses-d3132_T1w.nii'
output_type_file = './sub-OAS30001_ses-d2430_FLAIR.nii'

# Get original dimensions
input_type = nib.load(input_type_file)
input_type_size = input_type.shape
print('Input Type Size: ', input_type_size)

input_size_x = input_type_size[0]
input_size_y = input_type_size[1]
input_size_z = input_type_size[2]

# Get target dimensions
output_type = nib.load(output_type_file)
output_type_size = output_type.shape
print('Output Type Size: ', output_type_size)

output_size_x = output_type_size[0]
output_size_y = output_type_size[1]
output_size_z = output_type_size[2]

# Convert to NIFTI file
voxel_size = [input_size_x / output_size_x, input_size_y / output_size_y, input_size_z / output_size_z]
interpolation = resample_to_output(input_type, voxel_size)
interpolation = conform(interpolation, output_type_size, voxel_size)
print('Actual output size: ', interpolation.shape)
nib.save(interpolation, "interpolation.nii")

# Split nifti into png slices
subprocess.run(["python3", "nii2png.py", "-i", "interpolation.nii", "-o", "../results/nii2png/t1w_test/"], input=b'n')


