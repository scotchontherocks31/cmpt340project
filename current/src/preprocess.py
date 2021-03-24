import numpy as np
from scipy import ndimage, misc
import nibabel as nib
from nibabel.processing import resample_to_output, conform

# Variables for file to be interpolated (input_type_file) and reference file w/ wanted dimensions (output_type_file)
# (change these for testing – make sure files are in the same directory as .py file)
input_type_file = 'sub-OAS30003_ses-d1631_run-01_T1w.nii'
output_type_file = 'sub-OAS30003_ses-d1631_FLAIR.nii'


# Get original dimensions
input_type = nib.load(input_type_file)
input_type_array = np.array(input_type.dataobj).astype("uint8")
input_type_size = input_type_array.shape
print('Input Type Size: ', input_type_size)

input_size_x = input_type_size[0]
input_size_y = input_type_size[1]
input_size_z = input_type_size[2]

# Get target dimensions
output_type = nib.load(output_type_file)
output_type_array = np.array(output_type.dataobj).astype("uint8")
output_type_size = output_type_array.shape
print('Output Type Size: ', output_type_size)

output_size_x = output_type_size[0]
output_size_y = output_type_size[1]
output_size_z = output_type_size[2]

# Convert to NIFTI file
interpolation = resample_to_output(input_type, [ input_size_x/output_size_x, input_size_y/output_size_y, input_size_z/output_size_z ])
interpolation = conform(interpolation, output_type_size, [ input_size_x/output_size_x, input_size_y/output_size_y, input_size_z/output_size_z ])
print('Actual output size: ', interpolation.shape)
nib.save(interpolation, "interpolation.nii.gz")
