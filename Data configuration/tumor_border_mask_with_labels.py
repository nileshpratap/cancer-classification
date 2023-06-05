import os
import numpy as np
import h5py
import nibabel as nib

input_mat_folder = r"D:\College Stuff\6th Sem\DS-AI Project\Data"
output_label_folder = r"D:\College Stuff\6th Sem\DS-AI Project\nnUNet_raw\Dataset001_BrainCancerClassification\labelsTs"

# Iterate over the .mat files in the input folder
for file in os.listdir(input_mat_folder):
    if file.endswith(".mat"):
        file_path = os.path.join(input_mat_folder, file)

        # Load the .mat file using h5py
        with h5py.File(file_path, "r") as mat_data:
            tumor_mask = np.array(mat_data["cjdata"]["tumorMask"])
            label = np.array(mat_data["cjdata"]["label"])

        # Convert the binary mask to label mask
        label_mask = np.zeros_like(tumor_mask)
        label_mask[np.where(tumor_mask == 1)] = label

        # Create a NIfTI image from the label mask
        label_nifti = nib.Nifti1Image(label_mask, affine=np.eye(4))

        # Save the label file in the labelsTr folder with the corresponding name
        label_filename = file[:-4] + ".nii.gz"
        label_filepath = os.path.join(output_label_folder, label_filename)
        nib.save(label_nifti, label_filepath)
