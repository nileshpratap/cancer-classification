import os
import numpy as np
import nibabel as nib
import h5py

input_mat_folder = r"D:\College Stuff\6th Sem\DS-AI Project\Data"
output_label_folder = r"D:\College Stuff\6th Sem\DS-AI Project\nnUNet_raw\Dataset001_Brain_Cancer_Classification\labelsTr"

for file in os.listdir(input_mat_folder):
    if file.endswith(".mat"):
        file_path = os.path.join(input_mat_folder, file)

        with h5py.File(file_path, 'r') as mat_data:
            tumor_mask = np.array(mat_data['cjdata']["tumorMask"])

            # Create the NIfTI image from the tumor mask data
            label_data = np.array(tumor_mask, dtype=np.uint8)
            label_nifti = nib.Nifti1Image(label_data, affine=np.eye(4))

            # Save the label file with the corresponding name in the labelsTr folder
            label_filename = file[:-3] + ".nii.gz"
            label_filepath = os.path.join(output_label_folder, label_filename)
            nib.save(label_nifti, label_filepath)
