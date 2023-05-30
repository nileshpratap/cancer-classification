import os
import h5py

def rename_files(folder_path, prefix_dict, group_name, label_var_name):
    # List all files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.mat')]

    # Initialize a dictionary to store the count of each label
    label_count = {label: 0 for label in prefix_dict.keys()}

    # Iterate through the files in the folder
    for file in files:
        # Load the .mat file
        with h5py.File(os.path.join(folder_path, file), 'r') as mat_file:
            # Assuming the label is stored under a variable named label_var_name
            label = int(mat_file[group_name][label_var_name][0][0])

        # Update the count for the corresponding label
        label_count[label] += 1

        # Create the new file name with the desired naming convention
        new_file_name = f"{prefix_dict[label]}_{label_count[label]:03}.mat"

        # Create the full paths for the original and new file names
        original_file_path = os.path.join(folder_path, file)
        new_file_path = os.path.join(folder_path, new_file_name)

        # Rename the file
        os.rename(original_file_path, new_file_path)

# Provide the folder path, a dictionary containing the desired prefix for each label, and the variable name of the label
folder_path = r"D:\College Stuff\6th Sem\DS-AI Project\Data"
prefix_dict = {1: 'Meningioma',2: 'Glioma', 3: 'Pituitary'}
group_name = 'cjdata'
label_var_name = 'label'  # Replace this with the correct variable name for the label in your .mat files

# Call the rename_files function
rename_files(folder_path, prefix_dict,group_name, label_var_name)
