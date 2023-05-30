import os

def rename_files_serially(directory, prefix):
    # Iterate through all files in the directory
    for index, filename in enumerate(sorted(os.listdir(directory)), start=1):
        # Construct new filename with prefix and padded index
        file_extension = os.path.splitext(filename)[1]
        if file_extension == ".gz":
            file_extension = ".nii.gz"
        new_filename = f"{prefix}{index:03d}_0000{file_extension}"

        # Generate the absolute paths for the source and destination files
        src = os.path.join(directory, filename)
        dest = os.path.join(directory, new_filename)

        # Rename the file
        os.rename(src, dest)

if __name__ == "__main__":
    # Set your desired directory path and prefix here
    directory = r"D:\College Stuff\6th Sem\DS-AI Project\nnUNet_raw\Dataset001_BrainCancerClassification\labelsTs"
    prefix = "MRI_"

    rename_files_serially(directory, prefix)
