import json

import os


data = {
    "name": "BrainTumorDataset",
    "description": "Brain cancer classification using T1 weighted MRI images",
    "tensorImageSize": "2D",
    "reference": "School of Biomedical Engineering, China",
    "licence": "",
    "release": "",
    "modality": {
        "0": "MRI"
    },
    "labels": {
        "0": "background",
        "1": "meningioma",
        "2": "glioma",
        "3": "pituitary"
    },
    "numTraining": 2144,
    "numTest": 920,
    "file_ending": ".nii.gz",
    "training": {
        "folder": "imagesTr",
        "label": "labelsTr"
    },
    "test": {
        "folder": "imagesTs"
    },
    "numModalities": 1
}

with open('dataset.json', 'w') as json_file:
    json.dump(data, json_file, indent=2)


image_folder = r"C:\Users\Yash\nnUnet_Frame\Data_set\nnUNet_raw\nnUNet_raw_data\Task001_BrainCancerClassification\imagesTr"
label_folder = r"C:\Users\Yash\nnUnet_Frame\Data_set\nnUNet_raw\nnUNet_raw_data\Task001_BrainCancerClassification\labelsTr"

image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(".nii.gz")])
label_files = sorted([f for f in os.listdir(label_folder) if f.endswith(".nii.gz")])

training_list = []

for image_file, label_file in zip(image_files, label_files):
    training_list.append({
        "image": os.path.join(image_folder, image_file),
        "label": os.path.join(label_folder, label_file)
    })

data["training"] = training_list

# ... (write the JSON file as before)

