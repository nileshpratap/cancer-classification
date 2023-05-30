<!-- Link to the nnUnet framework -->
https://github.com/MIC-DKFZ/nnUNet

prerequisites to run the model in you system:
1. pytorch
2. nnUNet dependencies

command to preprocess
nnUNet_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

command to train the model
nnUNet_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD --val --npz

command for inference of the model
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
