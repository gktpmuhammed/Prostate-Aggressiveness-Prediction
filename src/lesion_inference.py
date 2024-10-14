import subprocess
import os

def inference():
    # nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
    command_lesion = [
        "nnUNetv2_predict",
        "-i", "/local_ssd/practical_wise24/prostate_cancer/NNUNet_Lesion/nnUNet_raw/Dataset777_ProstateLesion/imagesTs",
        "-o", "/local_ssd/practical_wise24/prostate_cancer/NNUNet_Lesion/Private_Dataset_Segmentation_Results",
        "-d", "Dataset777_ProstateLesion",
        "-c", "3d_fullres",
        "-f", "0",
        "-chk", "checkpoint_best.pth",
        "-device", 'cuda'
    ]

    try:
        subprocess.run(command_lesion, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running picai_lesion: {e}")
        # Handle the error as needed

inference()
