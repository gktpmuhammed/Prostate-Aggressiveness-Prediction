import subprocess
import os


def plan_and_preprocess():
    command_lesion = [
        "nnUNetv2_plan_and_preprocess",
        "-d", "600",
        "--verify_dataset_integrity"
    ]
    try:
        subprocess.run(command_lesion, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running picai_lesion: {e}")
        # Handle the error as needed


def train_picai():
    # nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD [additional options, see -h]
    # "--c",

    command_lesion = [
        "nnUNetv2_train",
        "Dataset600_Hum_AI",
        "3d_fullres",
        "1",
        "-device", 'cuda'
    ]

    try:
        subprocess.run(command_lesion, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running picai_lesion: {e}")
        # Handle the error as needed

def inference():
    # nnUNetv2_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -d DATASET_NAME_OR_ID -c CONFIGURATION --save_probabilities
    command_lesion = [
        "nnUNetv2_predict",
        "-i", "/local_ssd/practical_wise24/prostate_cancer/NNUNet_Lesion/nnUNet_raw/Dataset600_Hum_AI/imagesTs",
        "-o", "/local_ssd/practical_wise24/prostate_cancer/NNUNet_Lesion/Deneme",
        "-d", "Dataset600_Hum_AI",
        "-c", "3d_fullres",
        "-f", "1",
        "-chk", "checkpoint_best.pth",
        "-device", 'cuda'
    ]

    try:
        subprocess.run(command_lesion, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running picai_lesion: {e}")
        # Handle the error as needed

plan_and_preprocess()
train_picai()
inference()
