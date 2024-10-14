import subprocess
import os
import shutil
import json

def run_nnUNet_predict():

    command_prostate = [
        "nnUNet_predict",
        "-i", "/local_ssd/practical_wise24/prostate_cancer/NNUNetModel/nnUNet_raw_data_base/nnUNet_raw_data/Task005_Prostate/imagesTs",
        "-o", "/local_ssd/practical_wise24/prostate_cancer/NNUNetModel/Gland_Segmentation/Results",
        "-t", "Task005_Prostate",
        "-m", "2d"
    ]

    try:
        subprocess.run(command_prostate, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running nnUNet_predict: {e}")
        # Handle the error as needed

run_nnUNet_predict()
