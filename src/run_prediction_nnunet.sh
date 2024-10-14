#!/usr/bin/bash

#SBATCH -J "NNUNET" # job name
#SBATCH --output=/local_ssd/practical_wise24/prostate_cancer/NNUNetModel/Slurm_Outputs/%j_out.out # standard output directory
#SBATCH --error=/local_ssd/practical_wise24/prostate_cancer/NNUNetModel/Slurm_Outputs/%j_err.err # standard error directory
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=course

export nnUNet_raw_data_base="/local_ssd/practical_wise24/prostate_cancer/NNUNetModel/nnUNet_raw_data_base"
export nnUNet_preprocessed="/local_ssd/practical_wise24/prostate_cancer/NNUNetModel/nnUNet_preprocessed"
export RESULTS_FOLDER="/local_ssd/practical_wise24/prostate_cancer/NNUNetModel/nnUNet_trained_models"

echo "nnUNet_raw_data_base: $nnUNet_raw_data_base"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "RESULTS_FOLDER: $RESULTS_FOLDER"
echo "Current working directory: $(pwd)"


# load python module
module load python/anaconda3

source /opt/anaconda3/etc/profile.d/conda.sh

# activate conda environment
conda activate Segmentation

echo "SLURM Job ID: $SLURM_JOBID"

python prediction_nnunet.py


