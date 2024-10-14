#!/usr/bin/bash

#SBATCH -J "Picai" # job name
#SBATCH --output=/local_ssd/practical_wise24/prostate_cancer/NNUNet_Lesion/Slurm_Outputs/%j_out.out # standard output directory
#SBATCH --error=/local_ssd/practical_wise24/prostate_cancer/NNUNet_Lesion/Slurm_Outputs/%j_err.err # standard error directory
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=course

export nnUNet_raw="/local_ssd/practical_wise24/prostate_cancer/NNUNet_Lesion/nnUNet_raw"
export nnUNet_preprocessed="/local_ssd/practical_wise24/prostate_cancer/NNUNet_Lesion/nnUNet_preprocessed"
export nnUNet_results="/local_ssd/practical_wise24/prostate_cancer/NNUNet_Lesion/nnUNet_results"
export nnUNet_n_proc_DA=32
export MKL_THREADING_LAYER=GNU

echo "nnUNet_raw: $nnUNet_raw"
echo "nnUNet_preprocessed: $nnUNet_preprocessed"
echo "nnUNet_results: $nnUNet_results"
echo "Current working directory: $(pwd)"


# load python module
module load python/anaconda3

source /opt/anaconda3/etc/profile.d/conda.sh

# activate conda environment
conda activate Segmentation

echo "SLURM Job ID: $SLURM_JOBID"

python training_picai.py


