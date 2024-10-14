#!/usr/bin/bash

#SBATCH -J "Prostate Cancer Encoder Training" # job name
#SBATCH -o slurm_output.out
#BATCH -e slurm_error.out
#SBATCH --mem=60G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --partition=course

# load python module
module load python/anaconda3

source /opt/anaconda3/etc/profile.d/conda.sh

# activate conda environment
conda activate ProstateCancer

# call training.py and pass arguments
python encoder_training.py "$@"