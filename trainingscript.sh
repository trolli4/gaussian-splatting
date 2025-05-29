#!/bin/bash
#SBATCH --partition=mlgpu_devel
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_train

# Activate environment
source activate gaussian_splatting

# Run training
python /lustre/scratch/data/s76mfroe_hpc-bpg_gaussian_splatting/gaussian-splatting/train.py \
    -s /lustre/scratch/data/s76mfroe_hpc-bpg_gaussian_splatting/nerf-360-scenes/bonsai 
