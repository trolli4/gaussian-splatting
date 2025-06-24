#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=3:00:00
#SBATCH --gpus=2
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_train

# Activate environment
source activate gaussian_splatting

# Run training
CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/train.py \
    -s /home/s76mfroe_hpc/nerf-360-scenes/garden -m output/error-based-densification
