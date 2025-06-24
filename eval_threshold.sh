#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=3:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_eval

# Activate environment
source activate gaussian_splatting

# Run training
python /home/s76mfroe_hpc/gaussian-splatting/train.py \
    -s /home/s76mfroe_hpc/nerf-360-scenes/garden \
    -m output/error-based-densification \
    --quiet \
    --eval
