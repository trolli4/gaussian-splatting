#!/bin/bash
#SBATCH --partition=mlgpu_devel
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_old_NaN

# Activate environment
source activate gaussian_splatting_old

# Run training
python /home/s76mfroe_hpc/gaussian-splatting/train.py \
    -s /home/s76mfroe_hpc/nerf-360-scenes/garden -m output/original-3dgs-test-NaN -r 8

python /home/s76mfroe_hpc/gaussian-splatting/render.py \
    -m output/original-3dgs-test-NaN

python /home/s76mfroe_hpc/gaussian-splatting/metrics.py \
    -m output/original-3dgs-test-NaN
