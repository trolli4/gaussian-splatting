#!/bin/bash
#SBATCH --partition=mlgpu_devel
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_debug_NaN

MODEL_PATH="output/debug_NaN_densification"

# Source conda.sh to enable 'conda activate' in this script
source $(conda info --base)/etc/profile.d/conda.sh

# Activate environment
conda activate gaussian_splatting

echo "trainng.."
CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/train.py \
    -s /home/s76mfroe_hpc/nerf-360-scenes/garden \
    -m "$MODEL_PATH" \
    --eval \
    -r 8

echo "rendering.."
CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/render.py \
    -m "$MODEL_PATH" 

echo "evaluating.."
CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/metrics.py \
    -m "$MODEL_PATH" 
