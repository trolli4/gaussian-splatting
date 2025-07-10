#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=3:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_train

MODEL_PATH="output/garden_eval"

# fill test_iterations with all iterations to compute PSNR at
iterations_to_test="1000"
for i in $(seq 2000 1000 30000); do
     iterations_to_test+=" $i"
done

# Source conda.sh to enable 'conda activate' in this script
source $(conda info --base)/etc/profile.d/conda.sh

# Activate environment
conda activate gaussian_splatting

# Run training
CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/train.py \
    -s /home/s76mfroe_hpc/nerf-360-scenes/garden \
    -m "$MODEL_PATH" \
    --eval \
    --test_iterations $iterations_to_test

