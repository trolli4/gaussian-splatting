#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=8:00:00
#SBATCH --gpus=2
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_err_growth
#SBATCH --output=logs/counter_error_based_densification_growth_control_eval_1.out

MODEL_PATH="output/counter_error_based_densification_growth_control_eval_1"

# fill test_iterations with all iterations to compute PSNR at
iterations_to_test="1000"
for i in $(seq 2000 1000 30000); do
     iterations_to_test+=" $i"
done

# Source conda.sh to enable 'conda activate' in this script
source $(conda info --base)/etc/profile.d/conda.sh

# Activate environment
conda activate gaussian_splatting

echo "training & rendering.."
CUDA_LAUNCH_BLOCKING=1 python train_render_metrics.py \
    -s /home/s76mfroe_hpc/nerf-360-scenes/counter \
    -m "$MODEL_PATH" \
    --eval \
    --test_iterations $iterations_to_test \
    --densify_error_threshold 1 \
    -r -1 \
    --disable_viewer

echo "evaluating.."
CUDA_LAUNCH_BLOCKING=1 python metrics.py \
    -m "$MODEL_PATH" 
