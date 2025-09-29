#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=8:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_err_growth
#SBATCH --output=logs/flowers_r_8/error_based_densification_growth_control_mixed_original/percentage_%x_%j.out

if [ -z "$1" ]; then
    echo "Usage: $0 <error_grad_weight>"
    exit 1
fi

weight_fmt=$(printf "%.1f" "$1")

MODEL_PATH="output/flowers_r_8/error_based_densification_growth_control_mixed_original/percentage_${weight_fmt}"
mkdir -p "$MODEL_PATH"
mkdir -p "logs/flowers_r_8/error_based_densification_growth_control_mixed_original"

# fill test_iterations
iterations_to_test="1000"
for i in $(seq 2000 1000 30000); do
    iterations_to_test+=" $i"
done

# Load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate gaussian_splatting

echo "=================================================="
echo "Running with error_grad_weight=${weight_fmt}"
echo "Model path: $MODEL_PATH"
echo "=================================================="

echo "training & rendering..."
CUDA_LAUNCH_BLOCKING=1 python train_render_metrics.py \
    -s /home/s76mfroe_hpc/nerf-360-scenes/flowers \
    -m "$MODEL_PATH" \
    --eval \
    --test_iterations $iterations_to_test \
    -r 8 \
    --error_grad_weight "$weight_fmt" \
    --disable_viewer

echo "evaluating..."
CUDA_LAUNCH_BLOCKING=1 python metrics.py \
    -m "$MODEL_PATH"
