#!/bin/bash
#SBATCH --partition=mlgpu_devel
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_full_impl
#SBATCH --output=logs/counter_full_implementation_eval.out

MODEL_PATH="output/counter_full_implementation_eval"
rm -rf "$MODEL_PATH"

# fill test_iterations with all iterations to compute PSNR at
iterations_to_test="1000"
for i in $(seq 2000 1000 30000); do
     iterations_to_test+=" $i"
done

# Source conda.sh to enable 'conda activate' in this script
source $(conda info --base)/etc/profile.d/conda.sh

# Activate environment
conda activate gaussian_splatting_opacity_reset

echo "training & rendering.."
CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/train_render_metrics.py \
    -s /home/s76mfroe_hpc/nerf-360-scenes/counter \
    -m "$MODEL_PATH" \
    --eval \
    --test_iterations $iterations_to_test \
    -r 8 \
    --disable_viewer

echo "evaluating.."
CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/metrics.py \
    -m "$MODEL_PATH" 
