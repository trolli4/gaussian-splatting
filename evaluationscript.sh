#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=4:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_own
# #SBATCH --output=logs/visualize_errors/flowers.out

export 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512'

MY_PATH="nerf-360-scenes/flowers"
SCENE_FOLDER=${MY_PATH%/*}
SCENE=${MY_PATH#*/}
MODEL_PATH="output/visualize_errors/${SCENE_FOLDER}/eval/${SCENE}"

# fill test_iterations with all iterations to compute PSNR at
iterations_to_test="1000"
for i in $(seq 2000 1000 30000); do
     iterations_to_test+=" $i"
done

# Source conda.sh to enable 'conda activate' in this script
source $(conda info --base)/etc/profile.d/conda.sh

# Activate environment
conda activate gaussian_splatting_full

# Run training
CUDA_LAUNCH_BLOCKING=1 python train_render_metrics.py \
    -s /home/s76mfroe_hpc/"${MY_PATH}" \
    -m "${MODEL_PATH}" \
    --test_iterations $iterations_to_test \
    -r -1 \
    --disable_viewer \
    --eval \
    --visualize_gradient_cam 69 \
    --iterations 2_000

: <<'COMMENT'
CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/render.py \
    -m "${MODEL_PATH}"
COMMENT

CUDA_LAUNCH_BLOCKING=1 python metrics.py \
    -m "${MODEL_PATH}"
