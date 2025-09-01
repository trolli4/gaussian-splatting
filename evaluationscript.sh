#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=3:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_{$1}
# #SBATCH --output=logs/own-scenes/eval/{$1}.out

MY_PATH="own-scenes/$1"
SCENE_FOLDER=${MY_PATH%/*}
SCENE=${MY_PATH#*/}
MODEL_PATH="output/${SCENE_FOLDER}/eval/${SCENE}"

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
CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/train_render_metrics.py \
    -s /home/s76mfroe_hpc/"${MY_PATH}" \
    -m "${MODEL_PATH}" \
    --test_iterations $iterations_to_test \
    -r 8 \
    --disable_viewer \
    --eval

: <<'COMMENT'
CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/render.py \
    -m "${MODEL_PATH}"
COMMENT

CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/metrics.py \
    -m "${MODEL_PATH}"
