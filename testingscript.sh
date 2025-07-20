#!/bin/bash

# Base path where scene folders are located
BASE_DATASET_PATH="/home/s76mfroe_hpc/nerf-360-scenes"

# Log directory
LOG_DIR="./logs/densification_growth-control_opacity-correction"
mkdir -p "$LOG_DIR"

# Model size file
MODEL_SIZE_FILE="../model_sizes.txt"

# Loop over all folders in dataset path
for folder in "$BASE_DATASET_PATH"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        log_file="${LOG_DIR}/${folder_name}.out"
        model_path="output/densification_growth-control_opacity-correction/${folder_name}"

        # Extract max_model_size from model_sizes.txt
        max_model_size=$(grep "${folder_name}.out" "$MODEL_SIZE_FILE" | \
                         grep -oP 'Number of points at end: \K[0-9]+')

        # Skip if no value found
        if [ -z "$max_model_size" ]; then
            echo "Warning: No model size found for '${folder_name}' — skipping."
            continue
        fi

        sbatch <<EOF
#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=3:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_train_${folder_name}
#SBATCH --output=${log_file}

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate gaussian_splatting

#echo "training & rendering.."
#CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/train_render_metrics.py \\
CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/train.py \\
    -s "${BASE_DATASET_PATH}/${folder_name}" \\
    -m "${model_path}" \\
    --disable_viewer \\
    --eval \\
    # -r 8 \\
    --max_number_gaussians "${max_model_size}" \\
    --densify_error_threshold 1

#echo "evaluating.."
#CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/metrics.py \
#    -m "${model_path}"
EOF

    fi
done
