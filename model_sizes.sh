#!/bin/bash

# Base path where scene folders are located
BASE_DATASET_PATH="/home/s76mfroe_hpc/nerf-360-scenes"

# Log directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

# Loop over all folders in dataset path
for folder in "$BASE_DATASET_PATH"/*; do
    if [ -d "$folder" ]; then
        folder_name=$(basename "$folder")
        log_file="${LOG_DIR}/${folder_name}.out"
        model_path="output/${folder_name}"

        sbatch <<EOF
#!/bin/bash
#SBATCH --partition=mlgpu_devel
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_train_${folder_name}
#SBATCH --output=${log_file}

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate gaussian_splatting_old

python /home/s76mfroe_hpc/gaussian-splatting/train.py \\
    -s "${BASE_DATASET_PATH}/${folder_name}" \\
    -m "${model_path}" \\
    -r 8
EOF

    fi
done
