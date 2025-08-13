#!/bin/bash

# Base path where scene folders are located
BASE_DATASET_PATH="/home/s76mfroe_hpc/nerf-360-scenes"

# Log directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

scenes="bicycle flowers garden"

# Loop over all folders in dataset path
for folder in $scenes; do
    if [ -d "$BASE_DATASET_PATH"/"$folder" ]; then
        folder_name=$(basename "$folder")
        log_file="${LOG_DIR}/original/eval/${folder_name}.out"
        model_path="output/original/eval/${folder_name}"

        sbatch <<EOF
#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=5:00:00
#SBATCH --gpus=2
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_train_${folder_name}
#SBATCH --output=${log_file}

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate gaussian_splatting_old

: <<'comment'
python train.py \\
    -s "${BASE_DATASET_PATH}/${folder_name}" \\
    -m "${model_path}" \\
    --disable_viewer \\
    -r -1 \\
    --eval
comment

python render.py \\
    -m "${model_path}"

python metrics.py \\
    -m "${model_path}" 
EOF

    fi
done
