#!/bin/bash

# Base path where scene folders are located
BASE_DATASET_PATH="/home/s76mfroe_hpc/nerf-360-scenes"

# Log directory
LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

scenes="flowers"

# fill test_iterations with all iterations to compute PSNR at
iterations_to_test="1000"
for i in $(seq 2000 1000 30000); do
     iterations_to_test+=" $i"
done

iterations_to_visualize_gradient="500"
for i in $(seq 1000 500 30000); do
    iterations_to_visualize_gradient+=" $i"
done

# Loop over all folders in dataset path
for folder in $scenes; do
    if [ -d "$BASE_DATASET_PATH"/"$folder" ]; then
        folder_name=$(basename "$folder")
        log_file="${LOG_DIR}/visualize_gradients_v6/eval/${folder_name}.out"
        model_path="output/visualize_gradients_v6/eval/${folder_name}"

        sbatch <<EOF
#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=5:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_train_${folder_name}
#SBATCH --output=${log_file}

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate gaussian_splatting_old

# : <<'comment'
python train.py \\
    -s "${BASE_DATASET_PATH}/${folder_name}" \\
    -m "${model_path}" \\
    --disable_viewer \\
    -r -1 \\
    --eval \
    --test_iterations $iterations_to_test \
    --visualize_gradient_iterations $iterations_to_visualize_gradient \
    --iterations 30_000
# comment

python render.py \\
    -m "${model_path}"

python metrics.py \\
    -m "${model_path}" 
EOF

    fi
done
