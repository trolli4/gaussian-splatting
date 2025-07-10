#!/bin/bash
[ ! -d "./logs/" ] && mkdir "./logs/"

iterations_to_test="1000"
for i in $(seq 2000 1000 30000); do
     iterations_to_test+=" $i"
done

# Loop over threshold values from 20 to 110 in steps of 10
for threshold in {1..10..1}
do
    # Create a unique job script for this threshold
    JOB_SCRIPT="run_threshold_${threshold}.sh"

    cat <<EOF > $JOB_SCRIPT
#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=3:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_train_${threshold}
#SBATCH --output=logs/garden_${threshold}.out

# Source conda.sh to enable 'conda activate' in this script
source $(conda info --base)/etc/profile.d/conda.sh

# Activate environment
conda activate gaussian_splatting

# Run training with custom threshold
CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/train.py \\
    -s /home/s76mfroe_hpc/nerf-360-scenes/garden \\
    -m output/garden_${threshold} \\
    --densify_error_threshold ${threshold} \
    --test_iterations $iterations_to_test

EOF

    # Submit the job
    sbatch $JOB_SCRIPT

rm -v run_threshold_*.sh
done
