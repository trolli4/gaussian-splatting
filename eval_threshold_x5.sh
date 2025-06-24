#!/bin/bash
[ ! -d "./logs/" ] && mkdir "./logs/"

# Loop 5 times to remove statistical influence
for turn in {1..5..1}
do

# Loop over threshold values from 20 to 110 in steps of 10
for threshold in {20..110..10}
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
#SBATCH --output=logs/turn_${turn}/garden_${threshold}.out

# Activate environment
source activate gaussian_splatting

# Run training with custom threshold
python /home/s76mfroe_hpc/gaussian-splatting/train.py \\
    -s /home/s76mfroe_hpc/nerf-360-scenes/garden \\
    -m output/turn_${turn}/garden_${threshold} \\
    --quiet \\
    --eval \\
    --densify_error_threshold ${threshold}

CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/render.py \
    -m output/turn_${turn}/garden_${threshold} \

CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/metrics.py \
    -m output/turn_${turn}/garden_${threshold}
EOF

    # Submit the job
    sbatch $JOB_SCRIPT

rm -v run_threshold_*.sh
done
done
