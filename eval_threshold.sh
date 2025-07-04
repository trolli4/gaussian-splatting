#!/bin/bash
[ ! -d "./logs/" ] && mkdir "./logs/"

iterations_to_test="1000"
for i in $(seq 2000 1000 30000); do
     iterations_to_test+=" $i"
done

# Loop over threshold values from 20 to 110 in steps of 10
for counter in {1..10..1}
do
    threshold=$(echo "scale=2; $counter / 10" | bc)
    # Create a unique job script for this threshold
    JOB_SCRIPT="run_threshold_${threshold}.sh"

    cat <<EOF > $JOB_SCRIPT
#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=3:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_train_${counter}
#SBATCH --output=logs/garden_${counter}.out

# Activate environment
source activate gaussian_splatting

# Run training with custom threshold
CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/train.py \\
    -s /home/s76mfroe_hpc/nerf-360-scenes/garden \\
    -m output/garden_${threshold} \\
    # --quiet \\
    --eval \\
    --densify_error_threshold ${threshold} \
    --test_iterations $iterations_to_test

CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/render.py \
    -m output/garden_${counter}

CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/metrics.py \
    -m output/garden_${counter}
EOF

    # Submit the job
    sbatch $JOB_SCRIPT

rm -v run_threshold_*.sh
done
