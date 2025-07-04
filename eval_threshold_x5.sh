#!/bin/bash
[ ! -d "./logs/" ] && mkdir "./logs/"

# fill test_iterations with all iterations to compute PSNR at
iterations_to_test="1000"
for i in $(seq 2000 1000 30000); do
     iterations_to_test+=" $i"
done

# Loop 5 times to remove statistical influence
for turn in {1..$1..1}
do

# Loop over threshold values from 20 to 110 in steps of 10
for threshold in {1..20..1}
do
    # Create a unique job script for this threshold
    JOB_SCRIPT="run_threshold_${threshold}.sh"

    cat <<EOF > $JOB_SCRIPT
#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=3:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_${turn}_${threshold}
#SBATCH --output=logs/turn_${turn}/garden_${threshold}.out

# Activate environment
source activate gaussian_splatting

# Run training with custom threshold
python /home/s76mfroe_hpc/gaussian-splatting/train.py \\
    -s /home/s76mfroe_hpc/nerf-360-scenes/garden \\
    -m /lustre/scratch/data/s76mfroe_hpc-bpg_gaussian_splatting/gaussian-splatting/output/turn_${turn}/garden_${threshold} \\
    --densify_error_threshold ${threshold} \
    --test_iterations $iterations_to_test

EOF

    # Submit the job
    sbatch $JOB_SCRIPT

rm -v run_threshold_*.sh
done
done
