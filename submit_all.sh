#!/bin/bash
for weight in $(seq 0 0.1 1.0); do
    sbatch --job-name="w${weight}" run_gs_job.sh "$weight"
done
