#!/bin/bash
#SBATCH --partition=mlgpu_devel
#SBATCH --time=1:00:00
#SBATCH --gpus=1
# #SBATCH --mem-per-gpu=24
#SBATCH --account=ag_ifi_laehner


module load Miniforge3
module load CUDA/11.8.0
module load CMake


# clean-up old conda env
conda env remove --name gaussian_splatting -y

# create new conda env
conda env create --file /lustre/scratch/data/s76mfroe_hpc-bpg_gaussian_splatting/gaussian-splatting/environment.yml
conda activate gaussian_splatting

# run training script
python ./gaussian-splatting/train.py -s /lustre/scratch/data/s76mfroe_hpc-bpg_gaussian_splatting/nerf-360-scenes/garden
