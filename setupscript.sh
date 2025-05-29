#!/bin/bash
#SBATCH --partition=mlgpu_devel
#SBATCH --time=1:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_train

module purge
module load Miniforge3
module load CMake
module load CUDA/11.8.0

# Set CUDA path explicitly if needed
export CUDA_HOME=$CUDA_HOME

# Clean existing env
conda env remove --name gaussian_splatting -y

# Create environment
conda env create --file /lustre/scratch/data/s76mfroe_hpc-bpg_gaussian_splatting/gaussian-splatting/environment.yml

# Activate environment
source activate gaussian_splatting

# Install C++/CUDA submodules (after torch is installed)
pip install /lustre/scratch/data/s76mfroe_hpc-bpg_gaussian_splatting/gaussian-splatting/submodules/diff-gaussian-rasterization \
            /lustre/scratch/data/s76mfroe_hpc-bpg_gaussian_splatting/gaussian-splatting/submodules/simple-knn \
            /lustre/scratch/data/s76mfroe_hpc-bpg_gaussian_splatting/gaussian-splatting/submodules/fused-ssim
