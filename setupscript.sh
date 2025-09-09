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
env_name="gaussian_splatting_opacity_reset_only"

if conda info --envs | grep -q "$env_name"; then
conda env remove --name "$env_name" -y;
fi

# Create environment
conda env create --file environment.yml

# Activate environment
source activate gaussian_splatting_opacity_reset_only

# Install C++/CUDA submodules (after torch is installed)
pip install /home/s76mfroe_hpc/gaussian-splatting/submodules/diff-gaussian-rasterization \
            /home/s76mfroe_hpc/gaussian-splatting/submodules/simple-knn \
            /home/s76mfroe_hpc/gaussian-splatting/submodules/fused-ssim
