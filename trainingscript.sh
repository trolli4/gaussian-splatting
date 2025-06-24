#!/bin/bash
#SBATCH --partition=mlgpu_short
#SBATCH --time=3:00:00
#SBATCH --gpus=1
#SBATCH --account=ag_ifi_laehner
#SBATCH --job-name=gs_train

# Activate environment
source activate gaussian_splatting

# debug
which python
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.__version__, torch.version.cuda)"
module load CUDA/11.8.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
nvcc --version

# Run training
CUDA_LAUNCH_BLOCKING=1 python /home/s76mfroe_hpc/gaussian-splatting/train.py \
    -s /home/s76mfroe_hpc/nerf-360-scenes/garden
