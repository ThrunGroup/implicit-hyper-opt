#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH -p gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --job-name=%A_%a
#SBATCH --output=slurm_out/slurm_%A_%a.out

export LD_LIBRARY_PATH=/pkgs/cuda-9.2/lib64:$LD_LIBRARY_PATH
#export PATH=/pkgs/anaconda3/bin:$PATH
#

conda activate ift-env

echo "${SLURM_ARRAY_TASK_ID}"

python train_augment_net_slurm.py --deploy_num "${SLURM_ARRAY_TASK_ID}"

# srun -p gpu --gres=gpu:1 --mem=4GB python train_augment_net_slurm.py --deploy_num 2
# sbatch --array=0-29%4 srun_script.sh