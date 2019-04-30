#!/bin/bash
#SBATCH --account=def-aloise

# RESSOURCES
#SBATCH --cpus-per-task=32		# Number of CPUs
#SBATCH --gres=gpu:2			# Number of GPUs
#SBATCH --mem=127000M			# Memory
#SBATCH --time=01-00:00			# Brackets: 3h, 12h, 1d, 3d, 7d 

# JOB SPECIFICATION
#SBATCH --job-name=uji
#SBATCH --output=/home/qfournie/logs/%x-%j

# ARGS
#SBATCH --array=1,3,5,7,9,11,13,15,17,19,21,23,25,27,29

# LOAD MODULES
module load python/3.5
module load cuda/9.0
module load cudnn/7.0

# LOAD VIRTUAL ENVIRONMENT
source ~/keras-env/bin/activate

# TASK
cd /home/qfournie/dimensionality_reduction
python3 main.py -d uji -t dimension -c knn --start_dim $SLURM_ARRAY_TASK_ID --n_dim 1
