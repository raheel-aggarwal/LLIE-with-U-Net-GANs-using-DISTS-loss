#!/bin/bash 
#SBATCH -N 1 
#SBATCH --job-name=test
#SBATCH --ntasks-per-node=1 
#SBATCH --gres=gpu:1    
#SBATCH --error=job.%x.err 
#SBATCH --output=job.%x.out 
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00 
#SBATCH --partition=l40 
#SBATCH --qos=l40  

cd ~/workspace/LLIE-replication

source ~/.bashrc
conda activate llie

python scripts/training.py --config config.yaml --skip_cache_check
