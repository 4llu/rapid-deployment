#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=01:55:00
#SBATCH --mem=24G

module restore torch
source activate env/

srun python src/anomality_detection_training.py --config arotor_replication/anomality_detection --repetitions 50 --job_id $SLURM_JOB_ID
