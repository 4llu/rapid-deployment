#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=02:55:00
#SBATCH --mem=8G

module restore torch
source activate env/

srun python src/anomality_detection_training.py --config arotor_replication/anomality_detection --repetitions 10 --job_id $SLURM_JOB_ID
