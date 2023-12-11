#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
#SBATCH --mem=8G

module restore torch
source activate env/

srun python src/anomality_detection_training.py --config arotor/anomality_detection --repetitions 25 --job_id $SLURM_JOB_ID
