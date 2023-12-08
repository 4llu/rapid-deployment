#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=01:55:00
#SBATCH --mem=8G
#SBATCH --array=1-2

module restore torch
source activate env/

srun python src/result_generation.py --config arotor/result_generation_base --config_override_base arotor/result_generation/config_override --array_task_id $SLURM_ARRAY_TASK_ID --block_size 3 --repetitions 3 --ensemble_size 5 --job_name "arotor-$SLURM_ARRAY_JOB_ID"
