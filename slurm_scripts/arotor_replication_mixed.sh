#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=02:55:00
#SBATCH --mem=8G
#SBATCH --array=1-20

module restore torch
source activate env/

srun python src/result_generation.py --config arotor_replication/result_generation_mixed_base --config_override_base arotor_replication/result_generation/config_override --array_task_id $SLURM_ARRAY_TASK_ID --block_size 5 --repetitions 9 --job_name "arotor_replication_mixed-$SLURM_ARRAY_JOB_ID"
