#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=02:55:00
#SBATCH --mem=8G

module restore torch
source activate env/

srun python src/h_opt.py --config arotor_replication/h_opt_mixed --n_trials 20 --n_repetitions 3
