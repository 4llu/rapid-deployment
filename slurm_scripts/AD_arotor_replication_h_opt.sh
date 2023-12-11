#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=02:55:00
#SBATCH --mem=24G

module restore torch
source activate env/

srun python src/h_opt.py --config arotor_replication/anomality_detection_h_opt --n_trials 20 --n_repetitions 3
