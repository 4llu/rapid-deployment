#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=03:55:00
#SBATCH --mem=24G

module restore torch
source activate env/

srun python src/h_opt.py --config arotor_replication/anomality_detection_h_opt --n_trials 30 --n_repetitions 3
