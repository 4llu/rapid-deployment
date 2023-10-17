#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=02:55:00
#SBATCH --mem=8G

module restore torch
source activate env/

srun python src/h_opt.py --config arotor/anomality_detection_h_opt --n_trials 100 --n_repetitions 5