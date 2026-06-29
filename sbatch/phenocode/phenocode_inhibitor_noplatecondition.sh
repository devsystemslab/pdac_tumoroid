#!/bin/bash

#SBATCH --job-name=004_noplate      # Job name
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=phenocoder-run-%j.out            # Standard output (%j = job ID)
#SBATCH --error=phenocoder-run-%j.err             # Standard error (%j = job ID)
#SBATCH --gres=gpu:1                                # GPU
#SBATCH --partition=batch_gpu                       # GPU partition
#SBATCH --qos=3d

python /pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen/whole_mount_tumoroid/analysis/pilotscreen/run_phenocoder.py
