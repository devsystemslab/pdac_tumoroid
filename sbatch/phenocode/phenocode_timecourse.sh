#!/bin/bash

#SBATCH --job-name=phenocode-run-timecourse      # Job name
#SBATCH --ntasks=1                                
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=phenocoder-run-%j.out            # Standard output (%j = job ID)
#SBATCH --error=phenocoder-run-%j.err             # Standard error (%j = job ID)
#SBATCH --gres=gpu:1                                # GPU
#SBATCH --partition=batch_gpu                       # GPU partition
#SBATCH --qos=1d

python analysis/timecourse/run_phenocoder.py
