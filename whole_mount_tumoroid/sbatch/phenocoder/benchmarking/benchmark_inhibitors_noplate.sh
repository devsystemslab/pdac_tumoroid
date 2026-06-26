#!/bin/bash
#SBATCH --job-name=benchmark_inhibitors
#SBATCH --array=1-32
#SBATCH --ntasks=1
#SBATCH --qos=1d
#SBATCH --output=benchmark_inhibitors-%A-%a.out
#SBATCH --error=benchmark_inhibitors-%A-%a.err
#SBATCH --mem=50G

BATCHES=$(seq 0 31)
PARAMS=$(echo $BATCHES | cut -d' ' -f${SLURM_ARRAY_TASK_ID})
SCRIPT=/pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen/whole_mount_tumoroid/analysis/inhibitors/run_benchmark.py
python $SCRIPT --batch $PARAMS --batch_size=24