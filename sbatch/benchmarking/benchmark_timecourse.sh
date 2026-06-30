#!/bin/bash
#SBATCH --job-name=benchmark_pilotscreen
#SBATCH --array=1-32
#SBATCH --ntasks=1
#SBATCH --qos=1d
#SBATCH --output=benchmark_pilotscreen-%A-%a.out
#SBATCH --error=benchmark_pilotscreen-%A-%a.err
#SBATCH --mem=50G

BATCHES=$(seq 0 31)
PARAMS=$(echo $BATCHES | cut -d' ' -f${SLURM_ARRAY_TASK_ID})
SCRIPT=analysis/timecourse/run_benchmark.py
python $SCRIPT --batch $PARAMS --batch_size=24
