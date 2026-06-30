#!/bin/bash
#SBATCH --job-name=dask_scheduler
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH --qos 1d
#SBATCH --mem=64GB
#SBATCH -o segmentation_validation-%j.out
#SBATCH -e segmentation_validation-%j.err
export PYTHONPATH="${PYTHONPATH}:."
METADATA=metafiles/imaging_data_overview.csv
CONFIG=configs/config_segmentation_validation.yaml
PLATE='001'
echo "Processing plate $PLATE"
python image_processing/cli.py $PLATE --config $CONFIG
