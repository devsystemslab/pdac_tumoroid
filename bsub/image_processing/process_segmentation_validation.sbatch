#!/bin/bash
#SBATCH --job-name=dask_scheduler
#SBATCH --cpus-per-task=1
#SBATCH -N 1
#SBATCH --qos 1d
#SBATCH --mem=64GB
#SBATCH -o segmentation_validation-%j.out
#SBATCH -e segmentation_validation-%j.err
export PYTHONPATH="${PYTHONPATH}:/pstore/home/harmelc/tumoroid_screen"
METADATA=whole_mount_tumoroid/metafiles/imaging_data_overview.csv
CONFIG=whole_mount_tumoroid/configs/config_segmentation_validation.yaml
PLATE='001'
echo "Processing plate $PLATE"
python whole_mount_tumoroid/image_processing/cli.py $PLATE --config $CONFIG
