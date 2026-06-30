#!/bin/bash

#SBATCH --job-name=train-tumoroidscreen_3      # Job name
#SBATCH --ntasks=8                                # Number of tasks (processes)
#SBATCH --output=phenocoder-train-%j.out            # Standard output (%j = job ID)
#SBATCH --error=phenocoder-train-%j.err             # Standard error (%j = job ID)
#SBATCH --mem=50G                                   # Memory per node
#SBATCH --gres=gpu:1                                # GPU
#SBATCH --partition=batch_gpu                       # GPU partition
#SBATCH --qos=3d

TMP_DIR=/home/schulzp9/scratch/datasets/pilotscreen/2nd_cycle_nuclei
#mkdir -p $TMP_DIR
echo "TMP_DIR: $TMP_DIR"
DATA_DIR=data/pilotscreen/phenocoder/2nd_cycle_nuclei
echo "DATA_DIR: $DATA_DIR"
# copy DATA_DIR to scratch
#echo "Copying $DATA_DIR to $TMP_DIR"
#rclone copy $DATA_DIR $TMP_DIR -P
# beta 0.01
echo "Training with beta 0.01..."
python phenocoder/train.py $TMP_DIR --n_dense_dim 32 --n_latent_dim 16  --n_epochs 10 --conditional --ignore_dataset_condition --plates 004 --n_workers 4 --beta 0.01
python phenocoder/train.py $TMP_DIR --n_dense_dim 64 --n_latent_dim 32 --n_epochs 10 --conditional --ignore_dataset_condition --plates 004 --n_workers 4 --beta 0.01
#python phenocoder/train.py $TMP_DIR --n_dense_dim 128 --n_latent_dim 64 --n_epochs 10 --conditional --ignore_dataset_condition --plates 004 --n_workers 4 --beta 0.01
#python phenocoder/train.py $TMP_DIR --n_dense_dim 256 --n_latent_dim 128 --n_epochs 10 --conditional --ignore_dataset_condition --plates 004 --n_workers 4 --beta 0.01

# move models and tensorboard logs back to DATA_DIR
echo "Moving models and tensorboard logs back to $DATA_DIR"
rclone copy $TMP_DIR/models $DATA_DIR/models
rclone copy $TMP_DIR/tensorboard_logs $DATA_DIR/tensorboard_logs
# clean up
#echo "Cleaning up $TMP_DIR"
#rclone purge $TMP_DIR
