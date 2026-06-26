#!/bin/bash

#SBATCH --job-name=dataset-timecourse      # Job name
#SBATCH --ntasks=4                                   # Number of tasks (processes)
#SBATCH --output=phenocoder-data-%j.out               # Standard output (%j = job ID)
#SBATCH --error=phenocoder-data-%j.err                # Standard error (%j = job ID)
#SBATCH --mem=50G                                    # Memory per node
#SBATCH --qos=1d                                    #short queue

CHANNELS=("01")

DIR_INPUT=/pstore/data/ihb-tumoroid/data/processed/timecourse/001/001-01/TIF_OVR_BG
DIR_OUTPUT=/pstore/data/ihb-tumoroid/data/processed/timecourse/phenocoder-001-01
DIR_SEG=/pstore/data/ihb-tumoroid/data/processed/timecourse/001/001-01/features/nuclei/TIF_OVR_BG
QC_PATH=/pstore/data/ihb-tumoroid/data/processed/timecourse/001/001-01/001-01_qc.csv
echo "DIR_INPUT: $DIR_INPUT"
echo "DIR_OUTPUT: $DIR_OUTPUT"
echo "DIR_OUTPUT: $DIR_SEG"
echo "QC: $QC_PATH"
python /pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen/whole_mount_tumoroid/phenocoder/dataset.py $DIR_INPUT $DIR_OUTPUT --dir_segmented $DIR_SEG --qc_path $QC_PATH --patch_mode "segmented" --n_patches 500000 --channels "${CHANNELS[@]}" --max_workers 4

DIR_INPUT=/pstore/data/ihb-tumoroid/data/processed/timecourse/002/002-01/TIF_OVR_BG
DIR_OUTPUT=/pstore/data/ihb-tumoroid/data/processed/timecourse/phenocoder/002-01
DIR_SEG=/pstore/data/ihb-tumoroid/data/processed/timecourse/002/002-01/features/nuclei/TIF_OVR_BG
QC_PATH=/pstore/data/ihb-tumoroid/data/processed/timecourse/002/002-01/002-01_qc.csv
echo "DIR_INPUT: $DIR_INPUT"
echo "DIR_OUTPUT: $DIR_OUTPUT"
echo "DIR_OUTPUT: $DIR_SEG"
echo "QC: $QC_PATH"
python /pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen/whole_mount_tumoroid/phenocoder/dataset.py $DIR_INPUT $DIR_OUTPUT --dir_segmented $DIR_SEG --qc_path $QC_PATH --patch_mode "segmented" --n_patches 500000 --channels "${CHANNELS[@]}" --max_workers 4

DIR_INPUT=/pstore/data/ihb-tumoroid/data/processed/timecourse/003/003-01/TIF_OVR_BG
DIR_OUTPUT=/pstore/data/ihb-tumoroid/data/processed/timecourse/phenocoder/003-01
DIR_SEG=/pstore/data/ihb-tumoroid/data/processed/timecourse/003/003-01/features/nuclei/TIF_OVR_BG
QC_PATH=/pstore/data/ihb-tumoroid/data/processed/timecourse/003/003-01/003-01_qc.csv
echo "DIR_INPUT: $DIR_INPUT"
echo "DIR_OUTPUT: $DIR_OUTPUT"
echo "DIR_OUTPUT: $DIR_SEG"
echo "QC: $QC_PATH"
python /pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen/whole_mount_tumoroid/phenocoder/dataset.py $DIR_INPUT $DIR_OUTPUT --dir_segmented $DIR_SEG --qc_path $QC_PATH --patch_mode "segmented" --n_patches 500000 --channels "${CHANNELS[@]}" --max_workers 4

DIR_INPUT=/pstore/data/ihb-tumoroid/data/processed/timecourse/004/004-01/TIF_OVR_BG
DIR_OUTPUT=/pstore/data/ihb-tumoroid/data/processed/timecourse/phenocoder/004-01
DIR_SEG=/pstore/data/ihb-tumoroid/data/processed/timecourse/004/004-01/features/nuclei/TIF_OVR_BG
QC_PATH=/pstore/data/ihb-tumoroid/data/processed/timecourse/004/004-01/004-01_qc.csv
echo "DIR_INPUT: $DIR_INPUT"
echo "DIR_OUTPUT: $DIR_OUTPUT"
echo "DIR_OUTPUT: $DIR_SEG"
echo "QC: $QC_PATH"
python /pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen/whole_mount_tumoroid/phenocoder/dataset.py $DIR_INPUT $DIR_OUTPUT --dir_segmented $DIR_SEG --qc_path $QC_PATH --patch_mode "segmented" --n_patches 500000  --channels "${CHANNELS[@]}" --max_workers 4

DIR_INPUT=/pstore/data/ihb-tumoroid/data/processed/timecourse/005/005-01/TIF_OVR_BG
DIR_OUTPUT=/pstore/data/ihb-tumoroid/data/processed/timecourse/phenocoder/005-01
DIR_SEG=/pstore/data/ihb-tumoroid/data/processed/timecourse/005/005-01/features/nuclei/TIF_OVR_BG
QC_PATH=/pstore/data/ihb-tumoroid/data/processed/timecourse/005/005-01/005-01_qc.csv
echo "DIR_INPUT: $DIR_INPUT"
echo "DIR_OUTPUT: $DIR_OUTPUT"
echo "DIR_OUTPUT: $DIR_SEG"
echo "QC: $QC_PATH"
python /pstore/data/ihb-g-deco/USERS/schulzp9/git/tumoroid_screen/whole_mount_tumoroid/phenocoder/dataset.py $DIR_INPUT $DIR_OUTPUT --dir_segmented $DIR_SEG --qc_path $QC_PATH --patch_mode "segmented" --n_patches 500000 --channels "${CHANNELS[@]}" --max_workers 4