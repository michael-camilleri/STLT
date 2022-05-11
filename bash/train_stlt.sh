#!/bin/bash
#  Author: Michael Camilleri
#
#  Scope:
#     Trains the STLT Model on own data
#
#  Script takes the following parameters:
#     [Spatial]  - Number of Spatial Layers
#     [Temporal] - Number of Temporal Layers
#     [Identity] - If Y, then maintain the identity of the cage-mates (with augmentation)
#     [Batch]    - Batch-Size
#     [Rate]     - Learning Rate
#     [Epochs]   - Maximum Number of Epochs to train for
#     [Warmup]   - Number of Warmup Epochs
#     [Offset]   - Offset from base data location to retrieve the data splits
#
#  USAGE:
#     srun --time=23:00:00 --gres=gpu:1 --partition=apollo --nodelist=apollo1 bash/train_stlt.sh 64 0.00005 50 Fixed 2 4 &> ~/logs/train_stlt.00005.Fixed.out
#     * N.B.: The above should be run from the root STLT directory.

#  Data Structures
#    Data is expected to be under ${HOME}/data/behaviour/ which follows the definitions laid out
#        in my Jupyter notebook.

# Do some Calculations/Preprocessing
SPATIAL=${1}
TEMPORAL=${2}
IDENTITY=${3,,}

BATCH=${4}
LR=${5}
EPOCHS=${6}
WARMUP=${7}

OFFSET=${8}


OUT_NAME=A[${SPATIAL}-${TEMPORAL}-${IDENTITY^}]_D[25_1]_L[${BATCH}_${LR}_${EPOCHS}_${WARMUP}]_STLT

# ===================
# Environment setup
# ===================
echo "Setting up Conda enviroment on ${SLURM_JOB_NODELIST}: Config=${OUT_NAME}"
set -e # Make script bail out after first error
source activate py3stlt   # Activate Conda Environment
echo "Libraries from: ${LD_LIBRARY_PATH}"

# Setup NCCL Debug Status
export NCCL_DEBUG=INFO

# Make your own folder on the node's scratch disk
SCRATCH_HOME=/disk/scratch/${USER}
mkdir -p ${SCRATCH_HOME}
echo ""

# ================================
# Download Data and Models if necessary
# ================================
echo " ===================================="
echo "Consolidating Data in ${SCRATCH_HOME}"
SCRATCH_DATA=${SCRATCH_HOME}/data/behaviour/
mkdir -p ${SCRATCH_DATA}
echo "  -> Synchronising Data"
echo "     .. Schemas .."
cp ${HOME}/data/behaviour/Common/STLT* ${SCRATCH_DATA}/
echo "     .. Annotations .."
rsync --archive --update --compress --include '*/' --include 'STLT*' --exclude '*' \
      --info=progress2 ${HOME}/data/behaviour/Train/${OFFSET}/ ${SCRATCH_DATA}
echo "   ----- DONE -----"
mail -s "Train_STLT on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Synchronised Data and Models."
echo ""

# ===========
# Train Model
# ===========
echo " ===================================="
echo " Training Model ${OUT_NAME}"
SCRATCH_MODELS=${SCRATCH_HOME}/models/stlt
mkdir -p ${SCRATCH_MODELS}

if [ "${IDENTITY}" = "y" ]; then
  IDS="--maintain_identities"
else
  IDS=""
fi
python src/train.py \
  --dataset_name mouse --dataset_type layout --model_name stlt $IDS \
  --labels_path "${SCRATCH_DATA}/STLT.Schema.json" \
  --videoid2size_path "${SCRATCH_DATA}/STLT.Sizes.json"  \
  --train_dataset_path "${SCRATCH_DATA}/Train/STLT.Annotations.json"  \
  --val_dataset_path "${SCRATCH_DATA}/Validate/STLT.Annotations.json" \
  --save_model_path "${SCRATCH_MODELS}/${OUT_NAME}.pth" \
  --layout_num_frames 25 --num_spatial_layers ${SPATIAL} --num_temporal_layers ${TEMPORAL} \
  --batch_size ${BATCH} --learning_rate ${LR} --weight_decay 1e-5 --clip_val 5.0 \
  --epochs ${EPOCHS} --warmup_epochs ${WARMUP} --num_workers 2 --select_best top1
echo "   == Training Done =="
mail -s "Train_STLT on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Model Training Completed."
echo ""

# ===========
# Copy Data
# ===========
echo " ===================================="
echo " Copying Model Weights (as ${OUT_NAME})"
mkdir -p "${HOME}/models/STLT/Trained/"
rsync --archive --compress --info=progress2 "${SCRATCH_MODELS}/" "${HOME}/models/STLT/Trained/"
rm -rf ${SCRATCH_MODELS}
echo "   ++ ALL DONE! Hurray! ++"
mail -s "Train_STLT on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Output Models copied as '${HOME}/models/STLT/Trained/${OUT_NAME}.pth'."
conda deactivate