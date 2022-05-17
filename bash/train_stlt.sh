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
#     [Hopper]   - If Y, then include Hopper in the set of objects
#     [Samples]  - Number of frames to get
#     [Stride]   - Chosen Stride
#     [Batch]    - Batch-Size
#     [Rate]     - Learning Rate
#     [Epochs]   - Maximum Number of Epochs to train for
#     [Warmup]   - Number of Warmup Epochs
#     [Offset]   - Offset from base data location to retrieve the data splits
#
#  USAGE:
#     srun --time=23:00:00 --gres=gpu:1 --mem=30G --partition=apollo --nodelist=apollo1 bash/train_stlt.sh 4 8 Y Y 12 2 64 0.000001 60 2 Fixed &> ~logfile.log
#     * N.B.: The above should be run from the root STLT directory.
#     * N.B.: It may be that this requires specifying extra memory with the --mem=30G option

#  Data Structures
#    Data is expected to be under ${HOME}/data/behaviour/ which follows the definitions laid out
#        in my Jupyter notebook.

# Do some Calculations/Preprocessing
SPATIAL=${1}
TEMPORAL=${2}
IDENTITY=${3,,}
HOPPER=${4,,}
SAMPLES=${5}
STRIDE=${6}

BATCH=${7}
LR=${8}
EPOCHS=${9}
WARMUP=${10}

OFFSET=${11}

# Define Output Name and associated directories
OUT_NAME=A[${SPATIAL}-${TEMPORAL}-${IDENTITY^}-${HOPPER^}]_D[${SAMPLES}_${STRIDE}]_L[${BATCH}_${LR}_${EPOCHS}_${WARMUP}]_STLT
SCRATCH_HOME=/disk/scratch/${USER}
SCRATCH_DATA=${SCRATCH_HOME}/data/behaviour/
SCRATCH_MODELS=${SCRATCH_HOME}/models/stlt/${OUT_NAME}

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
mkdir -p ${SCRATCH_HOME}
echo ""

# ================================
# Download Data and Models if necessary
# ================================
echo " ===================================="
echo "Consolidating Data in ${SCRATCH_HOME}"
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
mkdir -p ${SCRATCH_MODELS}

if [ "${IDENTITY}" = "y" ]; then
  IDS="--maintain_identities"
else
  IDS=""
fi
if [ "${HOPPER}" = "y" ]; then
  HOP="--include_hopper"
else
  HOP=""
fi
python src/train.py \
  --dataset_name mouse --dataset_type layout --model_name stlt $IDS $HOP \
  --labels_path "${SCRATCH_DATA}/STLT.Schema.json" \
  --train_dataset_path "${SCRATCH_DATA}/Train/STLT.Annotations.json"  \
  --val_dataset_path "${SCRATCH_DATA}/Validate/STLT.Annotations.json" \
  --save_model_path "${SCRATCH_MODELS}/${OUT_NAME}.pth" \
  --layout_samples ${SAMPLES} --layout_stride ${STRIDE} --video_size 1280 720 \
  --num_spatial_layers ${SPATIAL} --num_temporal_layers ${TEMPORAL} \
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