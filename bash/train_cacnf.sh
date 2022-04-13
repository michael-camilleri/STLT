#!/bin/bash
#  Author: Michael Camilleri
#
#  Scope:
#     Trains the CACNF Model on own data
#
#  Script takes the following parameters:
#   -- Architecture Parameters --
#     [Spatial]  - Number of Spatial Layers
#     [Temporal] - Number of Temporal Layers
#     [Appearance] - Number of Appearance Layers (on top of ResNet)
#     [Fusion] - Number of Fusion Layers
#     [Resolution] - Image Height Size
#   -- Features --
#     [Resize]   - Random-Resize augmentation (float)
#   -- Training Parameters --
#     [Batch]    - Batch-Size
#     [Rate]     - Learning Rate
#     [Epochs]   - Maximum Number of Training Epochs
#     [Warmup]   - Number of Warmup Epochs
#   -- Paths/Setup --
#     [Offset]   - Offset from base data location to retrieve the data splits
#     [Frames]   - Y/N: Indicates if Frames should be rsynced: this is done to save time if it is
#                       known that the machine contains the right data splits.

#
#  USAGE:
#     srun --time=2:00:00 --gres=gpu:1 --partition=apollo --nodelist=apollo2 bash/train_cacnf.sh 4 8 4 4 32 1.0 64 0.00001 1 1 Fixed N &> ~/logs/train_debug_1.log
#     * N.B.: The above should be run from the root STLT directory.

#  Data Structures
#    Data is expected to be under ${HOME}/data/behaviour/ which follows the definitions laid out
#        in my Jupyter notebook.

# Do some Calculations/Preprocessing
####  Some Configurations
# Get and store the main Parameters
SPATIAL=${1}
TEMPORAL=${2}
APPEARANCE=${3}
FUSION=${4}
RESOLUTION=${5}

RESIZE_CROP=${6}

BATCH_SIZE=${7}
LR=${8}
MAX_EPOCHS=${9}
WARMUP_ITER=${10}

PATH_OFFSET=${11}
FORCE_FRAMES=${12,,}

# Derivative Values
OUT_NAME=A[${SPATIAL}-${TEMPORAL}-${APPEARANCE}-${FUSION}]_I[${RESOLUTION}_${RESIZE_CROP}]_L[${BATCH_SIZE}_${LR}_${MAX_EPOCHS}_${WARMUP_ITER}]_CAF

# Path Values
SCRATCH_HOME=/disk/scratch/${USER}
SCRATCH_DATA=${SCRATCH_HOME}/data/behaviour
OUTPUT_DIR="${HOME}/models/CACNF/Trained/${PATH_OFFSET}"

# ===================
# Environment setup
# ===================
echo "Setting up Conda enviroment on ${SLURM_JOB_NODELIST}: Config=${OUT_NAME}"
set -e # Make script bail out after first error
source activate py3stlt   # Activate Conda Environment
echo "Libraries from: ${LD_LIBRARY_PATH}"

# Setup NCCL Debug Status
export NCCL_DEBUG=INFO

# Make own folder on the node's scratch disk
mkdir -p "${SCRATCH_HOME}"
echo ""

# ================================
# Download Data and Models if necessary
# ================================
echo " ===================================="
echo "Consolidating Data in ${SCRATCH_HOME}"
mkdir -p "${SCRATCH_DATA}"
echo "  -> Synchronising Data"
echo "     .. Schemas .."
rsync --archive --update --compress --include 'STLT*' --exclude '*' \
      --info=progress2 ${HOME}/data/behaviour/Common/ ${SCRATCH_DATA}
echo "     .. Annotations .."
rsync --archive --update --compress --include '*/' --include 'STLT*' --exclude '*' \
      --info=progress2 ${HOME}/data/behaviour/Train/${PATH_OFFSET}/ ${SCRATCH_DATA}
if [ "${FORCE_FRAMES}" = "y" ]; then
  echo "     .. Frames .."
  rsync --archive --update --info=progress2 ${HOME}/data/behaviour/Train/Frames ${SCRATCH_DATA}/
else
  echo "     .. Skipping Frames .."
fi

# ===========
# Train Model
# ===========
echo " ===================================="
echo " Training Model (BS=${BATCH_SIZE}, LR=${LR}) on ${PATH_OFFSET} for ${MAX_EPOCHS}(${WARMUP_ITER}) epochs"
mkdir -p "${OUTPUT_DIR}"

python src/train.py  \
  --dataset_name something --dataset_type multimodal --model_name cacnf --videos_as_frames \
  --train_dataset_path "${SCRATCH_DATA}/Train/STLT.Annotations.json" \
  --val_dataset_path "${SCRATCH_DATA}/Validate/STLT.Annotations.json" \
  --labels_path "${SCRATCH_DATA}/STLT.Schema.json" \
  --videoid2size_path "${SCRATCH_DATA}/STLT.Sizes.json"  \
  --videos_path "${SCRATCH_DATA}/Frames" \
  --resnet_model_path "${HOME}/models/CACNF/Base/r3d50_KMS_200ep.pth" \
  --save_model_path "${OUTPUT_DIR}/${OUT_NAME}.pth" \
  --layout_num_frames 25 --appearance_num_frames 25 --resize_height "${RESOLUTION}" \
  --num_spatial_layers "${SPATIAL}" --num_temporal_layers "${TEMPORAL}" \
  --num_appearance_layers "${APPEARANCE}" --num_fusion_layers "${FUSION}" \
  --normaliser_mean 69.201 69.201 69.201 --normaliser_std 58.571 58.571 58.571 \
  --batch_size "${BATCH_SIZE}" --learning_rate "${LR}" --weight_decay 1e-5 --clip_val 5.0 \
  --epochs "${MAX_EPOCHS}" --warmup_epochs "${WARMUP_ITER}" --crop_scale "${RESIZE_CROP}" \
  --select_best top1 --which_score caf --num_workers 2
echo "   == Training Done =="
echo ""

# ===========
# Nothing to copy, since I save directly to output disk
# ===========
echo "   ++ ALL DONE! Hurray! ++"
conda deactivate