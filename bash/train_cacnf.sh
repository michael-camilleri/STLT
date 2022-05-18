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
#   -- Data Format Parameters --
#     [Layout Samples] - Number of samples in layout stream
#     [Layout Stride] - Stride for layout sampling
#     [Resolution] - Image Height Size
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
#     srun --time=4-23:00:00 --gres=gpu:1 --mem=40G --partition=apollo --nodelist=apollo1 bash/train_cacnf.sh 4 8 4 4 36 3 256 1 4 0.0000001 50 2 Fixed N &> ~/logs/train_cacnf_36+3_1e-7.log
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

LAYOUT_SAMPLES=${5}
LAYOUT_STRIDE=${6}
RESOLUTION=${7}
RESIZE_CROP=${8}

BATCH_SIZE=${9}
LR=${10}
MAX_EPOCHS=${11}
WARMUP_ITER=${12}

PATH_OFFSET=${13}
FORCE_FRAMES=${14,,}

# Derivative Values
ARCHITECTURE="A[${SPATIAL}-${TEMPORAL}-${APPEARANCE}-${FUSION}-Y-Y]"
DATA_FORMAT="D[${LAYOUT_SAMPLES}_${LAYOUT_STRIDE}-25-${RESOLUTION}_${RESIZE_CROP}]"
LEARNING="L[${BATCH_SIZE}_${LR}_${MAX_EPOCHS}_${WARMUP_ITER}]"
OUT_NAME=${ARCHITECTURE}_${DATA_FORMAT}_${LEARNING}_CAF

# Path Values
SCRATCH_HOME=/disk/scratch/${USER}
SCRATCH_DATA=${SCRATCH_HOME}/data/behaviour
OUTPUT_DIR="${HOME}/models/CACNF/Trained/${PATH_OFFSET}/"

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
echo " Training Model (${OUT_NAME})"
mkdir -p "${OUTPUT_DIR}"

python src/train.py  \
  --dataset_name something --dataset_type multimodal --model_name cacnf \
  --maintain_identities --include_hopper --video_size 1280 720 \
  --train_dataset_path "${SCRATCH_DATA}/Train/STLT.Annotations.json" \
  --val_dataset_path "${SCRATCH_DATA}/Validate/STLT.Annotations.json" \
  --labels_path "${SCRATCH_DATA}/STLT.Schema.json" \
  --videos_path "${SCRATCH_DATA}/Frames" \
  --resnet_model_path "${HOME}/models/CACNF/Base/r3d50_KMS_200ep.pth" \
  --save_model_path "${OUTPUT_DIR}/${OUT_NAME}.pth" \
  --layout_samples "${LAYOUT_SAMPLES}" --layout_stride "${LAYOUT_STRIDE}" \
  --appearance_num_frames 25 --resize_height "${RESOLUTION}" --crop_scale "${RESIZE_CROP}" \
  --num_spatial_layers "${SPATIAL}" --num_temporal_layers "${TEMPORAL}" \
  --num_appearance_layers "${APPEARANCE}" --num_fusion_layers "${FUSION}" \
  --normaliser_mean 69.201 69.201 69.201 --normaliser_std 58.571 58.571 58.571 \
  --batch_size "${BATCH_SIZE}" --learning_rate "${LR}" --weight_decay 1e-5 --clip_val 5.0 \
  --epochs "${MAX_EPOCHS}" --warmup_epochs "${WARMUP_ITER}"  \
  --select_best top1 --which_score caf --num_workers 2
echo "   == Training Done =="
echo ""

# ===========
# Nothing to copy, since I save directly to output disk
# ===========
echo "   ++ ALL DONE! Hurray! ++"
conda deactivate