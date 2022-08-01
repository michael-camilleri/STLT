#!/bin/bash
#  Author: Michael Camilleri
#  **OBSOLETE** Deprecated
#  Scope:
#     Infers behaviours according to the CACNF Model
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
#     [App Samples] - Number of samples in appearance stream
#     [App Stride] - Stride for appearance sampling
#     [Resolution] - Image Height Size
#     [Resize]   - Centre-Crop Size

#   -- Loading Parameters --
#     [Batch_size] - Due to a weird architectural strategy, it seems that the batch sizes must match

#   -- Paths/Setup --
#     [Model]    - Model Path to load checkpoint from: note it should NOT contain the .pth extension
#     [DataSet]  - Which Dataset to infer for
#     [Offset]   - Offset from base data location to retrieve the data splits
#     [Frames]   - Name of specific Frames Directory
#     [Force]    - Y/N: Indicates if Frames should be rsynced: this is done to save time if it is
#                       known that the machine contains the right data splits.
#
#  USAGE:
#     srun --time=2:00:00 --gres=gpu:1 --mem=40G --nodelist=charles17 bash/infer_cacnf.sh 4 8 4 4 36 3 12 2 256 1 4 SOTA/trained Train Fixed Frames_Raw_Ext N &> ~/logs/infer_cacnf.SOTA.train.out
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
APP_SAMPLES=${7}
APP_STRIDE=${8}
RESOLUTION=${9}
RESIZE_CROP=${10}

BATCH_SIZE=${11}

MODEL_PATH=${12}
DATASET=${13}
PATH_OFFSET=${14}
FRAMES_DIR=${15}
FORCE_FRAMES=${16,,}

# Derivative Values
if [ "${DATASET,,}" = "test" ]; then
  PARENT_DIR='Test'
else
  PARENT_DIR='Train'
fi

# Path Values
SCRATCH_HOME=/disk/scratch/${USER}
SCRATCH_DATA=${SCRATCH_HOME}/data/behaviour
RESULT_PATH="${HOME}/results/CACNF/${MODEL_PATH}"

# ===================
# Environment setup
# ===================
echo "Setting up Conda enviroment on ${SLURM_JOB_NODELIST}"
echo "Using Model ${MODEL_PATH} on ${DATASET} (from ${PATH_OFFSET} split) with ${FRAMES_DIR} Frames."
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
      --info=progress2 "${HOME}/data/behaviour/Common/" "${SCRATCH_DATA}"
echo "     .. Annotations .."
rsync --archive --update --compress --include '*/' --include 'STLT*' --exclude '*' \
      --info=progress2 "${HOME}/data/behaviour/${PARENT_DIR}/${PATH_OFFSET}/" "${SCRATCH_DATA}"
if [ "${FORCE_FRAMES}" = "y" ]; then
  echo "     .. Frames .."
  rsync --archive --update --info=progress2 "${HOME}/data/behaviour/${PARENT_DIR}/${FRAMES_DIR}" \
        "${SCRATCH_DATA}/${FRAMES_DIR}"
else
  echo "     .. Skipping Frames .."
fi
echo "   ----- DONE -----"
echo ""

# ===========
# Infer Model
# ===========
echo " ===================================="
echo " Inferring Behaviours for ${DATASET} using model ${MODEL_PATH}"
mkdir -p "${RESULT_PATH}"

python src/inference.py \
    --dataset_name mouse --dataset_type multimodal --model_name cacnf \
    --maintain_identities --include_hopper --video_size 1280 720 \
    --test_dataset_path "${SCRATCH_DATA}/${DATASET}/STLT.Annotations.json" \
    --labels_path "${SCRATCH_DATA}/STLT.Schema.json" \
    --videos_path "${SCRATCH_DATA}/${FRAMES_DIR}" \
    --checkpoint_path "${HOME}/models/CACNF/Trained/${MODEL_PATH}.pth" \
    --resnet_model_path "${HOME}/models/CACNF/Base/r3d50_KMS_200ep.pth" \
    --output_path "${RESULT_PATH}/cacnf_${DATASET}" \
    --layout_samples "${LAYOUT_SAMPLES}" --layout_stride "${LAYOUT_STRIDE}" \
    --appearance_samples "${APP_SAMPLES}" --appearance_stride "${APP_STRIDE}" \
    --resize_height "${RESOLUTION}" --crop_scale "${RESIZE_CROP}" \
    --num_spatial_layers "${SPATIAL}" --num_temporal_layers "${TEMPORAL}" \
    --num_appearance_layers "${APPEARANCE}" --num_fusion_layers "${FUSION}" \
    --normaliser_mean 69.201 69.201 69.201 --normaliser_std 58.571 58.571 58.571 \
    --which_logits caf --batch_size "${BATCH_SIZE}" --num_workers 2
echo "   == Inference Done =="
echo ""

# ======================================================
# No Need to copy Data as writing directly to out at end.
# ======================================================
echo "   ++ ALL DONE! Hurray! ++"
conda deactivate