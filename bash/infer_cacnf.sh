#!/bin/bash
#  Author: Michael Camilleri
#
#  Scope:
#     Infers behaviours according to the CACNF Model
#
#  Script takes the following parameters:
#   -- Architecture Parameters --
#     [Spatial]  - Number of Spatial Layers
#     [Temporal] - Number of Temporal Layers
#     [Appearance] - Number of Appearance Layers (on top of ResNet)
#     [Fusion] - Number of Fusion Layers
#     [Resolution] - Image Height Size
#     [Batch_size] - Due to a weird architectural strategy, it seems that the batch sizes must
#                    match

#   -- Paths/Setup --
#     [Model]    - Model Path to load checkpoint from: note it should NOT contain the .pth extension
#     [DataSet]  - Which Dataset to infer for
#     [Offset]   - Offset from base data location to retrieve the data splits
#     [Frames]   - Y/N: Indicates if Frames should be rsynced: this is done to save time if it is
#                       known that the machine contains the right data splits.
#
#  USAGE:
#     srun --time=1-23:00:00 --gres=gpu:1 --partition=apollo --nodelist=apollo2 bash/infer_cacnf.sh 4 8 4 4 128 1.0 16 "Fixed/A[4-8-4-4]_I[128]_L[16_0.000005_2_1]_CAF" Validate Fixed N &> ~/logs/infer_1.log
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

MODEL_PATH=${8}
DATASET=${9}
PATH_OFFSET=${10}
FORCE_FRAMES=${11,,}

# Derivative Values
if [ "${DATASET,,}" = "test" ]; then
  PARENT_DIR='Test'
else
  PARENT_DIR='Train'
fi

# Path Values
SCRATCH_HOME=/disk/scratch/${USER}
SCRATCH_DATA=${SCRATCH_HOME}/data/behaviour
SCRATCH_MODELS=${SCRATCH_HOME}/models/infer_cacnf
RESULT_PATH="${HOME}/results/CACNF/${MODEL_PATH}"

# ===================
# Environment setup
# ===================
echo "Setting up Conda enviroment on ${SLURM_JOB_NODELIST}"
set -e # Make script bail out after first error
source activate py3stlt   # Activate Conda Environment
echo "Libraries from: ${LD_LIBRARY_PATH}"

# Setup NCCL Debug Status
export NCCL_DEBUG=INFO

# Make own folder on the node's scratch disk
mkdir -p ${SCRATCH_HOME}
echo ""

# ================================
# Download Data and Models if necessary
# ================================
echo " ===================================="
echo "Consolidating Data/Models in ${SCRATCH_HOME}"
mkdir -p ${SCRATCH_DATA}
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
mkdir -p ${SCRATCH_MODELS}
echo "  -> Synchronising Models"
rsync --archive --update --compress ${HOME}/models/CACNF/Base/r3d50_KMS_200ep.pth ${SCRATCH_MODELS}/resnet.base.pth
rsync --archive --update --compress ${HOME}/models/CACNF/Trained/${MODEL_PATH}.pth ${SCRATCH_MODELS}/inference.trained.pth
echo "   ----- DONE -----"
echo ""

# ===========
# Infer Model
# ===========
echo " ===================================="
echo " Inferring Behaviours for ${DATASET} using model ${MODEL_PATH}"
mkdir -p ${RESULT_PATH}

python src/inference.py \
    --dataset_name something --dataset_type multimodal --model_name cacnf --videos_as_frames \
    --test_dataset_path "${SCRATCH_DATA}/${DATASET}/STLT.Annotations.json" \
    --labels_path "${SCRATCH_DATA}/STLT.Schema.json" \
    --videoid2size_path "${SCRATCH_DATA}/STLT.Sizes.json" \
    --videos_path "${SCRATCH_DATA}/Frames" \
    --checkpoint_path "${SCRATCH_MODELS}/inference.trained.pth" \
    --resnet_model_path "${SCRATCH_MODELS}/resnet.base.pth" \
    --output_path "${RESULT_PATH}/cacnf_${DATASET}" \
    --layout_num_frames 25 --appearance_num_frames 25 --resize_height ${RESOLUTION} \
    --num_spatial_layers ${SPATIAL} --num_temporal_layers ${TEMPORAL} \
    --num_appearance_layers ${APPEARANCE} --num_fusion_layers ${FUSION} \
    --normaliser_mean 69.201 69.201 69.201 --normaliser_std 58.571 58.571 58.571 \
    --which_logits caf --batch_size ${BATCH_SIZE} --num_workers 2
echo "   == Inference Done =="
echo ""

# ======================================================
# No Need to copy Data as writing directly to out at end.
# ======================================================
echo "   ++ ALL DONE! Hurray! ++"
conda deactivate