#!/bin/bash
#  Author: Michael Camilleri
#
#  Scope:
#     Trains the STLT Model on own data
#
#  Script takes the following parameters: note that the batch size is defined by the product of
#    Cores and Images.
#     [Batch]    - Batch-Size
#     [Rate]     - Learning Rate
#     [Epochs]   - Maximum Number of Epochs to train for
#     [Offset]   - Offset from base data location to retrieve the data splits
#
#  USAGE:
#     srun --time=23:00:00 --gres=gpu:1 --nodelist=charles18 bash/train_stlt.sh 64 0.00005 60 Fixed &> ~/logs/train_stlt.0001.Fixed.out
#     * N.B.: The above should be run from the root STLT directory.

#  Data Structures
#    Data is expected to be under ${HOME}/data/behaviour/ which follows the definitions laid out
#        in my Jupyter notebook.

# Do some Calculations/Preprocessing
OUT_NAME=${3}_${1}_${2}

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
      --info=progress2 ${HOME}/data/behaviour/Train/$4/ ${SCRATCH_DATA}
echo "   ----- DONE -----"
mail -s "Train_STLT on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Synchronised Data and Models."
echo ""

# ===========
# Train Model
# ===========
echo " ===================================="
echo " Training Model (BS=${1}, LR=${2}) on ${4} for ${3} epochs"
SCRATCH_MODELS=${SCRATCH_HOME}/models/stlt
mkdir -p ${SCRATCH_MODELS}

python src/train_stlt.py  \
  --labels_path "${SCRATCH_DATA}/STLT.Schema.json" \
  --videoid2size_path "${SCRATCH_DATA}/STLT.Sizes.json"  \
  --train_dataset_path "${SCRATCH_DATA}/Train/STLT.Annotations.json"  \
  --val_dataset_path "${SCRATCH_DATA}/Validate/STLT.Annotations.json" \
  --save_model_path "${SCRATCH_MODELS}/${OUT_NAME}.pth" \
  --layout_num_frames 25 --num_spatial_layers 4 --num_temporal_layers 8 \
  --batch_size ${1} --learning_rate ${2} --weight_decay 1e-3 --clip_val 5.0 \
  --epochs ${3} --warmup_epochs 2 --num_workers 2
echo "   == Training Done =="
mail -s "Train_STLT on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Model Training Completed."
echo ""

# ===========
# Copy Data
# ===========
echo " ===================================="
echo " Copying Model Weights (as ${OUT_NAME})"
mkdir -p "${HOME}/models/STLT/Trained/${OUT_NAME}"
rsync --archive --compress --info=progress2 "${SCRATCH_MODELS}/" "${HOME}/models/STLT/Trained/"
echo "   ++ ALL DONE! Hurray! ++"
mail -s "Train_STLT on ${SLURM_JOB_NODELIST}:${OUT_NAME}" ${USER}@sms.ed.ac.uk <<< "Output Models copied as '${HOME}/models/STLT/Trained/${OUT_NAME}.pth'."
conda deactivate