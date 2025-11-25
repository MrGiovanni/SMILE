#!/bin/bash
# Inference script for multiple input, the data are structured as BDMAP AbdomenAtlasPro form
# 7 Nov update: 
#   - modified to support multiple cases in one script. replaced inference.py with inference_mul.py
# ⚠️ caution: chunk size 32 for A5000, 64 for A6000. 128 + for A100 and Higher Level GPUs

set -e

export MODELS_FOLDER="../ckpt" # DO NOT CHANGE
export SD_MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"


# ================ User Defined ================ #
MODEL_CKPT_NAME="SMILE/Version (default as v0.2)"
DATA_FOLDER="/path/to/nifti/dataset/parent/folder"
DATASET_NAME="/dataset/name/as/nnunetv2/from"

# Translation targets
TARGETS=("arterial" "venous" "delayed")
GUIDE_CSV="/nifti/you/want/to/inference"
# ============================================== #


INPUT_DIR="${DATA_FOLDER}/${DATASET_NAME}"
OUTPUT_FOLDER="../out/${DATASET_NAME}-${MODEL_CKPT_NAME}"

# Path to trained UNet
TRAINED_UNET_NAME="${MODELS_FOLDER}/${MODEL_CKPT_NAME}"
FT_VAE_NAME="${MODELS_FOLDER}/autoencoder/vae" # VAE


CUDA_VISIBLE_DEVICES=0 python -W ignore inference_mul.py \
          --input_path "$INPUT_DIR" \
          --output_path "$OUTPUT_FOLDER" \
          --target_phase "${TARGETS[@]}" \
          --chunk_size=128 \
          --finetuned_vae_name_or_path "$FT_VAE_NAME" \
          --finetuned_unet_name_or_path "$TRAINED_UNET_NAME" \
          --sd_model_name_or_path "$SD_MODEL_NAME" \
          --guide_CSV "$GUIDE_CSV"

