#!/bin/bash
# ============================================================
# SMILE Multi-phase CT Translation Inference Script
# Author: Marcus / CVPR-SMILE
# ============================================================

set -e

# ======== Basic paths ======== #
export MODELS_FOLDER="../ckpt"
export SD_MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export FT_VAE_NAME="../ckpt/autoencoder/vae"

# ======== User-defined arguments ======== #
MODEL_CKPT_NAME="SMILE_v0.2"
INPUT_DIR="/mnt/data/jliu452/Data/Dataset906_MPT/image"
OUTPUT_BASE="../out/translation/model-${MODEL_CKPT_NAME}"

echo "🛠️  Using model checkpoint: ${MODEL_CKPT_NAME}"
# ======== Parse CLI arguments ======== #
while [[ $# -gt 0 ]]; do
  case "$1" in
    --origin)
      ORIGIN="$2"
      shift 2
      ;;
    --targets)
      shift
      TARGETS=("$@")
      break
      ;;
    *)
      echo "❌ Unknown argument: $1"
      echo "Usage: bash inference.sh --origin <phase> --targets <target1 [target2 ...]>"
      exit 1
      ;;
  esac
done

echo "📘 Origin phase: $ORIGIN"
echo "🎯 Target phases: ${TARGETS[*]}"

# ======== Paths ======== #
TRAINED_UNET_NAME="${MODELS_FOLDER}/${MODEL_CKPT_NAME}"

# ======== Create output dir ======== #
mkdir -p "$OUTPUT_BASE"

# ======== Loop through input cases ======== #
for INPUT_PATH in "${INPUT_DIR}"/*"${ORIGIN}".nii.gz; do
    if [[ ! -f "$INPUT_PATH" ]]; then
        continue
    fi
    BID=$(basename "$INPUT_PATH" | sed "s/_${ORIGIN}.nii.gz//")
    echo "===== Running inference for ${BID} (origin=${ORIGIN}) ====="

    for TARGET in "${TARGETS[@]}"; do
        OUTPUT_PATH="${OUTPUT_BASE}/${BID}_${ORIGIN}_to_${TARGET}/"
        mkdir -p "$OUTPUT_PATH"

        echo "➡️  Translating ${BID}: ${ORIGIN} → ${TARGET}"
        CUDA_VISIBLE_DEVICES=0 python -W ignore inference.py \
          --patient_id "$BID" \
          --input_path "$INPUT_PATH" \
          --output_path "$OUTPUT_PATH" \
          --target_phase "$TARGET" \
          --chunk_size=64 \
          --finetuned_vae_name_or_path "$FT_VAE_NAME" \
          --finetuned_unet_name_or_path "$TRAINED_UNET_NAME" \
          --sd_model_name_or_path "$SD_MODEL_NAME"
    done

    echo "✅ Finished ${BID}"
done

echo "🎉 All ${ORIGIN} → ${TARGETS[*]} translations completed."