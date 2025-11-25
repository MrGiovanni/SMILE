#!/bin/bash
# ============================================================================
# SMILE Inference Script (Pass multi-target directly to Python)
#
# Example:
#   bash inference.sh --gpu_id 0 --source non-contrast \
#                     --target arterial,venous,delayed \
#                     --patient_id RS-GIST-121
# ============================================================================


# ------------------------- Parse Arguments -----------------------------------
GPU_ID=""
SOURCE=""
TARGETS=""        # <-- keep full raw string
PATIENT_ID=""
MODEL_NAME="SMILE_v0.2.2-a100-200k"   # default model

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --gpu_id)     GPU_ID="$2"; shift ;;
        --source)     SOURCE="$2"; shift ;;
        --target)     TARGETS="$2"; shift ;;   # pass directly
        --patient_id) PATIENT_ID="$2"; shift ;;
        --model)      MODEL_NAME="$2"; shift ;;
        *) echo "[ERROR] Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done


# --------------------------- Sanity Checks -----------------------------------
if [ -z "$GPU_ID" ]; then
    echo "[WARN] No GPU specified. Defaulting to GPU 0."
    GPU_ID="0"
fi

if [ -z "$SOURCE" ] || [ -z "$TARGETS" ]; then
    echo "[ERROR] Missing required args: --source and/or --target"
    exit 1
fi

if [ -z "$PATIENT_ID" ]; then
    echo "[ERROR] Missing --patient_id"
    exit 1
fi


echo "[INFO] GPU(s):        $GPU_ID"
echo "[INFO] Model:         $MODEL_NAME"
echo "[INFO] Patient:       $PATIENT_ID"
echo "[INFO] Source phase:  $SOURCE"
echo "[INFO] Target phases: $TARGETS"   


# ------------------------------- Fixed Paths ---------------------------------

export MODELS_FOLDER="/projects/bodymaps/jliu452/ckpt"
export SD_MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export FT_VAE_NAME="../STEP1-AutoEncoderModel/klvae/autoencoder_no_kl"

DATASET="Dataset901_SMILE/PT_data"

INPUT_CASE_NAME="${PATIENT_ID}_${SOURCE}"
CT_PATH="/projects/bodymaps/jliu452/Data/$DATASET/${INPUT_CASE_NAME}.nii.gz"

TRAINED_UNET_PATH="$MODELS_FOLDER/$MODEL_NAME"
OUTPUT_ROOT="/projects/bodymaps/jliu452/TRANS/translation/model-$MODEL_NAME"

echo "[INFO] Input CT:      $CT_PATH"


# ------------------------------ Run Python -----------------------------------
CUDA_VISIBLE_DEVICES=$GPU_ID python -W ignore inference.py \
    --patient_id "$PATIENT_ID" \
    --input_path "$CT_PATH" \
    --output_path "$OUTPUT_ROOT" \
    --target_phase "$TARGETS" \
    --chunk_size 128 \
    --finetuned_vae_name_or_path "$FT_VAE_NAME" \
    --finetuned_unet_name_or_path "$TRAINED_UNET_PATH" \
    --sd_model_name_or_path "$SD_MODEL_NAME"
