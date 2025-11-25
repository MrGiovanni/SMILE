export SD_MODEL_NAME="stable-diffusion-v1-5/stable-diffusion-v1-5"
export FT_VAE_NAME="../ckpt/autoencoder"
export TRAINED_UNET_NAME="../STEP2-DiffusionModel/diffusion"
export SEG_MODEL_NAME="../ckpt/segmenter/nnUNetTrainer__nnUNetResEncUNetLPlans__2d"
export CLS_MODEL_NAME="../ckpt/classifier/best_model_99_smile.pth" # monai backbone

# Temporary path with soft link
export TRAIN_DATA_DIR="/mnt/data/jliu452/Data/Dataset901_SMILE/h5" 
export TRAIN_DATA_SHAPE_EXCEL_DIR="./dataset901/step3-shapes.csv"

# output location
export OUTPUT_DIR="../logs/smile_logs-v0.3-no-seg"
export CHECKPOINT_DIR="/mnt/data/jliu452/SMILE/logs/smile_logs-v0.3-no-seg/checkpoint-40000" # the best model so far now
export INIT_STEP=15000

# resume trained model: --resume_from_checkpoint=$CHECKPOINT_DIR \ 
# --output_dir OUTPUT_DIR
#  --use_same_shape \
accelerate launch --mixed_precision="no" --num_processes=1 train_text_to_image.py \
  --sd_model_name_or_path=$SD_MODEL_NAME \
  --finetuned_vae_name_or_path=$FT_VAE_NAME \
  --pretrained_unet_name_or_path=$TRAINED_UNET_NAME \
  --seg_model_path=$SEG_MODEL_NAME \
  --cls_model_path=$CLS_MODEL_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --dataset_shape_file=$TRAIN_DATA_SHAPE_EXCEL_DIR \
  --init_global_step=$INIT_STEP \
  --output_dir=$OUTPUT_DIR \
  --resume_from_checkpoint=$CHECKPOINT_DIR \
  --vae_loss="l1" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=2 \
  --dataloader_num_workers=2 \
  --max_train_steps=200_000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --report_to=wandb \
  --validation_steps=1000 \
  --checkpointing_steps=10000 \
  --checkpoints_total_limit=3 \
  --warmup_end_df_only=2000 \
  --warmup_end_add_cls=10000 \
  --warmup_end_add_cls_seg_hu=15000 \
  --warmup_end_add_cycle=190000 \
  --uc_area_loss_weight=1e-3 \
  --cls_loss_weight=1e-3 \
  --seg_loss_weight=1e-2 \
  --hu_loss_weight=1e-2 \
  --cycle_loss_weight=1e-1 \
  --validation_images ./data/validation/baichaoxiao20240416_venous/ct.h5 ./data/validation/baichaoxiao20240416_non-contrast/ct.h5 ./data/validation/baichaoxiao20240416_non-contrast/ct.h5 ./data/validation/baichaoxiao20240416_non-contrast/ct.h5 \
  --validation_prompt "An non-contrast phase CT slice." "An arterial phase CT slice." "An venous phase CT slice." "An delayed phase CT slice." \

 