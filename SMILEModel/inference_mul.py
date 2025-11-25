"""
SMILE Multi-volume Inference (batch version)

Automatically runs inference on all subfolders in dataset_path.
If --guide_CSV is provided, only process those listed IDs.

Expected structure:
(1) 
 dataset_path/
   ├── BDMAP_00012345/
   │     └── ct.nii.gz
   ├── BDMAP_00067890/
   │     └── ct.nii.gz

(2) 
 dataset_path/
   ├── patient_id.nii.gz
   ├── ...

Author: jliu452
Email: jliu452@jh.edu
"""

from diffusers import AutoencoderKL, DDIMScheduler
from diffusers.utils import is_torch_xla_available
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import nibabel as nib
import os, cv2, safetensors, argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
from types import SimpleNamespace

from dataset import CTDatasetInference, collate_fn_inference
from testEnhanceCTPipeline import init_unet, ConcatInputStableDiffusionPipeline

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


# ------------------------- MAIN -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SMILE Multi-volume Inference on dataset folder")
    parser.add_argument("--input_path", type=str, required=True, help="Dataset folder containing multiple BDMAP_xxxxx cases")
    parser.add_argument("--output_path", type=str, required=True, help="Folder to save outputs")
    parser.add_argument("--target_phase", type=str, required=True, nargs="+", help="Target contrast phase lists (arterial / venous / delayed / noncontrast)")
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--finetuned_vae_name_or_path", type=str, required=True)
    parser.add_argument("--finetuned_unet_name_or_path", type=str, required=True)
    parser.add_argument("--sd_model_name_or_path", type=str, required=True)
    parser.add_argument("--guide_CSV", type=str, default=None, help="Optional CSV to specify which IDs to process")
    args = parser.parse_args()

    print("\n============================= Inference Configuration =============================")
    for k, v in vars(args).items():
        print(f"{k:30s}: {v}")
    print("=====================================================================================\n\n")


    dataset_dir = args.input_path
    output_root = args.output_path
    os.makedirs(output_root, exist_ok=True)
    # ---------- Determine which cases to process ----------
    all_cases = sorted(os.listdir(dataset_dir))


    if args.guide_CSV:
        guide_df = pd.read_csv(args.guide_CSV)
        guided_ids = guide_df["Inference ID"].astype(str).tolist()
        case_list = [cid for cid in all_cases if cid in guided_ids]
        print(f"[CSV mode] Hej, I will only inference on the {len(case_list)} guided cases found in guidence CSV.")
    else:
        case_list = all_cases
        print(f"[Auto mode] Hej, found {len(case_list)} cases under {dataset_dir}, will inference on all of them.")

    if len(case_list) == 0:
        print("[Error] Hej, ⚠️ No valid cases found, exiting. See you then")
        exit(0)

    # ---------- Load models ----------
    print(f"\n[SMILE Info] Loading models, version {os.path.basename(args.finetuned_unet_name_or_path)} ...")

    vae = AutoencoderKL.from_pretrained(args.finetuned_vae_name_or_path, subfolder="vae", torch_dtype=torch.float16)
    unet_args = SimpleNamespace(pretrained_model_name_or_path=args.sd_model_name_or_path)
    unet = init_unet(unet_args.pretrained_model_name_or_path, zero_cond_conv_in=True)
    unet_ckpt = safetensors.torch.load_file(
        os.path.join(args.finetuned_unet_name_or_path, "unet", "diffusion_pytorch_model.safetensors")
    )
    unet.load_state_dict(unet_ckpt, strict=True)
    unet = unet.half()

    pipe = ConcatInputStableDiffusionPipeline.from_pretrained(
        args.sd_model_name_or_path,
        unet=unet,
        vae=vae,
        safety_checker=None,
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")

    # ---------- Inference loop for each case ----------
    resolution = 512
    inference_transforms = A.Compose([A.Resize(resolution, resolution, interpolation=cv2.INTER_CUBIC)])
    downsample_factor = int(8 * 512 / resolution)

    for case_id in case_list:
        input_ct_path = os.path.join(dataset_dir, case_id, "ct.nii.gz")
        
        if not os.path.exists(input_ct_path):
            try:
                input_ct_path = os.path.join(dataset_dir, f"{case_id}.nii.gz")
            except:
                print(f"⚠️ Skipping {case_id}: ct.nii.gz not found.")
                continue

        original_phase = case_id.split("_")[-1]
        if original_phase not in ["noncontrast", "arterial", "venous", "delayed"]:
            original_phase = ""
        
        

        for phase in args.target_phase:
            save_dir = os.path.join(output_root, f"{case_id}_to_{phase}")
            os.makedirs(save_dir, exist_ok=True)

            print(f"\n[SMILE Info] Working: {case_id} → target phase: {phase}, prompt: 'An {phase} phase CT slice.'")
            ct_dataset = CTDatasetInference(
                file_path=input_ct_path,
                image_transforms=inference_transforms,
                cond_transforms=inference_transforms,
            )
            ct_dataloader = DataLoader(
                ct_dataset,
                shuffle=False,
                collate_fn=collate_fn_inference,
                batch_size=args.chunk_size,
                num_workers=min(args.chunk_size, 16),
                drop_last=False,
            )

            nii_shape = list(ct_dataset.ct_xyz_shape)
            enhanced_ct = np.zeros(nii_shape, dtype=np.float32)
            weights_ct = np.zeros(nii_shape, dtype=np.float32)

            H, W, D = nii_shape
            base_noise = torch.randn(
                (1, 4, H // downsample_factor, W // downsample_factor, D),
                device="cuda",
                dtype=torch.float16,
            )

            for b_idx, batch in enumerate(tqdm(ct_dataloader, desc=f"Inference on {case_id}")):
                cond_image = batch["cond_pixel_values"].to("cuda").half()
                prompt = [f"An {phase} phase CT slice."] * len(cond_image)
                slice_idx = batch["slice_idx"]

                with torch.no_grad():
                    cond_latents = pipe.vae.encode(cond_image).latent_dist.sample() * pipe.vae.config.scaling_factor
                    slice_idx_tensor = torch.tensor(slice_idx, device="cuda", dtype=torch.float32)
                    z_idx = (slice_idx_tensor / D * base_noise.shape[-1]).clamp(0, base_noise.shape[-1] - 1).long()
                    latents = base_noise.squeeze(0).permute(3, 0, 1, 2)[z_idx].contiguous()

                    if cond_latents.shape != latents.shape:
                        latent_h, latent_w = cond_latents.shape[-2:]
                        latents = F.interpolate(latents, size=(latent_h, latent_w), mode="bilinear", align_corners=False)

                generator = torch.Generator(device="cuda").manual_seed(42)
                images = pipe(
                    num_inference_steps=200,
                    prompt=prompt,
                    latents=latents,
                    cond_latents=cond_latents,
                    output_type="np",
                    generator=generator,
                ).images

                for idx, img in enumerate(images):
                    slice_id = slice_idx[idx]
                    if slice_id + 3 > D:
                        continue
                    enhanced_slice = cv2.resize(img, nii_shape[:2][::-1], cv2.INTER_CUBIC)
                    enhanced_ct[:, :, slice_id : slice_id + 3] += enhanced_slice
                    weights_ct[:, :, slice_id : slice_id + 3] += 1

            weights_ct[weights_ct == 0] = 1
            enhanced_ct = enhanced_ct / weights_ct
            enhanced_ct = (enhanced_ct * 2 - 1) * 1000
            enhanced_ct = enhanced_ct.astype(np.int16)

            # save visualization
            num_slices = enhanced_ct.shape[2]
            indices = np.linspace(0, num_slices - 1, 8, dtype=int)
            for i, idx in enumerate(indices):
                channel = np.clip(enhanced_ct[:, :, idx], -200, 200)
                norm = ((channel + 200) / 400 * 255).astype(np.uint8)
                norm = np.rot90(norm)
                Image.fromarray(norm).save(os.path.join(save_dir, f"{original_phase}_to_{phase}_{i:02d}.png"))

            original_nii = nib.load(input_ct_path)
            out_nii = nib.Nifti1Image(enhanced_ct, original_nii.affine, original_nii.header)
            out_nii.header.set_data_dtype(np.int16)
            out_nii.to_filename(os.path.join(save_dir, "ct.nii.gz"))

    print("\n🎉 All inference completed successfully.")