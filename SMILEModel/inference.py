from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDIMScheduler, DDPMScheduler
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.utils import is_torch_xla_available


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from PIL import Image
import PIL
import nibabel as nib 
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import albumentations as A
import argparse

from typing import Any, Callable, Dict, List, Optional, Union

from types import SimpleNamespace
import cv2
import safetensors
from dataset import CTDatasetInference, collate_fn_inference, CTDatasetInferenceH5
from testEnhanceCTPipeline import init_unet, ConcatInputStableDiffusionPipeline

import pandas as pd

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

class HWCarrayToCHWtensor(A.ImageOnlyTransform):
    """Converts (H, W, C) NumPy array to (C, H, W) PyTorch tensor."""
    def apply(self, img, **kwargs):
        return torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) → (C, H, W)

resolution = 512 # NOTE: DO NOT CHANGE



"""
Camera-ready version of SMILE inference

Expected input data form: .nii.gz


@author: jliu452
email: jliu452@jh.edu

"""



def run_single_slice_inference(
    pipe,
    ct_path: str,
    target_phase: str,
    output_dir: str,
    slice_location: float = 0.5,
    resolution: int = 512,
    num_inference_steps: int = 500,
    hu_clip: tuple = (-200, 200),
    seed: int = 42,
):
    """

    Run single-slice SMILE inference directly on a .nii.gz CT volume.
    Light-weight design, only 10 GB of VARM required.

    
    Args:
        pipe: A loaded ConcatInputStableDiffusionPipeline
        ct_path (str): Path to input CT volume (.nii.gz)
        target_phase (str): Target contrast phase (e.g., 'arterial', 'venous', 'delayed')
        output_dir (str): Directory to save the results
        slice_location (float): Slice position ratio [0, 1] along z-axis (default=0.5)
        resolution (int): Resize resolution for model input (default=512)
        num_inference_steps (int): Diffusion sampling steps
        hu_clip (tuple): HU range for visualization and clipping (default=(-200, 200))
        seed (int): Random seed for deterministic output
    Returns:
        enhanced_slice (np.ndarray): Translated CT slice in HU space
    """
    
    ct_volume = nib.load(ct_path)
    nii_shape = ct_volume.shape  # (H, W, D)
    z_shape = nii_shape[2]
    slice_idx = int(z_shape * slice_location)

    print(f"\n Running SMILE on single-slice inference ... \n[outdated warning ⚠️: this function will be deprecated in future releases, please use the full-volume inference instead] ")
    def load_CT_slice_from_nfiti(ct_data, slice_idx):
        
        ct_slice = ct_data.dataobj[:, :, slice_idx:slice_idx + 3].copy()
        ct_slice = np.clip(ct_slice, -1000, 1000)

        # visualization
        original_slice = ct_slice[:,:,0] # HU range
        original_slice = np.clip(original_slice, hu_clip[0], hu_clip[1])
        original_slice = np.rot90(original_slice)

        original_phase = args.patient_id.split("_")[-1]
        out_path = os.path.join(output_dir, f"slice__{slice_idx:03d}_from_{original_phase}.npz")
        np.savez(out_path, original_slice.astype(np.float32))

        norm_vis = ((original_slice - hu_clip[0]) / (hu_clip[1] - hu_clip[0]) * 255).astype(np.uint8)
        png_path = os.path.join(output_dir, f"slice_{slice_idx:03d}_from_{original_phase}.png")
        Image.fromarray(norm_vis).save(png_path)


        ct_slice = (ct_slice + 1000.) / 2000.0  # → [0,1]
        return ct_slice  # (H, W, 3)

    cond_ct_slice = load_CT_slice_from_nfiti(ct_volume, slice_idx)
    cond_ct_slice = cv2.resize(cond_ct_slice, (resolution, resolution), interpolation=cv2.INTER_CUBIC)

    # Normalize to model range
    norm = A.Normalize(mean=(0.5, 0.5, 0.5),
                       std=(0.5, 0.5, 0.5),
                       max_pixel_value=1.0)
    cond_ct_slice = norm(image=cond_ct_slice)["image"]
    cond_ct_slice = torch.from_numpy(cond_ct_slice).permute(2, 0, 1).unsqueeze(0).half().to("cuda")

    prompt = [f"An {target_phase} phase CT slice."]
    with torch.no_grad():
        cond_latents = pipe.vae.encode(cond_ct_slice).latent_dist.sample() * pipe.vae.config.scaling_factor
        latents = torch.randn_like(cond_latents)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        result = pipe(
            prompt=prompt,
            latents=latents,
            cond_latents=cond_latents,
            num_inference_steps=num_inference_steps,
            output_type="np",
            generator=generator
        ).images[0]  # (H, W, 3), range [0,1]

    enhanced_slice = (result * 2 - 1) * 1000  # → HU
    enhanced_slice = enhanced_slice[:,:,0]

    enhanced_slice = np.clip(enhanced_slice, hu_clip[0], hu_clip[1])
    enhanced_slice = np.rot90(enhanced_slice)

    # save as .npz form
    out_path = os.path.join(output_dir, f"slice__{slice_idx:03d}_{target_phase}.npz")
    np.savez(out_path, enhanced_slice.astype(np.float32))

    norm_vis = ((enhanced_slice - hu_clip[0]) / (hu_clip[1] - hu_clip[0]) * 255).astype(np.uint8)
    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"slice_{slice_idx:03d}_{target_phase}.png")
    Image.fromarray(norm_vis).save(png_path)
    print(f" Saved visualization to: {png_path}")

    return enhanced_slice





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process input and output paths along with a BDMAP ID.")
    parser.add_argument("--patient_id", type=str, required=True, help="example: 3A49sEObFLc_noncontrast")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the ct to be translated.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--target_phase", type=str, required=True, help="Tanslation target phase. (non_contrast, arterial, venous, delayed)")
    parser.add_argument("--chunk_size", type=int, required=False, default=32, help="Trunk Size")
    parser.add_argument("--finetuned_vae_name_or_path", type=str, required=True, help="Path to the trained VAE model.")
    parser.add_argument("--finetuned_unet_name_or_path", type=str, required=True, help="Path to the trained UNet model.")
    parser.add_argument("--sd_model_name_or_path", type=str, required=True, help="Path to the trained stable diffusion model.")
    parser.add_argument("--slice_location", type=float, default=None, help="Slice mode. If set, only inference on the slice location: Z_shape * slice_location")
    parser.add_argument("--bdmap_id", type=str, default=None, help="Inference for BDMAP format dataset.")
    parser.add_argument("--guide_CSV", type=str, default=None, help="Path to the guidance CSV file.")
    args = parser.parse_args()

    """Check by guidence CSV"""
    if args.guide_CSV is not None:
        check_df = pd.read_csv(args.guide_CSV)
        if args.patient_id not in check_df["Inference ID"].values:# same as ePAI pipeline
            print(f" [CSV checker]: Patient id {args.patient_id} not in guidance CSV, skipping...")
            exit(0)
    
    """StableDiffusionPipeline"""
    # Setting up models in the pipeline.
    finetuned_vae_name_or_path = args.finetuned_vae_name_or_path#"./VAE"
    finetuned_unet_name_or_path = args.finetuned_unet_name_or_path#"./UNet"
    # load vae
    vae = AutoencoderKL.from_pretrained(
            finetuned_vae_name_or_path, subfolder="vae", #revision=args.revision, variant=args.variant,
            torch_dtype=torch.float16
        )
    # load unet (network required)
    unet_args = SimpleNamespace(pretrained_model_name_or_path=args.sd_model_name_or_path)
    unet = init_unet(unet_args.pretrained_model_name_or_path, zero_cond_conv_in=True)
    unet_ckpt = safetensors.torch.load_file(os.path.join(finetuned_unet_name_or_path, "unet", "diffusion_pytorch_model.safetensors"))
    unet.load_state_dict(unet_ckpt, strict=True)
    unet = unet.half()


    # construct pipeline
    pipe = ConcatInputStableDiffusionPipeline.from_pretrained(
        args.sd_model_name_or_path, 
        unet=unet,
        vae=vae,
        safety_checker=None,
        torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)  
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")
    

    # Dataset settings
    data_dir = args.input_path
    save_dir = args.output_path
    os.makedirs(save_dir, exist_ok=True)

    inference_transforms = A.Compose([      # resize 
        A.Resize(resolution, resolution, interpolation=cv2.INTER_CUBIC), # model requires 512
    ])


    if args.slice_location is not None:
        # Compute slice index (e.g. 0.5 means middle slice)
        enhanced_slice = run_single_slice_inference(
            pipe=pipe,
            ct_path=args.input_path,
            target_phase=args.target_phase,
            output_dir=args.output_path,
            slice_location=args.slice_location,
        )
        exit(0)

       
    if args.bdmap_id:# form: Dataset/BDMAP_000xxxxx/ct.nii.gz
        data_dir = os.path.join(data_dir, args.bdmap_id, "ct.nii.gz")

    chunk_size = args.chunk_size  # NOTE: how many 3-channel images to input at once
    # wrap the CT as a dataset for batch_size > 1
    ct_dataset = CTDatasetInference(
        file_path=data_dir,
        image_transforms=inference_transforms,
        cond_transforms=inference_transforms)

    ct_dataloader = torch.utils.data.DataLoader(
        ct_dataset,
        shuffle=False,  
        collate_fn=collate_fn_inference,    # prompt rather than token
        batch_size=chunk_size,
        num_workers=chunk_size if chunk_size <= 16 else 16, # maximum 16 workers
        drop_last=False
    )

    nii_shape = list(ct_dataset.ct_xyz_shape)
    base_name = os.path.basename(args.input_path).replace(".nii.gz", "")
    original_phase = base_name.split("_")[-1]  
    patient_id = base_name.split("_")[0] 

    if "ct.nii.gz" in base_name:
        # BDMAP format, fall back
        base_name = args.input_path
        original_phase = "BDMAP Format (phase info not given)"
        patient_id = args.input_path.split("/")[-1:]


    MODEL_NAME = os.path.basename(args.finetuned_unet_name_or_path)
    print("Now running SMILE inference mode,\n CT volume shape info: ", nii_shape, f"\n patient id: {patient_id}", f"\n Task: {original_phase} --> {args.target_phase}", f"\n SMILE Version: {MODEL_NAME}")
    z_shape = nii_shape[2]

    # loading and weights
    enhanced_ct = np.zeros(nii_shape, dtype=np.float32)   # H, W, D
    weights_ct  = np.zeros(nii_shape, dtype=np.float32)

    # noise, from the same 3d domain
    H, W, D = nii_shape
    downsample_factor = int(8 * 512/resolution)
    base_noise = torch.randn((1, 4, H//downsample_factor, W//downsample_factor, D), device="cuda", dtype=torch.float16)  # latent space size, [1, 4, H_d, W_d, D]
    
    for batch in tqdm(ct_dataloader):
        cond_image = batch["cond_pixel_values"].to("cuda").half()   # same thing as `pixel_values`, [3, H, W]
        prompt = [f"An {args.target_phase} phase CT slice."] * len(cond_image)  # NOTE: hard-coded prompt, can be customized
        slice_idx = batch["slice_idx"]
        with torch.no_grad():
            cond_latents = pipe.vae.encode(cond_image).latent_dist.sample() * pipe.vae.config.scaling_factor
            slice_idx_tensor = torch.tensor(slice_idx, device="cuda", dtype=torch.float32)
            z_idx = slice_idx_tensor / D * base_noise.shape[-1]
            z_idx = z_idx.clamp(0, base_noise.shape[-1]-1).long()
            latents = base_noise.squeeze(0).permute(3, 0, 1, 2)[z_idx].contiguous()
            if cond_latents.shape != latents.shape:# add saftycheck, interpolate latent noise if shape checker is triggered
                latent_h, latent_w = cond_latents.shape[-2:]
                latents = F.interpolate(latents, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
            
        generator = torch.Generator(device="cuda").manual_seed(42)
        images = pipe(
            num_inference_steps=200, 
            prompt=prompt,
            latents=latents,  
            cond_latents=cond_latents,
            output_type="np",
            generator=generator
        ).images

        real_chunk_size = len(images)
        for idx in range(real_chunk_size):
            slice_id = slice_idx[idx]
            
            if slice_id + 3 > D:
                continue
            enhanced_slice = cv2.resize(
                images[idx], 
                nii_shape[:2][::-1],  # resize back to H,W
                cv2.INTER_CUBIC
            )  # (H, W, 3)

            # sum to the corres slice
            enhanced_ct[:, :, slice_id:slice_id+3] += enhanced_slice
            weights_ct[:, :, slice_id:slice_id+3]  += 1

    # average back
    weights_ct[weights_ct == 0] = 1
    enhanced_ct = enhanced_ct / weights_ct
    enhanced_ct = (enhanced_ct * 2 - 1) * 1000
    enhanced_ct = enhanced_ct.astype(np.int16)

    # visualization
    num_slices = enhanced_ct.shape[2]
    indices = np.linspace(0, num_slices - 1, 8, dtype=int)

    for i, idx in enumerate(indices):
        channel = enhanced_ct[:, :, idx]

        # --- HU clip to [-200, 200] ---
        hu_min, hu_max = -200, 200
        channel = np.clip(channel, hu_min, hu_max)
        norm = ((channel - hu_min) / (hu_max - hu_min) * 255).astype(np.uint8)
        norm = np.rot90(norm)
        Image.fromarray(norm).save(
            os.path.join(save_dir, f"{original_phase}_to_{args.target_phase}_{i:02d}.png")
        )

    os.makedirs(os.path.join(save_dir), exist_ok=True)

    original_nfiti = nib.load(args.input_path)   
    out_nii = nib.Nifti1Image(enhanced_ct, original_nfiti.affine, original_nfiti.header)
    out_nii.header.set_data_dtype(np.int16)
    out_nii.to_filename(os.path.join(save_dir, f"ct.nii.gz")) 


