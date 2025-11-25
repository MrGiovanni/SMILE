from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, AutoencoderKL, UNet2DConditionModel

from diffusers.utils.torch_utils import randn_tensor
from diffusers import DDIMScheduler, DDPMScheduler
import torch
import torch.nn as nn
from PIL import Image
import PIL
import nibabel as nib 
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import albumentations as A
import argparse
from torch.utils.data import Dataset

from typing import Any, Callable, Dict, List, Optional, Union
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import retrieve_timesteps, StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents
from diffusers.utils import is_torch_xla_available
from types import SimpleNamespace
import cv2
import safetensors
from dataset import CTDatasetInference, collate_fn_inference, CTDatasetInferenceH5

import pandas as pd

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

def load_CT_slice_from_nii(ct_data, slice_idx=None):
    """For any nii data during inference: ranging from [-1000, 1000], shape of (H W D) """

    ct_slice = ct_data.dataobj[:, :, slice_idx:slice_idx + 3].copy() 
    # target range: [-1000, 1000] -> [-1, 1]
    ct_slice[ct_slice > 1000.] = 1000.    # clipping range and normalize
    ct_slice[ct_slice < -1000.] = -1000.
    ct_slice = (ct_slice + 1000.) / 2000.       # [-1000, 1000] --> [0, 1]
    return ct_slice  # (H W 3)[0, 1]

class HWCarrayToCHWtensor(A.ImageOnlyTransform):
    """Converts (H, W, C) NumPy array to (C, H, W) PyTorch tensor."""
    def apply(self, img, **kwargs):
        return torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) → (C, H, W)



def init_unet(pretrained_model_name_or_path, zero_cond_conv_in=False):
    """
    Load the pretrained UNet model from the given path.
    
    """
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet", 
    )
    
    # double conv_in channel size, 
    # half with pretrained weight for input, 
    # half with zeros for cond
    if zero_cond_conv_in:   
        # 获取原始输入卷积层
        original_conv = unet.conv_in
        original_in_channels = original_conv.in_channels

        # create the new conv layer, with in-channel doubled
        new_conv = nn.Conv2d(
            in_channels=original_in_channels * 2,  
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            dilation=original_conv.dilation,
            groups=original_conv.groups,
            bias=original_conv.bias is not None
        )

        # 参数初始化
        with torch.no_grad():
            # re-initialize the weights
            new_weight = torch.zeros_like(
                new_conv.weight[:, :original_in_channels*2, :, :],
                device=new_conv.weight.device,
                dtype=new_conv.weight.dtype
            )
            
            # 前一半通道使用预训练参数
            new_weight[:, :original_in_channels] = original_conv.weight
            
            # 后一半通道保持0初始化（默认已经是0，这里显式强调）
            new_weight[:, original_in_channels:] = 0.0
            
            new_conv.weight.copy_(new_weight)
            
            # 复制偏置参数
            if new_conv.bias is not None:
                new_conv.bias.copy_(original_conv.bias)

        # 替换模型中的卷积层
        unet.conv_in = new_conv
        unet.config.in_channels = original_in_channels * 2
    return unet


class ConcatInputStableDiffusionPipeline(StableDiffusionPipeline):  # ONLY modified 3 lines lol
    # NOTE: COPIED from diffusers repo
    @torch.no_grad()
    # @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        cond_latents = None, # NOTE: added for concating a image's latents
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        ip_adapter_image = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            cond_latents (`torch.Tensor`):
                Pre-generated latents of the target image. Used to guide diffusion process for controlable generation.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
            ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
                Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
                IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
                contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
                provided, embeddings are computed from the `ip_adapter_image` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)

        if callback is not None:
            deprecate(
                "callback",
                "1.0.0",
                "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 0. Default height and width to unet
        if not height or not width:
            height = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[0]
            )
            width = (
                self.unet.config.sample_size
                if self._is_unet_config_sample_size_int
                else self.unet.config.sample_size[1]
            )
            height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            ip_adapter_image,
            ip_adapter_image_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
        )

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
            image_embeds = self.prepare_ip_adapter_image_embeds(
                ip_adapter_image,
                ip_adapter_image_embeds,
                device,
                batch_size * num_images_per_prompt,
                self.do_classifier_free_guidance,
            )

        # 4. Prepare timesteps
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Add image embeds for IP-Adapter
        added_cond_kwargs = (
            {"image_embeds": image_embeds}
            if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
            else None
        )

        # 6.2 Optionally get Guidance Scale Embedding
        timestep_cond = None
        if self.unet.config.time_cond_proj_dim is not None:
            guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
            timestep_cond = self.get_guidance_scale_embedding(
                guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
            ).to(device=device, dtype=latents.dtype)

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
                cond_latent_input = torch.cat([cond_latents] * 2) if self.do_classifier_free_guidance else cond_latents  # NOTE: added
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    torch.cat([latent_model_input, cond_latent_input], dim=1),   # NOTE: concate input!!!
                    t,
                    encoder_hidden_states=prompt_embeds,
                    timestep_cond=timestep_cond,
                    cross_attention_kwargs=self.cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

                if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if XLA_AVAILABLE:
                    xm.mark_step()

        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
                0
            ]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process input and output paths along with a BDMAP ID.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the ct to be translated.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--target_phase", type=str, required=True, help="Tanslation target phase. (non_contrast, arterial, delayed, portal-venous)")
    parser.add_argument("--reference_excel", type=str, required=True, help="The .csv/.xlsx file that contains BDMAP ID info.")
    parser.add_argument("--ct_folder_path", type=str, default=None, help="Path to the folder that contains the original CT nii.gz file. If provided, will use the original nii header and affine for saving.")
    parser.add_argument("--finetuned_vae_name_or_path", type=str, required=True, help="Path to the trained VAE model.")
    parser.add_argument("--finetuned_unet_name_or_path", type=str, required=True, help="Path to the trained UNet model.")
    parser.add_argument("--sd_model_name_or_path", type=str, required=True, help="Path to the trained stable diffusion model.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

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
    resolution = 512 # NOTE: DO NOT CHANGE

    # Dataset settings
    data_dir = args.input_path
    save_dir = args.output_path
    os.makedirs(save_dir, exist_ok=True)

    inference_transforms = A.Compose([      # resize 
        A.Resize(resolution, resolution, interpolation=cv2.INTER_CUBIC), # model requires 512
    ])
    chunk_size = 32  # NOTE: how many 3-channel images to input at once


    # wrap the CT as a dataset for batch_size > 1
    ct_dataset = CTDatasetInferenceH5(
        file_path=data_dir,#os.path.join(data_dir, bdmap_id, "ct.h5"),    # from the reconstruction method
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
    original_phase = args.input_path.split("/")[-2].split("_")[-1]  
    patient_id = args.input_path.split("/")[-2].split("_")[0] 
    print("Now running single case inference,\n shape info: ", nii_shape, f"\n patient id: {patient_id}, {original_phase} --> {args.target_phase}")
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
            
            if slice_id + 3 > D:# boundary check
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
        norm = ((channel + 1000) / 2000 * 255).clip(0, 255).astype(np.uint8)
        norm = np.rot90(norm) 
        Image.fromarray(norm).save(
            os.path.join(save_dir, f"{original_phase}_to_{args.target_phase}_{i:02d}.png")
        )

    os.makedirs(os.path.join(save_dir), exist_ok=True)

    # read the original nii header and affine
    # IMPORTANT: here login.ccvl.jhu.edu is not granted access to abdomen atlas pro data
    if args.ct_folder_path is not None:
        original_nfiti = nib.load(os.path.join(args.ct_folder_path, f"{patient_id}_{original_phase}.nii.gz"))
    
    
    else:

        print("Finding the original CT from abdomen atlas pro...")
        try:
            target_bdmap_ids_raw = pd.read_csv(args.reference_excel)
        except:
            target_bdmap_ids_raw = pd.read_excel(args.reference_excel)
            
        bdmap_id = target_bdmap_ids_raw[target_bdmap_ids_raw["Patient_ID"] == f"{patient_id}"][original_phase]
        # find from the larger dataset
        try:
            original_nfiti = nib.load(os.path.join("/projects/bodymaps/Data/image_only/AbdomenAtlasPro/AbdomenAtlasPro", bdmap_id.item(), "ct.nii.gz"))  # load the original CT volume
        except:
            original_nfiti = nib.load(os.path.join("/mnt/data/jliu452/Data/AbdomenAtlasKuma", bdmap_id.item(), "ct.nii.gz")) 
   
    out_nii = nib.Nifti1Image(enhanced_ct, original_nfiti.affine, original_nfiti.header)
    out_nii.header.set_data_dtype(np.int16)
    out_nii.to_filename(os.path.join(save_dir, f"ct.nii.gz")) 


