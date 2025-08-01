from typing import Callable, Dict, List, Optional, Union

import torch
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.models import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput

import scipy
import numpy as np
import torch.nn.functional as F
from transformer_minimax_remover import Transformer3DModel
from einops import rearrange

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

class Minimax_Remover_Pipeline(DiffusionPipeline):

    model_cpu_offload_seq = "transformer->vae"
    _callback_tensor_inputs = ["latents"]

    def __init__(
        self,
        transformer: Transformer3DModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
        )

        self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: 16,
        height: int = 720,
        width: int = 1280,
        num_latent_frames: int = 21,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def expand_masks(self, masks, iterations):
        masks = masks.cpu().detach().numpy()
        # numpy array, masks [0,1], f h w c
        masks2 = []
        for i in range(len(masks)):
            mask = masks[i]
            mask = mask > 0.5  # Use threshold instead of >0 for better precision
            
            if iterations > 0:
                # Use erosion followed by dilation for cleaner edges
                import cv2
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (iterations*2+1, iterations*2+1))
                mask = mask.astype(np.uint8) * 255
                # Light dilation to ensure coverage, but preserve edges
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = mask > 127
            
            masks2.append(mask.astype(np.float32))
        
        masks = np.array(masks2).astype(np.float32)
        masks = torch.from_numpy(masks)
        
        # Add channel dimension if not present
        if len(masks.shape) == 3:  # [f, h, w]
            masks = masks[:, :, :, None]  # [f, h, w, 1]
        
        # Repeat for 3 channels (RGB)
        masks = masks.repeat(1, 1, 1, 3)  # [f, h, w, 3]
        
        # Rearrange to [c, f, h, w]
        masks = rearrange(masks, "f h w c -> c f h w")
        
        # Add batch dimension [1, c, f, h, w]
        masks = masks[None, ...]
        
        print(f"Expand_masks output shape: {masks.shape}")
        return masks

    def resize(self, images, w, h):
        bsz,_,_,_,_ = images.shape
        images = rearrange(images, "b c f w h -> (b f) c w h")
        # Use bicubic for better quality, especially for text
        images = F.interpolate(images, (w,h), mode='bicubic', align_corners=False)
        images = rearrange(images, "(b f) c w h -> b c f w h", b=bsz)
        return images

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        images: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        iterations: int = 16
    ):

        self._current_timestep = None
        self._interrupt = False
        device = self._execution_device
        batch_size = 1
        transformer_dtype = torch.float16

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = 16
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        latents = self.prepare_latents(
            batch_size,
            num_channels_latents,
            height,
            width,
            num_latent_frames,
            torch.float16,
            device,
            generator,
            latents,
        )

        masks = self.expand_masks(masks, iterations)
        masks = self.resize(masks, height, width).to(device).half()
        masks[masks>0] = 1
        images = rearrange(images, "f h w c -> c f h w")
        images = self.resize(images[None,...], height, width).to(device).half()

        # Debug shapes
        print(f"Images shape: {images.shape}")
        print(f"Masks shape: {masks.shape}")
        
        # Ensure masks and images have compatible shapes
        if masks.shape != images.shape:
            # Adjust masks to match images dimensions
            if len(masks.shape) == 5 and len(images.shape) == 5:
                # Both are 5D, check channel dimension
                if masks.shape[1] != images.shape[1]:
                    # Repeat mask channels to match image channels
                    masks = masks.repeat(1, images.shape[1] // masks.shape[1], 1, 1, 1)
        
        print(f"After adjustment - Images: {images.shape}, Masks: {masks.shape}")
        masked_images = images * (1-masks)

        latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(device, torch.float16)
            )

        latents_std =  1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                device, torch.float16
            )

        with torch.no_grad():
            masked_latents = self.vae.encode(masked_images.half()).latent_dist.mode()
            masks_latents = self.vae.encode(2*masks.half()-1.0).latent_dist.mode()

            masked_latents = (masked_latents - latents_mean) * latents_std
            masks_latents = (masks_latents - latents_mean) * latents_std

        self._num_timesteps = len(timesteps)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                latent_model_input = latents.to(transformer_dtype)

                latent_model_input = torch.cat([latent_model_input, masked_latents, masks_latents], dim=1)
                timestep = t.expand(latents.shape[0])

                noise_pred = self.transformer(
                    hidden_states=latent_model_input.half(),
                    timestep=timestep
                )[0]

                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                
                progress_bar.update()

        latents = latents.half() / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)

        return WanPipelineOutput(frames=video)
