import torch
from diffusers.utils import export_to_video
from decord import VideoReader
from diffusers.models import AutoencoderKLWan
from transformer_minimax_remover import Transformer3DModel
from diffusers.schedulers import UniPCMultistepScheduler
from pipeline_minimax_remover import Minimax_Remover_Pipeline

random_seed = 42
# video_length will be set dynamically
device = torch.device("cuda:0")

# Get video info first - Colab paths
video_path = "/content/input_fixed.mp4"
mask_path = "/content/mask_fixed.mp4"

vae = AutoencoderKLWan.from_pretrained("./vae", torch_dtype=torch.float16)
transformer = Transformer3DModel.from_pretrained("./transformer", torch_dtype=torch.float16)
scheduler = UniPCMultistepScheduler.from_pretrained("./scheduler")

pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler)
pipe.to(device)

# the iterations is the hyperparameter for mask dilation
def inference(pixel_values, masks, video_length, iterations=1):
    # For 720x1280 video (width x height), use higher resolution for better quality
    # Keep aspect ratio: 720/1280 = 0.5625
    # Balance between quality and L4 GPU memory
    width = 576   # Increased from 512, closer to original 720
    height = int(576 / 0.5625)  # = 1024, exact match for aspect ratio
    height = 1024  # Better match for original 1280
    
    video = pipe(
        images=pixel_values,
        masks=masks,
        num_frames=video_length,
        height=height,
        width=width,
        num_inference_steps=6,  # Reduced from 12 to 6 for speed
        generator=torch.Generator(device=device).manual_seed(random_seed),
        iterations=iterations  # Reduced from 6 to 2
    ).frames[0]
    # Export with proper fps to maintain video duration
    export_to_video(video, "/content/output_fixed.mp4", fps=25)

def load_video(video_path, max_frames=None):
    vr = VideoReader(video_path)
    total_frames = len(vr)
    fps = vr.get_avg_fps()
    duration = total_frames / fps
    print(f"Video info: {total_frames} frames, {fps:.2f} fps, {duration:.2f}s duration")
    
    # For L4 GPU, increase frame limit but keep it reasonable
    if max_frames is None:
        # Allow more frames to preserve video length better
        max_frames = min(120, total_frames)  # Increased from 60 to 120
    
    # Sample frames more evenly to preserve timing
    if total_frames > max_frames:
        # Use uniform sampling to maintain temporal consistency
        step = total_frames / max_frames
        frame_indices = [int(i * step) for i in range(max_frames)]
    else:
        frame_indices = list(range(total_frames))
    
    print(f"Sampling {len(frame_indices)} frames from {total_frames} total frames")
    print(f"Frame indices: {frame_indices[:5]}...{frame_indices[-5:] if len(frame_indices) > 5 else ''}")
    
    images = vr.get_batch(frame_indices).asnumpy()
    images = torch.from_numpy(images)/127.5 - 1.0
    return images, len(frame_indices), frame_indices

def load_mask(mask_path, frame_indices):
    vr = VideoReader(mask_path)
    total_frames = len(vr)
    
    # Use same frame indices as video
    if len(frame_indices) > total_frames:
        # If we need more frames than available, repeat the last frame
        actual_indices = frame_indices[:total_frames] + [total_frames-1] * (len(frame_indices) - total_frames)
    else:
        actual_indices = frame_indices
    
    masks = vr.get_batch(actual_indices).asnumpy()
    masks = torch.from_numpy(masks)
    masks = masks[:, :, :, :1]
    masks[masks > 20] = 255
    masks[masks < 255] = 0
    masks = masks / 255.0
    return masks

# Load video with dynamic frame count
images, video_length, frame_indices = load_video(video_path)
print(f"Loaded video with {video_length} frames, shape: {images.shape}")

# Load corresponding masks
masks = load_mask(mask_path, frame_indices)
print(f"Loaded masks with shape: {masks.shape}")

# For L4 GPU with 22.5GB VRAM, try CPU offloading carefully
# If it causes device issues, we'll disable it
try:
    # Use attention slicing instead of full CPU offloading
    pipe.enable_attention_slicing()
    print("Enabled attention slicing for memory optimization")
    
    # Optionally try CPU offloading (comment out if it causes issues)
    # pipe.enable_model_cpu_offload()
    # print("Enabled model CPU offloading")
except Exception as e:
    print(f"Memory optimization not available: {e}")

inference(images, masks, video_length)