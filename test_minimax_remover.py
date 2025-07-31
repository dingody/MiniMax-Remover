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
def inference(pixel_values, masks, video_length, iterations=2):
    # For 720x1280 video (width x height), use appropriate resolution
    # Keep aspect ratio: 720/1280 = 0.5625
    # Use smaller resolution for L4 GPU
    width = 512   # Corresponding to original width 720
    height = int(512 / 0.5625)  # = 910, round to 896 for efficiency  
    height = 896  # Corresponding to original height 1280
    
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
    export_to_video(video, "/content/output_fixed.mp4")

def load_video(video_path, max_frames=None):
    vr = VideoReader(video_path)
    total_frames = len(vr)
    print(f"Total frames in video: {total_frames}")
    
    # For 15s video, limit to reasonable number of frames for L4 GPU
    if max_frames is None:
        # For L4 GPU, limit to 60 frames max to avoid OOM
        max_frames = min(60, total_frames)
    
    # Sample frames evenly if video is longer
    if total_frames > max_frames:
        frame_indices = list(range(0, total_frames, total_frames // max_frames))[:max_frames]
    else:
        frame_indices = list(range(total_frames))
    
    print(f"Processing {len(frame_indices)} frames: {frame_indices[:5]}...")
    images = vr.get_batch(frame_indices).asnumpy()
    images = torch.from_numpy(images)/127.5 - 1.0
    return images, len(frame_indices)

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
images, video_length = load_video(video_path)
print(f"Loaded video with {video_length} frames, shape: {images.shape}")

# Get frame indices for mask loading
if video_length <= 60:
    frame_indices = list(range(video_length))
else:
    vr_temp = VideoReader(video_path)
    total_frames = len(vr_temp)
    frame_indices = list(range(0, total_frames, total_frames // video_length))[:video_length]
    del vr_temp

masks = load_mask(mask_path, frame_indices)
print(f"Loaded masks with shape: {masks.shape}")

# Add optimization: enable model CPU offloading for L4 GPU
try:
    pipe.enable_model_cpu_offload()
    print("Enabled model CPU offloading")
except:
    print("CPU offloading not available")

inference(images, masks, video_length)