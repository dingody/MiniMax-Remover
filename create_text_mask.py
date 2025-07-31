#!/usr/bin/env python3
"""
创建更精确的文字掩码工具
针对固定位置的文字水印生成软边缘掩码
"""

import cv2
import numpy as np
from decord import VideoReader
import torch
from moviepy.editor import ImageSequenceClip
import argparse

def create_soft_mask(frame_shape, text_regions, feather_size=20):
    """
    创建软边缘掩码
    Args:
        frame_shape: (height, width) 视频帧尺寸
        text_regions: [(x, y, w, h), ...] 文字区域列表
        feather_size: 羽化大小，创建软边缘
    """
    height, width = frame_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    for (x, y, w, h) in text_regions:
        # 1. 创建基础矩形掩码
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    
    # 2. 高斯模糊创建软边缘 - 多次模糊获得更好效果
    mask_float = mask.astype(np.float32) / 255.0
    
    # 第一次大范围模糊
    mask_blur1 = cv2.GaussianBlur(mask_float, (feather_size*2+1, feather_size*2+1), feather_size/2)
    
    # 第二次小范围模糊细化边缘
    mask_blur2 = cv2.GaussianBlur(mask_blur1, (feather_size//2*2+1, feather_size//2*2+1), feather_size/4)
    
    # 3. 组合创建渐变边缘
    final_mask = mask_blur2 * 0.7 + mask_blur1 * 0.3
    
    # 4. 确保中心区域是完全不透明的
    core_mask = np.zeros_like(mask)
    for (x, y, w, h) in text_regions:
        # 创建稍小的核心区域
        core_x = x + w//6
        core_y = y + h//6
        core_w = w - w//3
        core_h = h - h//3
        cv2.rectangle(core_mask, (core_x, core_y), (core_x+core_w, core_y+core_h), 255, -1)
    
    core_mask = core_mask.astype(np.float32) / 255.0
    final_mask = np.maximum(final_mask, core_mask)
    
    return (final_mask * 255).astype(np.uint8)

def create_mask_video(input_video_path, output_mask_path, text_regions, feather_size=20):
    """
    为整个视频创建掩码
    """
    vr = VideoReader(input_video_path)
    total_frames = len(vr)
    first_frame = vr[0].asnumpy()
    height, width = first_frame.shape[:2]
    
    print(f"Creating mask for {total_frames} frames, resolution: {width}x{height}")
    print(f"Text regions: {text_regions}")
    print(f"Feather size: {feather_size}")
    
    # 创建掩码模板
    mask_template = create_soft_mask((height, width), text_regions, feather_size)
    
    # 为所有帧创建相同的掩码
    mask_frames = []
    for i in range(total_frames):
        # 将单通道掩码转换为3通道（模拟视频格式）
        mask_3ch = cv2.cvtColor(mask_template, cv2.COLOR_GRAY2RGB)
        mask_frames.append(mask_3ch)
        
        if i % 50 == 0:
            print(f"Generated mask for frame {i}/{total_frames}")
    
    # 保存掩码视频
    print(f"Saving mask video to: {output_mask_path}")
    fps = vr.get_avg_fps()
    clip = ImageSequenceClip(mask_frames, fps=fps)
    clip.write_videofile(output_mask_path, codec='libx264', audio=False, verbose=False, logger=None)
    
    print(f"Mask video created successfully!")
    
    # 显示掩码预览
    preview_path = output_mask_path.replace('.mp4', '_preview.png')
    cv2.imwrite(preview_path, mask_template)
    print(f"Mask preview saved to: {preview_path}")

def main():
    parser = argparse.ArgumentParser(description='Create text mask for video')
    parser.add_argument('--input', default='/content/input_fixed.mp4', help='Input video path')
    parser.add_argument('--output', default='/content/mask_fixed.mp4', help='Output mask video path')
    parser.add_argument('--regions', nargs='+', type=int, 
                       help='Text regions as x y w h (can specify multiple: x1 y1 w1 h1 x2 y2 w2 h2)')
    parser.add_argument('--feather', type=int, default=15, help='Feather size for soft edges')
    
    args = parser.parse_args()
    
    # 解析文字区域
    if args.regions:
        if len(args.regions) % 4 != 0:
            print("Error: regions must be specified as groups of 4 numbers (x y w h)")
            return
        
        text_regions = []
        for i in range(0, len(args.regions), 4):
            x, y, w, h = args.regions[i:i+4]
            text_regions.append((x, y, w, h))
    else:
        # 默认区域（需要根据你的视频调整）
        print("No regions specified, using default region")
        print("You should specify regions with --regions x y w h")
        # 示例：假设文字在底部中央
        text_regions = [(200, 1000, 320, 100)]  # 需要根据实际文字位置调整
    
    create_mask_video(args.input, args.output, text_regions, args.feather)

if __name__ == "__main__":
    main()