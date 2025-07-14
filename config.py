#!/usr/bin/env python3
"""
Configuration and utility functions for GoPro Highlights Script.
"""

import os
from dataclasses import dataclass
from typing import List


def get_optimal_workers() -> int:
    """
    Determine the optimal number of concurrent workers for video processing.
    
    For CPU-intensive tasks like video processing, we want to balance:
    - Available CPU cores
    - Memory constraints (video processing is memory-intensive)
    - I/O limitations (multiple ffmpeg processes)
    
    Returns a reasonable default that won't overwhelm the system.
    """
    cpu_count = os.cpu_count() or 4  # Fallback to 4 if cpu_count() returns None
    
    # For video processing, we typically don't want to use all cores
    # as ffmpeg itself can be multi-threaded and video processing is I/O intensive
    # A good rule of thumb is to use 75% of available cores, with a minimum of 2
    # and a reasonable maximum to avoid overwhelming the system
    optimal = max(2, min(cpu_count * 3 // 4, 8))
    
    return optimal


@dataclass
class Config:
    """Configuration for highlight extraction."""
    clip_duration: int = 15  # seconds
    motion_threshold: float = 0.2  # adjust based on your footage (used for ffmpeg fallback)
    transnetv2_threshold: float = 0.5  # TransNetV2 detection threshold
    min_gap_seconds: int = 10  # minimum gap between highlights to avoid duplicates
    max_workers: int = None  # for parallel processing - will be set to optimal value
    video_extensions: List[str] = None
    prefer_hilight_tags: bool = True  # prioritize GoPro HiLight tags over scene detection
    scene_detection_method: str = "transnetv2"  # "transnetv2", "ffmpeg", "auto"
    
    def __post_init__(self):
        if self.video_extensions is None:
            self.video_extensions = ['.mp4', '.MP4', '.mov', '.MOV']
        if self.max_workers is None:
            self.max_workers = get_optimal_workers() 