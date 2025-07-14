#!/usr/bin/env python3
"""
Automated GoPro Highlight Extraction Script (Python Version)
Usage: python extract_highlights.py /path/to/gopro/videos /path/to/output
"""

import asyncio
import argparse
import logging
import subprocess
import sys
from pathlib import Path

from config import Config, get_optimal_workers
from video_processor import VideoProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('highlight_extraction.log')
    ]
)
logger = logging.getLogger(__name__)


async def main():
    """Main function to orchestrate the highlight extraction process."""
    parser = argparse.ArgumentParser(description='Extract highlights from GoPro videos')
    parser.add_argument('input_dir', type=Path, help='Directory containing GoPro videos')
    parser.add_argument('output_dir', type=Path, help='Directory to save highlight clips')
    parser.add_argument('--clip-duration', type=int, default=15, help='Duration of each clip in seconds')
    parser.add_argument('--motion-threshold', type=float, default=0.2, help='Motion detection threshold (ffmpeg fallback)')
    parser.add_argument('--transnetv2-threshold', type=float, default=0.5, help='TransNetV2 detection threshold (0.0-1.0)')
    parser.add_argument('--min-gap', type=int, default=10, help='Minimum gap between highlights in seconds')
    parser.add_argument('--max-workers', type=int, default=get_optimal_workers(), help=f'Maximum concurrent workers (default: {get_optimal_workers()} based on system)')
    parser.add_argument('--disable-hilight-tags', action='store_true', help='Disable GoPro HiLight tag extraction')
    parser.add_argument('--scene-detection', choices=['transnetv2', 'ffmpeg', 'auto'], default='transnetv2', 
                       help='Scene detection method: transnetv2 (neural network), ffmpeg (traditional), auto (transnetv2 with ffmpeg fallback)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = Config(
        clip_duration=args.clip_duration,
        motion_threshold=args.motion_threshold,
        transnetv2_threshold=args.transnetv2_threshold,
        min_gap_seconds=args.min_gap,
        max_workers=args.max_workers,
        prefer_hilight_tags=not args.disable_hilight_tags,
        scene_detection_method=args.scene_detection
    )
    
    # Find all video files
    video_files = []
    for ext in config.video_extensions:
        video_files.extend(args.input_dir.glob(f"*{ext}"))
    
    if not video_files:
        logger.error(f"No video files found in {args.input_dir}")
        sys.exit(1)
    
    logger.info(f"🎬 Found {len(video_files)} videos to process")
    logger.info(f"⚡ Using {config.max_workers} concurrent workers for optimal performance")
    
    if config.prefer_hilight_tags:
        logger.info("🏷️  GoPro HiLight tag extraction enabled (recommended)")
    else:
        logger.info("🏷️  GoPro HiLight tag extraction disabled")
    
    if config.scene_detection_method == "transnetv2":
        logger.info("🧠 Using TransNetV2 for scene detection (neural network)")
    elif config.scene_detection_method == "ffmpeg":
        logger.info("📊 Using ffmpeg for scene detection (traditional)")
    else:
        logger.info("🔄 Auto mode: TransNetV2 with ffmpeg fallback")
    
    print("=" * 50)
    
    # Process videos
    processor = VideoProcessor(config)
    total_hilight_clips = 0
    total_motion_clips = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"📹 Video {i} of {len(video_files)}: {video_file.name}")
        hilight_clips, motion_clips = await processor.process_video(video_file, args.output_dir)
        total_hilight_clips += hilight_clips
        total_motion_clips += motion_clips
        print()
    
    print(f"🎉 Highlight extraction complete!")
    print(f"🏷️  HiLight clips extracted: {total_hilight_clips}")
    print(f"🎬 Motion-detected clips extracted: {total_motion_clips}")
    print(f"📁 All clips saved to: {args.output_dir}")
    print(f"   ├── HiLights/ ({total_hilight_clips} clips)")
    print(f"   └── Motion Detected Highlights/ ({total_motion_clips} clips)")


if __name__ == "__main__":
    # Check for required dependencies
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg and ffprobe are required but not found in PATH")
        sys.exit(1)
    
    # Run the main function
    asyncio.run(main()) 