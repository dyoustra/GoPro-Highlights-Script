#!/usr/bin/env python3
"""
Video processing functionality for GoPro Highlights Script.
"""

import asyncio
import logging
import re
import tempfile
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

from config import Config
from ui_components import ProgressBar, SpinnerContext
from gopro_hilight_extractor import GoProHiLightExtractor
from transnetv2_detector import TransNetV2Detector

logger = logging.getLogger(__name__)


class VideoProcessor:
    """Main class for processing GoPro videos and extracting highlights."""
    
    def __init__(self, config: Config):
        self.config = config
        self.hilight_extractor = GoProHiLightExtractor()
        self.transnetv2_detector = TransNetV2Detector(threshold=config.transnetv2_threshold)
        
    async def get_video_duration(self, video_path: Path) -> float:
        """Get video duration using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)
            ]
            result = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {stderr.decode()}")
                
            return float(stdout.decode().strip())
            
        except Exception as e:
            logger.error(f"Failed to get duration for {video_path}: {e}")
            raise
    
    async def detect_scenes_transnetv2(self, video_path: Path) -> List[float]:
        """Detect scenes using TransNetV2."""
        try:
            return await self.transnetv2_detector.detect_scenes(video_path)
        except Exception as e:
            logger.error(f"TransNetV2 detection failed for {video_path}: {e}")
            # Fall back to ffmpeg detection
            logger.info("Falling back to ffmpeg scene detection")
            return await self.transnetv2_detector._fallback_detection(video_path)
    
    async def analyze_motion_vectors(self, video_path: Path, temp_dir: Path) -> List[float]:
        """Analyze motion vectors to find high-action scenes (ffmpeg fallback)."""
        motion_file = temp_dir / "motion_analysis.txt"
        
        # Use scene detection and motion analysis
        cmd = [
            'ffmpeg', '-i', str(video_path),
            '-vf', f"select='gt(scene,{self.config.motion_threshold})',showinfo",
            '-f', 'null', '-'
        ]
        
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE
            )
            
            _, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.warning(f"Motion analysis had issues, but continuing...")
            
            # Write stderr to file for parsing
            motion_file.write_text(stderr.decode())
            
            # Extract timestamps from motion analysis
            timestamps = self._extract_timestamps_from_motion_analysis(motion_file)
            return timestamps
            
        except Exception as e:
            logger.error(f"Motion analysis failed for {video_path}: {e}")
            return []
    
    def _extract_timestamps_from_motion_analysis(self, motion_file: Path) -> List[float]:
        """Extract timestamps from motion analysis output."""
        timestamps = []
        
        try:
            content = motion_file.read_text()
            # Look for pts_time patterns in the output
            pattern = r'pts_time:(\d+\.?\d*)'
            matches = re.findall(pattern, content)
            
            for match in matches:
                timestamps.append(float(match))
                
        except Exception as e:
            logger.error(f"Failed to extract timestamps: {e}")
            
        return timestamps
    
    def _filter_timestamps(self, timestamps: List[float]) -> List[float]:
        """Filter out duplicate/overlapping timestamps."""
        if not timestamps:
            return []
            
        # Sort timestamps
        sorted_timestamps = sorted(timestamps)
        filtered = [sorted_timestamps[0]]
        
        for timestamp in sorted_timestamps[1:]:
            if timestamp - filtered[-1] > self.config.min_gap_seconds:
                filtered.append(timestamp)
                
        return filtered
    
    async def extract_clip(self, video_path: Path, timestamp: float, output_path: Path) -> bool:
        """Extract a single clip at the given timestamp."""
        try:
            cmd = [
                'ffmpeg', '-ss', str(timestamp), '-i', str(video_path),
                '-t', str(self.config.clip_duration), '-c', 'copy',
                '-avoid_negative_ts', 'make_zero', str(output_path)
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            
            await process.communicate()
            
            # Check if the file was created and has reasonable size
            if output_path.exists() and output_path.stat().st_size > 1000:  # At least 1KB
                return True
            else:
                logger.warning(f"Clip extraction failed or produced empty file: {output_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to extract clip at {timestamp}s: {e}")
            return False
    
    async def process_video(self, video_path: Path, output_dir: Path) -> Tuple[int, int]:
        """
        Process a single video file and extract highlights.
        
        Returns:
            Tuple of (hilight_clips_extracted, motion_clips_extracted)
        """
        logger.info(f"Processing: {video_path}")
        
        base_name = video_path.stem
        hilight_clips = 0
        motion_clips = 0
        
        # Create subdirectories for different types of highlights
        hilight_dir = output_dir / "HiLights"
        motion_dir = output_dir / "Motion Detected Highlights"
        
        try:
            hilight_dir.mkdir(parents=True, exist_ok=True)
            motion_dir.mkdir(parents=True, exist_ok=True)
        except (PermissionError, NotADirectoryError, FileExistsError) as e:
            # Handle mkdir errors gracefully (FileExistsError shouldn't happen with exist_ok=True but let's be safe)
            if isinstance(e, FileExistsError):
                # This is weird but let's handle it gracefully in tests
                logger.warning(f"Directory already exists (this should not happen with exist_ok=True): {e}")
            else:
                logger.error(f"Failed to create output directories: {e}")
                return 0, 0
        
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            try:
                # Step 1: Extract GoPro HiLight tags first (if enabled)
                hilight_timestamps = []
                if self.config.prefer_hilight_tags:
                    async with SpinnerContext("Extracting GoPro HiLight tags"):
                        hilight_timestamps = await self.hilight_extractor.extract_hilight_tags(video_path)
                    
                    if hilight_timestamps:
                        logger.info(f"Found {len(hilight_timestamps)} GoPro HiLight tags")
                        
                        # Filter and extract HiLight clips
                        filtered_hilight_timestamps = self._filter_timestamps(hilight_timestamps)
                        
                        if filtered_hilight_timestamps:
                            print(f"🏷️  Extracting {len(filtered_hilight_timestamps)} HiLight clips...")
                            
                            progress = ProgressBar(len(filtered_hilight_timestamps))
                            
                            # Use semaphore to limit concurrent ffmpeg processes
                            semaphore = asyncio.Semaphore(self.config.max_workers)
                            
                            async def extract_hilight_clip(timestamp: float, clip_num: int) -> bool:
                                async with semaphore:
                                    output_file = hilight_dir / f"{base_name}_highlight_{clip_num:03d}.mp4"
                                    return await self.extract_clip(video_path, timestamp, output_file)
                            
                            # Process HiLight clips concurrently
                            tasks = []
                            for i, timestamp in enumerate(filtered_hilight_timestamps, 1):
                                task = extract_hilight_clip(timestamp, i)
                                tasks.append(task)
                            
                            # Process in batches to show progress
                            for i, task in enumerate(asyncio.as_completed(tasks), 1):
                                success = await task
                                if success:
                                    hilight_clips += 1
                                progress.update(i)
                            
                            progress.finish()
                            logger.info(f"✓ Extracted {hilight_clips} HiLight clips from {base_name}")
                    else:
                        logger.info("No GoPro HiLight tags found")
                
                # Step 2: Perform scene detection for motion-based highlights
                motion_timestamps = []
                
                if self.config.scene_detection_method == "transnetv2":
                    async with SpinnerContext("Detecting scenes with TransNetV2"):
                        motion_timestamps = await self.detect_scenes_transnetv2(video_path)
                    
                    # Fall back to ffmpeg if TransNetV2 fails
                    if not motion_timestamps:
                        logger.warning("TransNetV2 detection failed, falling back to ffmpeg")
                        async with SpinnerContext("Analyzing motion vectors (ffmpeg fallback)"):
                            motion_timestamps = await self.analyze_motion_vectors(video_path, temp_dir)
                
                elif self.config.scene_detection_method == "ffmpeg":
                    async with SpinnerContext("Analyzing motion vectors with ffmpeg"):
                        motion_timestamps = await self.analyze_motion_vectors(video_path, temp_dir)
                
                else:  # auto mode
                    # Try TransNetV2 first, fall back to ffmpeg
                    async with SpinnerContext("Detecting scenes with TransNetV2"):
                        motion_timestamps = await self.detect_scenes_transnetv2(video_path)
                    
                    if not motion_timestamps:
                        async with SpinnerContext("Analyzing motion vectors (ffmpeg fallback)"):
                            motion_timestamps = await self.analyze_motion_vectors(video_path, temp_dir)
                
                if motion_timestamps:
                    # Filter motion timestamps
                    filtered_motion_timestamps = self._filter_timestamps(motion_timestamps)
                    
                    logger.info(f"Found {len(motion_timestamps)} potential motion highlights, filtered to {len(filtered_motion_timestamps)} unique highlights")
                    
                    if filtered_motion_timestamps:
                        print(f"🎬 Extracting {len(filtered_motion_timestamps)} motion-detected highlight clips...")
                        
                        progress = ProgressBar(len(filtered_motion_timestamps))
                        
                        # Use semaphore to limit concurrent ffmpeg processes
                        semaphore = asyncio.Semaphore(self.config.max_workers)
                        
                        async def extract_motion_clip(timestamp: float, clip_num: int) -> bool:
                            async with semaphore:
                                output_file = motion_dir / f"{base_name}_motion_{clip_num:03d}.mp4"
                                return await self.extract_clip(video_path, timestamp, output_file)
                        
                        # Process motion clips concurrently
                        tasks = []
                        for i, timestamp in enumerate(filtered_motion_timestamps, 1):
                            task = extract_motion_clip(timestamp, i)
                            tasks.append(task)
                        
                        # Process in batches to show progress
                        for i, task in enumerate(asyncio.as_completed(tasks), 1):
                            success = await task
                            if success:
                                motion_clips += 1
                            progress.update(i)
                        
                        progress.finish()
                        logger.info(f"✓ Extracted {motion_clips} motion-detected clips from {base_name}")
                else:
                    logger.warning(f"No motion highlights found in {video_path}")
                
                return hilight_clips, motion_clips
                
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                return 0, 0