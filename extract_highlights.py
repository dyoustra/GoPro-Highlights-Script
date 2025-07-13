#!/usr/bin/env python3
"""
Automated GoPro Highlight Extraction Script (Python Version)
Usage: python extract_highlights.py /path/to/gopro/videos /path/to/output
"""

import asyncio
import argparse
import logging
import os
import re
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
import json
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import time

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
    motion_threshold: float = 0.2  # adjust based on your footage
    min_gap_seconds: int = 10  # minimum gap between highlights to avoid duplicates
    max_workers: int = None  # for parallel processing - will be set to optimal value
    video_extensions: List[str] = None
    prefer_hilight_tags: bool = True  # prioritize GoPro HiLight tags over motion analysis
    
    def __post_init__(self):
        if self.video_extensions is None:
            self.video_extensions = ['.mp4', '.MP4', '.mov', '.MOV']
        if self.max_workers is None:
            self.max_workers = get_optimal_workers()


class GoProHiLightExtractor:
    """
    Extracts GoPro HiLight tags from MP4 files.
    
    This class parses MP4 container structure to find and extract HiLight tags
    that users create during recording by pressing the mode button on their GoPro.
    These tags are much more accurate than motion-based analysis.
    
    Supports both newer (HERO6+) and older GoPro formats.
    """
    
    def __init__(self):
        pass
    
    def find_boxes(self, f, start_offset: int = 0, end_offset: float = float("inf")) -> dict:
        """
        Parse MP4 box structure to find metadata containers.
        
        Args:
            f: File object positioned at the start of boxes
            start_offset: Byte offset to start parsing from
            end_offset: Byte offset to stop parsing at
            
        Returns:
            Dictionary mapping box types to (start_offset, end_offset) tuples
        """
        # Struct formats for parsing MP4 boxes
        # > means big-endian, I = unsigned int (4 bytes), 4s = 4-byte string, Q = unsigned long long (8 bytes)
        box_header_struct = struct.Struct("> I 4s")
        extended_size_struct = struct.Struct("> Q")
        
        boxes = {}
        offset = start_offset
        f.seek(offset, 0)
        
        while offset < end_offset:
            # Read the 8-byte box header
            data = f.read(8)
            if data == b"":  # EOF
                break
                
            try:
                length, box_type = box_header_struct.unpack(data)
            except struct.error:
                logger.warning(f"Failed to parse box header at offset {offset}")
                break
            
            # Handle UUID boxes (not implemented as they're rare in GoPro files)
            if box_type == b'uuid':
                logger.warning("UUID box type encountered - skipping")
                offset += length if length > 0 else 8
                f.seek(offset)
                continue
            
            # Handle extended size (64-bit) boxes
            if length == 1:
                extended_data = f.read(8)
                if len(extended_data) < 8:
                    break
                length = extended_size_struct.unpack(extended_data)[0]
            
            # Validate box length to prevent infinite loops
            if length <= 0 or length < 8:
                logger.warning(f"Invalid box length {length} at offset {offset}")
                break
            
            # Store box location
            boxes[box_type] = (offset, offset + length)
            
            # Move to next box
            offset += length
            f.seek(offset)
            
        return boxes
    
    def parse_hilight_tags_new_format(self, f, start_offset: int, end_offset: int) -> List[int]:
        """
        Parse HiLight tags from newer GoPro format (HERO6+).
        
        This format stores highlights in the GPMF (GoPro Metadata Format) section
        and uses a more complex structure with 'Highlights' -> 'HLMT' -> 'MANL' hierarchy.
        
        Args:
            f: File object
            start_offset: Start of GPMF section
            end_offset: End of GPMF section
            
        Returns:
            List of highlight timestamps in milliseconds
        """
        highlights = []
        
        # State tracking for parsing the nested structure
        in_highlights_section = False
        in_hlmt_section = False
        
        offset = start_offset
        f.seek(offset, 0)
        
        while offset < end_offset:
            # Read 4-byte chunks to look for markers
            data = f.read(4)
            if len(data) < 4:
                break
            
            # Look for "High" + "ligh" = "Highlights" marker
            if data == b'High' and not in_highlights_section:
                next_data = f.read(4)
                if next_data == b'ligh':
                    in_highlights_section = True
                    logger.debug("Found Highlights section")
                    continue
            
            # Look for "HLMT" (HiLight Metadata) marker within Highlights section
            if data == b'HLMT' and in_highlights_section and not in_hlmt_section:
                in_hlmt_section = True
                logger.debug("Found HLMT section")
                continue
            
            # Look for "MANL" (Manual highlight) marker within HLMT section
            if data == b'MANL' and in_highlights_section and in_hlmt_section:
                # Manual highlight found - timestamp is 20 bytes back from current position
                current_pos = f.tell()
                f.seek(current_pos - 20)
                
                try:
                    timestamp_data = f.read(4)
                    if len(timestamp_data) == 4:
                        timestamp = int.from_bytes(timestamp_data, "big")
                        if timestamp > 0:  # Valid timestamp
                            highlights.append(timestamp)
                            logger.debug(f"Found HiLight at {timestamp}ms")
                except Exception as e:
                    logger.warning(f"Failed to parse timestamp: {e}")
                
                # Return to original position
                f.seek(current_pos)
                continue
            
            # Move to next byte for continued parsing
            offset = f.tell()
        
        return highlights
    
    def parse_hilight_tags_old_format(self, f, start_offset: int, end_offset: int) -> List[int]:
        """
        Parse HiLight tags from older GoPro format (before HERO6).
        
        This format stores highlights in the HMMT section as a simple sequence
        of 4-byte timestamps terminated by a zero value.
        
        Args:
            f: File object
            start_offset: Start of HMMT section
            end_offset: End of HMMT section
            
        Returns:
            List of highlight timestamps in milliseconds
        """
        highlights = []
        
        offset = start_offset
        f.seek(offset, 0)
        
        while offset < end_offset:
            # Read 4-byte timestamp
            data = f.read(4)
            if len(data) < 4:
                break
            
            try:
                timestamp = int.from_bytes(data, "big")
                if timestamp == 0:
                    # Zero timestamp indicates end of highlights list
                    break
                elif timestamp > 0:
                    highlights.append(timestamp)
                    logger.debug(f"Found HiLight at {timestamp}ms")
            except Exception as e:
                logger.warning(f"Failed to parse timestamp: {e}")
                break
            
            offset = f.tell()
        
        return highlights
    
    async def extract_hilight_tags(self, video_path: Path) -> List[float]:
        """
        Extract HiLight tags from a GoPro MP4 file.
        
        This method parses the MP4 container structure to find and extract
        HiLight tags that users created during recording.
        
        Args:
            video_path: Path to the GoPro MP4 file
            
        Returns:
            List of highlight timestamps in seconds (converted from milliseconds)
        """
        try:
            with open(video_path, "rb") as f:
                # Parse top-level MP4 boxes
                boxes = self.find_boxes(f)
                
                # Verify this is a valid MP4 file
                if b'ftyp' not in boxes or boxes[b'ftyp'][0] != 0:
                    logger.warning(f"{video_path} does not appear to be a valid MP4 file")
                    return []
                
                # Check if we have a moov box (required for metadata)
                if b'moov' not in boxes:
                    logger.warning(f"{video_path} does not contain moov box")
                    return []
                
                # Parse moov box to find user data (udta)
                moov_start, moov_end = boxes[b'moov']
                moov_boxes = self.find_boxes(f, moov_start + 8, moov_end)
                
                if b'udta' not in moov_boxes:
                    logger.debug(f"{video_path} does not contain user data section")
                    return []
                
                # Parse user data box to find GoPro metadata
                udta_start, udta_end = moov_boxes[b'udta']
                udta_boxes = self.find_boxes(f, udta_start + 8, udta_end)
                
                highlights = []
                
                # Try newer format first (HERO6+)
                if b'GPMF' in udta_boxes:
                    logger.debug(f"Found GPMF section in {video_path}")
                    gpmf_start, gpmf_end = udta_boxes[b'GPMF']
                    highlights = self.parse_hilight_tags_new_format(f, gpmf_start + 8, gpmf_end)
                
                # Fall back to older format (pre-HERO6)
                elif b'HMMT' in udta_boxes:
                    logger.debug(f"Found HMMT section in {video_path}")
                    hmmt_start, hmmt_end = udta_boxes[b'HMMT']
                    highlights = self.parse_hilight_tags_old_format(f, hmmt_start + 8, hmmt_end)
                
                else:
                    logger.debug(f"No GoPro metadata sections found in {video_path}")
                    return []
                
                # Convert from milliseconds to seconds
                highlights_seconds = [ts / 1000.0 for ts in highlights]
                
                logger.info(f"Extracted {len(highlights_seconds)} HiLight tags from {video_path}")
                return highlights_seconds
                
        except Exception as e:
            logger.error(f"Failed to extract HiLight tags from {video_path}: {e}")
            return []


class ProgressBar:
    """Modern progress bar with better formatting."""
    
    def __init__(self, total: int, width: int = 50):
        self.total = total
        self.width = width
        self.current = 0
        
    def update(self, current: int):
        self.current = current
        percentage = (current * 100) // self.total
        filled = (self.width * current) // self.total
        empty = self.width - filled
        
        bar = '=' * filled + '-' * empty
        print(f"\rProgress: [{bar}] {percentage}% ({current}/{self.total})", end='', flush=True)
        
    def finish(self):
        print()  # New line after completion


class SpinnerContext:
    """Async context manager for spinner animation during long operations."""
    
    def __init__(self, message: str):
        self.message = message
        self.spinner_chars = '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
        self.running = False
        self._spin_task = None
        
    async def __aenter__(self):
        self.running = True
        self._spin_task = asyncio.create_task(self._spin())
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self._spin_task:
            self._spin_task.cancel()
            try:
                await self._spin_task
            except asyncio.CancelledError:
                pass
        print(f"\r✓ {self.message}")
        
    async def _spin(self):
        i = 0
        try:
            while self.running:
                print(f"\r{self.spinner_chars[i % len(self.spinner_chars)]} {self.message}", end='', flush=True)
                i += 1
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            pass


class VideoProcessor:
    """Main class for processing GoPro videos and extracting highlights."""
    
    def __init__(self, config: Config):
        self.config = config
        self.hilight_extractor = GoProHiLightExtractor()
        
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
    
    async def analyze_motion_vectors(self, video_path: Path, temp_dir: Path) -> List[float]:
        """Analyze motion vectors to find high-action scenes."""
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
    
    async def process_video(self, video_path: Path, output_dir: Path) -> int:
        """Process a single video file and extract highlights."""
        logger.info(f"Processing: {video_path}")
        
        base_name = video_path.stem
        
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            
            try:
                timestamps = []
                
                # Step 1: Try to extract GoPro HiLight tags first (if enabled)
                if self.config.prefer_hilight_tags:
                    async with SpinnerContext("Extracting GoPro HiLight tags"):
                        hilight_timestamps = await self.hilight_extractor.extract_hilight_tags(video_path)
                    
                    if hilight_timestamps:
                        timestamps = hilight_timestamps
                        logger.info(f"Using {len(timestamps)} GoPro HiLight tags")
                    else:
                        logger.info("No GoPro HiLight tags found, falling back to motion analysis")
                
                # Step 2: Fall back to motion analysis if no HiLight tags or if disabled
                if not timestamps:
                    async with SpinnerContext("Analyzing motion vectors"):
                        timestamps = await self.analyze_motion_vectors(video_path, temp_dir)
                
                if not timestamps:
                    logger.warning(f"No highlights found in {video_path}")
                    return 0
                
                # Step 3: Filter timestamps
                filtered_timestamps = self._filter_timestamps(timestamps)
                
                logger.info(f"Found {len(timestamps)} potential highlights, filtered to {len(filtered_timestamps)} unique highlights")
                
                if not filtered_timestamps:
                    logger.warning(f"No valid highlights found after filtering in {video_path}")
                    return 0
                
                # Step 4: Extract clips
                print(f"🎥 Extracting {len(filtered_timestamps)} highlight clips...")
                
                progress = ProgressBar(len(filtered_timestamps))
                successful_clips = 0
                
                # Use semaphore to limit concurrent ffmpeg processes
                semaphore = asyncio.Semaphore(self.config.max_workers)
                
                async def extract_with_semaphore(timestamp: float, clip_num: int) -> bool:
                    async with semaphore:
                        output_file = output_dir / f"{base_name}_highlight_{clip_num:03d}.mp4"
                        return await self.extract_clip(video_path, timestamp, output_file)
                
                # Process clips concurrently but with limited parallelism
                tasks = []
                for i, timestamp in enumerate(filtered_timestamps, 1):
                    task = extract_with_semaphore(timestamp, i)
                    tasks.append(task)
                
                # Process in batches to show progress
                for i, task in enumerate(asyncio.as_completed(tasks), 1):
                    success = await task
                    if success:
                        successful_clips += 1
                    progress.update(i)
                
                progress.finish()
                
                logger.info(f"✓ Extracted {successful_clips} highlights from {base_name}")
                return successful_clips
                
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                return 0


async def main():
    """Main function to orchestrate the highlight extraction process."""
    parser = argparse.ArgumentParser(description='Extract highlights from GoPro videos')
    parser.add_argument('input_dir', type=Path, help='Directory containing GoPro videos')
    parser.add_argument('output_dir', type=Path, help='Directory to save highlight clips')
    parser.add_argument('--clip-duration', type=int, default=15, help='Duration of each clip in seconds')
    parser.add_argument('--motion-threshold', type=float, default=0.2, help='Motion detection threshold')
    parser.add_argument('--min-gap', type=int, default=10, help='Minimum gap between highlights in seconds')
    parser.add_argument('--max-workers', type=int, default=get_optimal_workers(), help=f'Maximum concurrent workers (default: {get_optimal_workers()} based on system)')
    parser.add_argument('--disable-hilight-tags', action='store_true', help='Disable GoPro HiLight tag extraction and use motion analysis only')
    
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
        min_gap_seconds=args.min_gap,
        max_workers=args.max_workers,
        prefer_hilight_tags=not args.disable_hilight_tags
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
        logger.info("📊 GoPro HiLight tag extraction disabled. Using motion analysis only")
    print("=" * 50)
    
    # Process videos
    processor = VideoProcessor(config)
    total_clips = 0
    
    for i, video_file in enumerate(video_files, 1):
        print(f"📹 Video {i} of {len(video_files)}: {video_file.name}")
        clips_extracted = await processor.process_video(video_file, args.output_dir)
        total_clips += clips_extracted
        print()
    
    print(f"🎉 Highlight extraction complete!")
    print(f"Total clips extracted: {total_clips}")
    print(f"All clips saved to: {args.output_dir}")


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