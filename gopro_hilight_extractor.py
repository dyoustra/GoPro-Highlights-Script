#!/usr/bin/env python3
"""
GoPro HiLight tag extraction from MP4 files.
"""

import logging
import struct
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


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