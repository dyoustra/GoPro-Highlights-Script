"""
Unit tests for MP4 box parsing functionality in GoPro highlight extraction.

These tests focus on the core MP4 parsing logic without requiring actual video files.
"""

import pytest
import struct
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open
import io

# Import the classes we want to test
from extract_highlights import GoProHiLightExtractor


class TestMP4BoxParsing:
    """Test the basic MP4 box parsing functionality."""
    
    def test_find_boxes_empty_file(self):
        """Test find_boxes with an empty file."""
        extractor = GoProHiLightExtractor()
        
        with io.BytesIO(b"") as f:
            boxes = extractor.find_boxes(f)
            assert boxes == {}
    
    def test_find_boxes_single_box(self, mp4_generator):
        """Test find_boxes with a single valid box."""
        extractor = GoProHiLightExtractor()
        
        # Create a single ftyp box
        ftyp_box = mp4_generator.create_ftyp_box()
        
        with io.BytesIO(ftyp_box) as f:
            boxes = extractor.find_boxes(f)
            
            assert b'ftyp' in boxes
            start, end = boxes[b'ftyp']
            assert start == 0
            assert end == len(ftyp_box)
    
    def test_find_boxes_multiple_boxes(self, mp4_generator):
        """Test find_boxes with multiple boxes."""
        extractor = GoProHiLightExtractor()
        
        # Create multiple boxes
        ftyp_box = mp4_generator.create_ftyp_box()
        moov_box = mp4_generator.create_moov_box()
        
        combined_data = ftyp_box + moov_box
        
        with io.BytesIO(combined_data) as f:
            boxes = extractor.find_boxes(f)
            
            assert b'ftyp' in boxes
            assert b'moov' in boxes
            
            # Check positions
            ftyp_start, ftyp_end = boxes[b'ftyp']
            moov_start, moov_end = boxes[b'moov']
            
            assert ftyp_start == 0
            assert ftyp_end == len(ftyp_box)
            assert moov_start == len(ftyp_box)
            assert moov_end == len(combined_data)
    
    def test_find_boxes_with_offset_and_limit(self, mp4_generator):
        """Test find_boxes with start and end offsets."""
        extractor = GoProHiLightExtractor()
        
        # Create multiple boxes
        ftyp_box = mp4_generator.create_ftyp_box()
        moov_box = mp4_generator.create_moov_box()
        mdat_box = mp4_generator.create_box(b"mdat", b"fake_data" * 50)
        
        combined_data = ftyp_box + moov_box + mdat_box
        
        with io.BytesIO(combined_data) as f:
            # Only parse up to the end of moov box
            end_offset = len(ftyp_box) + len(moov_box)
            boxes = extractor.find_boxes(f, start_offset=0, end_offset=end_offset)
            
            assert b'ftyp' in boxes
            assert b'moov' in boxes
            assert b'mdat' not in boxes  # Should not be parsed due to end_offset
    
    def test_find_boxes_malformed_data(self):
        """Test find_boxes with malformed data."""
        extractor = GoProHiLightExtractor()
        
        # Create malformed data (less than 8 bytes)
        malformed_data = b"short"
        
        with io.BytesIO(malformed_data) as f:
            boxes = extractor.find_boxes(f)
            assert boxes == {}
    
    def test_find_boxes_invalid_box_size(self):
        """Test find_boxes with invalid box size."""
        extractor = GoProHiLightExtractor()
        
        # Create a box with size 0 (invalid)
        invalid_box = struct.pack(">I", 0) + b"test" + b"data"
        
        with io.BytesIO(invalid_box) as f:
            boxes = extractor.find_boxes(f)
            # Should handle gracefully and return empty dict
            assert boxes == {}


class TestHiLightTagParsingOldFormat:
    """Test HiLight tag parsing for older GoPro format (pre-HERO6)."""
    
    def test_parse_hilight_tags_old_format_valid_data(self, sample_hilight_timestamps):
        """Test parsing valid HMMT data."""
        extractor = GoProHiLightExtractor()
        
        # Create HMMT data with timestamps
        data = b""
        for timestamp in sample_hilight_timestamps:
            data += struct.pack(">I", timestamp)
        data += struct.pack(">I", 0)  # Terminator
        
        with io.BytesIO(data) as f:
            highlights = extractor.parse_hilight_tags_old_format(f, 0, len(data))
            
            assert highlights == sample_hilight_timestamps
    
    def test_parse_hilight_tags_old_format_empty_data(self):
        """Test parsing empty HMMT data."""
        extractor = GoProHiLightExtractor()
        
        # Only terminator
        data = struct.pack(">I", 0)
        
        with io.BytesIO(data) as f:
            highlights = extractor.parse_hilight_tags_old_format(f, 0, len(data))
            
            assert highlights == []
    
    def test_parse_hilight_tags_old_format_single_timestamp(self):
        """Test parsing single timestamp."""
        extractor = GoProHiLightExtractor()
        
        timestamp = 12345
        data = struct.pack(">I", timestamp) + struct.pack(">I", 0)
        
        with io.BytesIO(data) as f:
            highlights = extractor.parse_hilight_tags_old_format(f, 0, len(data))
            
            assert highlights == [timestamp]
    
    def test_parse_hilight_tags_old_format_truncated_data(self):
        """Test parsing truncated HMMT data."""
        extractor = GoProHiLightExtractor()
        
        # Incomplete timestamp (only 3 bytes instead of 4)
        data = b"\x00\x00\x01"
        
        with io.BytesIO(data) as f:
            highlights = extractor.parse_hilight_tags_old_format(f, 0, len(data))
            
            # Should handle gracefully and return empty list
            assert highlights == []


class TestHiLightTagParsingNewFormat:
    """Test HiLight tag parsing for newer GoPro format (HERO6+)."""
    
    def test_parse_hilight_tags_new_format_valid_data(self, sample_hilight_timestamps, mp4_generator):
        """Test parsing valid GPMF data with HiLight tags."""
        extractor = GoProHiLightExtractor()
        
        # Create GPMF structure with highlights using the generator
        data = mp4_generator.create_gpmf_hilight_data(sample_hilight_timestamps)
        
        with io.BytesIO(data) as f:
            highlights = extractor.parse_hilight_tags_new_format(f, 0, len(data))
            
            assert highlights == sample_hilight_timestamps
    
    def test_parse_hilight_tags_new_format_no_highlights_section(self):
        """Test parsing GPMF data without Highlights section."""
        extractor = GoProHiLightExtractor()
        
        # Data without "Highlights" marker
        data = b"GPMF" + b"DATA" + b"\x00" * 100
        
        with io.BytesIO(data) as f:
            highlights = extractor.parse_hilight_tags_new_format(f, 0, len(data))
            
            assert highlights == []
    
    def test_parse_hilight_tags_new_format_no_hlmt_section(self):
        """Test parsing GPMF data with Highlights but no HLMT section."""
        extractor = GoProHiLightExtractor()
        
        # Data with "Highlights" but no "HLMT"
        data = b"High" + b"ligh" + b"\x00" * 100
        
        with io.BytesIO(data) as f:
            highlights = extractor.parse_hilight_tags_new_format(f, 0, len(data))
            
            assert highlights == []
    
    def test_parse_hilight_tags_new_format_no_manl_markers(self):
        """Test parsing GPMF data with HLMT but no MANL markers."""
        extractor = GoProHiLightExtractor()
        
        # Data with "Highlights" and "HLMT" but no "MANL"
        data = b"High" + b"ligh" + b"\x00" * 8 + b"HLMT" + b"\x00" * 100
        
        with io.BytesIO(data) as f:
            highlights = extractor.parse_hilight_tags_new_format(f, 0, len(data))
            
            assert highlights == []
    
    def test_parse_hilight_tags_new_format_empty_data(self):
        """Test parsing empty GPMF data."""
        extractor = GoProHiLightExtractor()
        
        with io.BytesIO(b"") as f:
            highlights = extractor.parse_hilight_tags_new_format(f, 0, 0)
            
            assert highlights == []


class TestExtractHiLightTags:
    """Test the main extract_hilight_tags method."""
    
    @pytest.mark.asyncio
    async def test_extract_hilight_tags_new_format(self, tmp_path, mp4_generator, sample_hilight_timestamps):
        """Test extracting HiLight tags from new format MP4."""
        extractor = GoProHiLightExtractor()
        
        # Create MP4 with new format HiLight tags
        mp4_data = mp4_generator.create_complete_mp4_with_hilights(
            sample_hilight_timestamps, use_new_format=True
        )
        
        # Write to temporary file
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(mp4_data)
        
        # Extract highlights
        highlights = await extractor.extract_hilight_tags(video_file)
        
        # Convert from milliseconds to seconds
        expected_highlights = [ts / 1000.0 for ts in sample_hilight_timestamps]
        assert highlights == expected_highlights
    
    @pytest.mark.asyncio
    async def test_extract_hilight_tags_old_format(self, tmp_path, mp4_generator, sample_hilight_timestamps):
        """Test extracting HiLight tags from old format MP4."""
        extractor = GoProHiLightExtractor()
        
        # Create MP4 with old format HiLight tags
        mp4_data = mp4_generator.create_complete_mp4_with_hilights(
            sample_hilight_timestamps, use_new_format=False
        )
        
        # Write to temporary file
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(mp4_data)
        
        # Extract highlights
        highlights = await extractor.extract_hilight_tags(video_file)
        
        # Convert from milliseconds to seconds
        expected_highlights = [ts / 1000.0 for ts in sample_hilight_timestamps]
        assert highlights == expected_highlights
    
    @pytest.mark.asyncio
    async def test_extract_hilight_tags_no_highlights(self, tmp_path, mp4_generator):
        """Test extracting HiLight tags from MP4 without highlights."""
        extractor = GoProHiLightExtractor()
        
        # Create MP4 without HiLight tags
        mp4_data = mp4_generator.create_mp4_without_hilights()
        
        # Write to temporary file
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(mp4_data)
        
        # Extract highlights
        highlights = await extractor.extract_hilight_tags(video_file)
        
        assert highlights == []
    
    @pytest.mark.asyncio
    async def test_extract_hilight_tags_invalid_mp4(self, tmp_path, mp4_generator):
        """Test extracting HiLight tags from invalid MP4."""
        extractor = GoProHiLightExtractor()
        
        # Create invalid MP4
        mp4_data = mp4_generator.create_invalid_mp4()
        
        # Write to temporary file
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(mp4_data)
        
        # Extract highlights
        highlights = await extractor.extract_hilight_tags(video_file)
        
        assert highlights == []
    
    @pytest.mark.asyncio
    async def test_extract_hilight_tags_no_moov_box(self, tmp_path, mp4_generator):
        """Test extracting HiLight tags from MP4 without moov box."""
        extractor = GoProHiLightExtractor()
        
        # Create MP4 without moov box
        mp4_data = mp4_generator.create_mp4_without_moov()
        
        # Write to temporary file
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(mp4_data)
        
        # Extract highlights
        highlights = await extractor.extract_hilight_tags(video_file)
        
        assert highlights == []
    
    @pytest.mark.asyncio
    async def test_extract_hilight_tags_file_not_found(self):
        """Test extracting HiLight tags from non-existent file."""
        extractor = GoProHiLightExtractor()
        
        # Try to extract from non-existent file
        non_existent_file = Path("/path/that/does/not/exist.mp4")
        
        # Should handle gracefully and return empty list
        highlights = await extractor.extract_hilight_tags(non_existent_file)
        
        assert highlights == []


class TestMP4BoxParsingEdgeCases:
    """Test edge cases and error conditions in MP4 box parsing."""
    
    def test_find_boxes_uuid_box_handling(self):
        """Test handling of UUID boxes (should be skipped)."""
        extractor = GoProHiLightExtractor()
        
        # Create a UUID box (16 bytes type instead of 4)
        uuid_box = struct.pack(">I", 24) + b"uuid" + b"0123456789abcdef"
        
        with io.BytesIO(uuid_box) as f:
            boxes = extractor.find_boxes(f)
            
            # UUID boxes should be skipped
            assert b'uuid' not in boxes
    
    def test_find_boxes_extended_size(self):
        """Test handling of extended size boxes."""
        extractor = GoProHiLightExtractor()
        
        # Create a box with extended size (size = 1 indicates 64-bit size follows)
        data = b"test_data_for_extended_size_box"
        extended_size = len(data) + 16  # 4 + 4 + 8 + data
        
        box = struct.pack(">I", 1) + b"test" + struct.pack(">Q", extended_size) + data
        
        with io.BytesIO(box) as f:
            boxes = extractor.find_boxes(f)
            
            assert b'test' in boxes
            start, end = boxes[b'test']
            assert start == 0
            assert end == extended_size
    
    def test_find_boxes_partial_extended_size(self):
        """Test handling of truncated extended size data."""
        extractor = GoProHiLightExtractor()
        
        # Create a box that indicates extended size but doesn't have enough data
        incomplete_box = struct.pack(">I", 1) + b"test" + b"1234"  # Only 4 bytes instead of 8
        
        with io.BytesIO(incomplete_box) as f:
            boxes = extractor.find_boxes(f)
            
            # Should handle gracefully
            assert boxes == {}
    
    def test_parse_hilight_tags_with_exception_handling(self):
        """Test that parsing handles exceptions gracefully."""
        extractor = GoProHiLightExtractor()
        
        # Create data that might cause parsing issues
        problematic_data = b"\xff" * 1000  # All 0xFF bytes
        
        with io.BytesIO(problematic_data) as f:
            # Both parsing methods should handle this gracefully
            highlights_old = extractor.parse_hilight_tags_old_format(f, 0, len(problematic_data))
            highlights_new = extractor.parse_hilight_tags_new_format(f, 0, len(problematic_data))
            
            # Should return empty lists, not crash
            assert isinstance(highlights_old, list)
            assert isinstance(highlights_new, list)


class TestMP4GeneratorValidation:
    """Test that our MP4 test data generator creates valid structures."""
    
    def test_create_box_basic(self, mp4_generator):
        """Test basic box creation."""
        box_type = b"test"
        data = b"hello world"
        
        box = mp4_generator.create_box(box_type, data)
        
        # Check structure: size (4 bytes) + type (4 bytes) + data
        expected_size = 4 + 4 + len(data)
        assert len(box) == expected_size
        
        # Check size field
        size = struct.unpack(">I", box[:4])[0]
        assert size == expected_size
        
        # Check type field
        assert box[4:8] == box_type
        
        # Check data field
        assert box[8:] == data
    
    def test_create_box_invalid_type(self, mp4_generator):
        """Test that invalid box types raise errors."""
        with pytest.raises(ValueError, match="Box type must be exactly 4 bytes"):
            mp4_generator.create_box(b"toolong", b"data")
        
        with pytest.raises(ValueError, match="Box type must be exactly 4 bytes"):
            mp4_generator.create_box(b"sht", b"data")
    
    def test_ftyp_box_structure(self, mp4_generator):
        """Test that ftyp box has correct structure."""
        ftyp_box = mp4_generator.create_ftyp_box()
        
        # Should start with size and 'ftyp'
        size = struct.unpack(">I", ftyp_box[:4])[0]
        box_type = ftyp_box[4:8]
        
        assert box_type == b"ftyp"
        assert size == len(ftyp_box)
        
        # Check ftyp content structure
        major_brand = ftyp_box[8:12]
        minor_version = struct.unpack(">I", ftyp_box[12:16])[0]
        
        assert major_brand == b"mp42"
        assert minor_version == 0
    
    def test_complete_mp4_structure(self, mp4_generator, sample_hilight_timestamps):
        """Test that complete MP4 structure is valid."""
        mp4_data = mp4_generator.create_complete_mp4_with_hilights(sample_hilight_timestamps)
        
        # Should start with ftyp box
        box_type = mp4_data[4:8]
        assert box_type == b"ftyp"
        
        # Should be large enough to contain all components
        assert len(mp4_data) > 100  # Reasonable minimum size
        
        # Should be parseable by our extractor
        extractor = GoProHiLightExtractor()
        with io.BytesIO(mp4_data) as f:
            boxes = extractor.find_boxes(f)
            
            # Should contain expected top-level boxes
            assert b'ftyp' in boxes
            assert b'moov' in boxes
            assert b'mdat' in boxes 