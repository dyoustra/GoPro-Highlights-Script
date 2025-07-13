"""
Pytest configuration and shared fixtures for GoPro highlight extraction tests.
"""

import pytest
import struct
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import asyncio


@pytest.fixture
def sample_hilight_timestamps():
    """Sample HiLight timestamps in milliseconds for testing."""
    return [5000, 15000, 30000, 45000, 60000]  # 5s, 15s, 30s, 45s, 60s


@pytest.fixture
def temp_video_file(tmp_path):
    """Create a temporary file path for testing video operations."""
    video_file = tmp_path / "test_video.mp4"
    return video_file


@pytest.fixture
def mock_video_duration():
    """Mock video duration in seconds."""
    return 120.0  # 2 minutes


@pytest.fixture
def event_loop():
    """Create an event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class MP4TestDataGenerator:
    """Generate synthetic MP4 test data for testing without real video files."""
    
    @staticmethod
    def create_box(box_type: bytes, data: bytes) -> bytes:
        """Create a valid MP4 box with given type and data."""
        if len(box_type) != 4:
            raise ValueError("Box type must be exactly 4 bytes")
        
        size = len(data) + 8  # 4 bytes size + 4 bytes type + data
        return struct.pack(">I", size) + box_type + data
    
    @staticmethod
    def create_ftyp_box() -> bytes:
        """Create a valid ftyp (file type) box."""
        major_brand = b"mp42"
        minor_version = struct.pack(">I", 0)
        compatible_brands = b"mp42" + b"isom"
        data = major_brand + minor_version + compatible_brands
        return MP4TestDataGenerator.create_box(b"ftyp", data)
    
    @staticmethod
    def create_moov_box(child_boxes: List[bytes] = None) -> bytes:
        """Create a moov (movie) container box."""
        if child_boxes is None:
            child_boxes = []
        data = b"".join(child_boxes)
        return MP4TestDataGenerator.create_box(b"moov", data)
    
    @staticmethod
    def create_udta_box(child_boxes: List[bytes] = None) -> bytes:
        """Create a udta (user data) container box."""
        if child_boxes is None:
            child_boxes = []
        data = b"".join(child_boxes)
        return MP4TestDataGenerator.create_box(b"udta", data)
    
    @staticmethod
    def create_gpmf_box(gpmf_data: bytes) -> bytes:
        """Create a GPMF (GoPro metadata format) box."""
        return MP4TestDataGenerator.create_box(b"GPMF", gpmf_data)
    
    @staticmethod
    def create_hmmt_box(timestamps: List[int]) -> bytes:
        """Create an HMMT (HiLight metadata) box with timestamps."""
        # HMMT format: sequence of 4-byte big-endian timestamps, terminated by 0
        data = b""
        for timestamp in timestamps:
            data += struct.pack(">I", timestamp)
        data += struct.pack(">I", 0)  # Terminator
        return MP4TestDataGenerator.create_box(b"HMMT", data)
    
    @staticmethod
    def create_gpmf_hilight_data(timestamps: List[int]) -> bytes:
        """Create synthetic GPMF data with HiLight tags."""
        # Simplified GPMF structure with Highlights -> HLMT -> MANL
        data = b""
        
        # Add some initial padding to make parsing more realistic
        data += b"\x00" * 4
        
        # Add "High" + "ligh" markers consecutively (as expected by parsing logic)
        data += b"High" + b"ligh"
        
        # Add some padding/structure data
        data += b"\x00" * 8
        
        # Add "HLMT" marker
        data += b"HLMT"
        
        # Add some padding/structure data
        data += b"\x00" * 8
        
        # Add timestamps with "MANL" markers
        for timestamp in timestamps:
            # Add some structure before timestamp
            data += b"\x00" * 16
            # Add timestamp (exactly 20 bytes back from current position after reading MANL)
            data += struct.pack(">I", timestamp)
            data += b"\x00" * 12  # 12 bytes padding after timestamp (20 - 4 for MANL - 4 for timestamp = 12)
            # Add "MANL" marker - this should be exactly 20 bytes from current position after reading
            data += b"MANL"
        
        return data
    
    @staticmethod
    def create_complete_mp4_with_hilights(timestamps: List[int], use_new_format: bool = True) -> bytes:
        """Create a complete MP4 structure with HiLight tags."""
        ftyp_box = MP4TestDataGenerator.create_ftyp_box()
        
        if use_new_format:
            # HERO6+ format with GPMF
            gpmf_data = MP4TestDataGenerator.create_gpmf_hilight_data(timestamps)
            gpmf_box = MP4TestDataGenerator.create_gpmf_box(gpmf_data)
            udta_box = MP4TestDataGenerator.create_udta_box([gpmf_box])
        else:
            # Pre-HERO6 format with HMMT
            hmmt_box = MP4TestDataGenerator.create_hmmt_box(timestamps)
            udta_box = MP4TestDataGenerator.create_udta_box([hmmt_box])
        
        moov_box = MP4TestDataGenerator.create_moov_box([udta_box])
        
        # Create a minimal mdat box
        mdat_data = b"fake_video_data" * 100  # Some fake video data
        mdat_box = MP4TestDataGenerator.create_box(b"mdat", mdat_data)
        
        return ftyp_box + moov_box + mdat_box
    
    @staticmethod
    def create_mp4_without_hilights() -> bytes:
        """Create an MP4 file without any HiLight tags."""
        ftyp_box = MP4TestDataGenerator.create_ftyp_box()
        
        # Empty udta box (no GPMF or HMMT)
        udta_box = MP4TestDataGenerator.create_udta_box([])
        moov_box = MP4TestDataGenerator.create_moov_box([udta_box])
        
        # Create a minimal mdat box
        mdat_data = b"fake_video_data" * 100
        mdat_box = MP4TestDataGenerator.create_box(b"mdat", mdat_data)
        
        return ftyp_box + moov_box + mdat_box
    
    @staticmethod
    def create_invalid_mp4() -> bytes:
        """Create an invalid MP4 file for error testing."""
        return b"This is not a valid MP4 file"
    
    @staticmethod
    def create_mp4_without_moov() -> bytes:
        """Create an MP4 file without a moov box."""
        ftyp_box = MP4TestDataGenerator.create_ftyp_box()
        mdat_data = b"fake_video_data" * 100
        mdat_box = MP4TestDataGenerator.create_box(b"mdat", mdat_data)
        return ftyp_box + mdat_box


@pytest.fixture
def mp4_generator():
    """Provide the MP4 test data generator."""
    return MP4TestDataGenerator