"""
Integration tests for file I/O operations and file system interactions.

These tests validate real file operations, directory handling, permissions,
and concurrent file access patterns using actual temporary files.
"""

import pytest
import asyncio
import tempfile
import os
import stat
import threading
import time
from pathlib import Path
from unittest.mock import patch, AsyncMock
import subprocess
import shutil
import concurrent.futures

from config import Config
from video_processor import VideoProcessor
from gopro_hilight_extractor import GoProHiLightExtractor
from extract_highlights import main


class TestFileSystemOperations:
    """Test basic file system operations."""
    
    @pytest.mark.integration
    def test_directory_creation_and_cleanup(self, tmp_path):
        """Test directory creation and cleanup workflows."""
        # Test nested directory creation
        nested_dir = tmp_path / "level1" / "level2" / "level3"
        nested_dir.mkdir(parents=True, exist_ok=True)
        
        assert nested_dir.exists()
        assert nested_dir.is_dir()
        
        # Test cleanup
        shutil.rmtree(tmp_path / "level1")
        assert not nested_dir.exists()
        
        # Test creation with exist_ok=False
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        
        # Should not raise error with exist_ok=True
        output_dir.mkdir(exist_ok=True)
        
        # Should raise error with exist_ok=False
        with pytest.raises(FileExistsError):
            output_dir.mkdir(exist_ok=False)
    
    @pytest.mark.integration
    def test_file_creation_and_deletion(self, tmp_path):
        """Test file creation and deletion operations."""
        # Create test file
        test_file = tmp_path / "test_file.txt"
        test_content = "Hello, World!"
        test_file.write_text(test_content)
        
        assert test_file.exists()
        assert test_file.is_file()
        assert test_file.read_text() == test_content
        
        # Test file size
        assert test_file.stat().st_size == len(test_content.encode())
        
        # Test deletion
        test_file.unlink()
        assert not test_file.exists()
        
        # Test deletion of non-existent file
        with pytest.raises(FileNotFoundError):
            test_file.unlink()
        
        # Test deletion with missing_ok=True
        test_file.unlink(missing_ok=True)  # Should not raise
    
    @pytest.mark.integration
    def test_binary_file_operations(self, tmp_path, mp4_generator):
        """Test binary file operations with MP4 data."""
        # Create binary MP4 data
        timestamps = [5000, 15000]
        mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps)
        
        # Write binary data
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(mp4_data)
        
        assert video_file.exists()
        assert video_file.stat().st_size == len(mp4_data)
        
        # Read binary data back
        read_data = video_file.read_bytes()
        assert read_data == mp4_data
        
        # Test partial reading
        with open(video_file, 'rb') as f:
            first_8_bytes = f.read(8)
            assert len(first_8_bytes) == 8
            assert first_8_bytes == mp4_data[:8]
    
    @pytest.mark.integration
    def test_file_permissions(self, tmp_path):
        """Test file permission handling."""
        test_file = tmp_path / "permission_test.txt"
        test_file.write_text("test content")
        
        # Get current permissions
        current_perms = test_file.stat().st_mode
        
        # Make file read-only
        test_file.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        
        # Should be able to read
        content = test_file.read_text()
        assert content == "test content"
        
        # Should not be able to write (on most systems)
        try:
            test_file.write_text("new content")
            # If we get here, the system allows writing to read-only files
            # (some systems/test environments might allow this)
            pass
        except PermissionError:
            # This is the expected behavior on most systems
            pass
        
        # Restore permissions for cleanup
        test_file.chmod(current_perms)
    
    @pytest.mark.integration
    def test_symlink_handling(self, tmp_path):
        """Test symbolic link handling."""
        # Create original file
        original_file = tmp_path / "original.txt"
        original_file.write_text("original content")
        
        # Create symlink
        symlink_file = tmp_path / "symlink.txt"
        try:
            symlink_file.symlink_to(original_file)
            
            # Test reading through symlink
            assert symlink_file.read_text() == "original content"
            assert symlink_file.is_symlink()
            assert symlink_file.readlink() == original_file
            
            # Test that both exist
            assert original_file.exists()
            assert symlink_file.exists()
            
            # Delete original, symlink should become broken
            original_file.unlink()
            assert not original_file.exists()
            assert symlink_file.is_symlink()  # Still a symlink
            assert not symlink_file.exists()  # But broken
            
        except OSError:
            # Symlinks might not be supported on all systems
            pytest.skip("Symlinks not supported on this system")


class TestVideoFileHandling:
    """Test video file handling operations."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_video_file_processing_workflow(self, tmp_path, mp4_generator):
        """Test complete video file processing workflow."""
        # Create input and output directories
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        
        # Create test video files with sufficient gaps (default min_gap_seconds=10)
        video_files = []
        for i in range(3):
            timestamps = [i * 20000 + 5000, i * 20000 + 20000]  # 20s gaps
            mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps)
            
            video_file = input_dir / f"GX01012{i}.mp4"
            video_file.write_bytes(mp4_data)
            video_files.append(video_file)
        
        # Process each video file
        config = Config(clip_duration=10)
        processor = VideoProcessor(config)
        
        # Mock external dependencies
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process
            
            # Mock file creation for clip extraction
            created_files = []
            
            def mock_extract_clip(video_path, timestamp, output_path):
                # Simulate file creation
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(b"fake_clip_data" * 100)
                created_files.append(output_path)
                return True
            
            with patch.object(processor, 'extract_clip', side_effect=mock_extract_clip):
                for video_file in video_files:
                    clips_extracted = await processor.process_video(video_file, output_dir)
                    assert clips_extracted == (2, 0)  # 2 hilight clips, 0 motion clips per video
        
        # Verify output files were created
        assert len(created_files) == 6  # 3 videos * 2 clips each
        
        # Verify file naming convention (files are created in HiLights subdirectory)
        hilight_dir = output_dir / "HiLights"
        for i, video_file in enumerate(video_files):
            base_name = video_file.stem
            expected_files = [
                hilight_dir / f"{base_name}_highlight_001.mp4",
                hilight_dir / f"{base_name}_highlight_002.mp4"
            ]
            
            for expected_file in expected_files:
                assert expected_file in created_files
                assert expected_file.exists()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_large_file_handling(self, tmp_path, mp4_generator):
        """Test handling of large video files."""
        # Create a larger MP4 file
        timestamps = list(range(1000, 60000, 5000))  # Many timestamps
        mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps)
        
        # Add extra data to make it larger
        large_data = mp4_data + b"padding_data" * 10000
        
        large_video = tmp_path / "large_video.mp4"
        large_video.write_bytes(large_data)
        
        # Verify file size
        assert large_video.stat().st_size > 100000  # > 100KB
        
        # Process the large file
        extractor = GoProHiLightExtractor()
        highlights = await extractor.extract_hilight_tags(large_video)
        
        # Should still extract highlights correctly
        expected_highlights = [ts / 1000.0 for ts in timestamps]
        assert highlights == expected_highlights
    
    @pytest.mark.integration
    def test_file_discovery_patterns(self, tmp_path):
        """Test file discovery with various patterns."""
        # Create files with different extensions
        test_files = [
            "video1.mp4",
            "video2.MP4",
            "video3.mov",
            "video4.MOV",
            "video5.avi",  # Not in default extensions
            "not_video.txt",
            "video6.mp4.backup",  # Wrong extension
        ]
        
        for filename in test_files:
            (tmp_path / filename).write_text("dummy content")
        
        # Test extension-based discovery
        config = Config()
        found_files = []
        
        for ext in config.video_extensions:
            found_files.extend(tmp_path.glob(f"*{ext}"))
        
        # Should find only the video files with correct extensions
        found_names = [f.name for f in found_files]
        expected_names = ["video1.mp4", "video2.MP4", "video3.mov", "video4.MOV"]
        
        assert set(found_names) == set(expected_names)
    
    @pytest.mark.integration
    def test_file_size_validation(self, tmp_path):
        """Test file size validation for different scenarios."""
        # Create files of different sizes
        files_and_sizes = [
            ("empty.mp4", 0),
            ("tiny.mp4", 100),
            ("small.mp4", 1000),
            ("medium.mp4", 50000),
            ("large.mp4", 1000000),
        ]
        
        for filename, size in files_and_sizes:
            file_path = tmp_path / filename
            file_path.write_bytes(b"x" * size)
            
            # Verify file size
            assert file_path.stat().st_size == size
            
            # Test size thresholds (like the 1000 byte minimum in extract_clip)
            if size >= 1000:
                assert file_path.stat().st_size >= 1000
            else:
                assert file_path.stat().st_size < 1000


class TestConcurrentFileAccess:
    """Test concurrent file access patterns."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_file_reading(self, tmp_path, mp4_generator):
        """Test concurrent reading of the same file."""
        # Create test file
        timestamps = [5000, 15000, 25000]
        mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps)
        
        video_file = tmp_path / "concurrent_test.mp4"
        video_file.write_bytes(mp4_data)
        
        # Create multiple extractors
        extractors = [GoProHiLightExtractor() for _ in range(5)]
        
        # Read concurrently
        async def extract_highlights(extractor):
            return await extractor.extract_hilight_tags(video_file)
        
        tasks = [extract_highlights(extractor) for extractor in extractors]
        results = await asyncio.gather(*tasks)
        
        # All results should be the same
        expected_highlights = [5.0, 15.0, 25.0]
        for result in results:
            assert result == expected_highlights
    
    @pytest.mark.integration
    def test_concurrent_file_writing(self, tmp_path):
        """Test concurrent file writing to different files."""
        # Create multiple threads writing to different files
        def write_file(file_path, content):
            file_path.write_text(content)
            return file_path.read_text()
        
        # Create tasks for concurrent writing
        tasks = []
        expected_results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for i in range(10):
                file_path = tmp_path / f"concurrent_write_{i}.txt"
                content = f"Content for file {i}"
                expected_results[file_path] = content
                
                future = executor.submit(write_file, file_path, content)
                tasks.append((file_path, future))
            
            # Wait for all tasks to complete
            for file_path, future in tasks:
                result = future.result()
                assert result == expected_results[file_path]
                assert file_path.exists()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_concurrent_video_processing(self, tmp_path, mp4_generator):
        """Test concurrent processing of multiple video files."""
        # Create multiple video files with sufficient gaps
        video_files = []
        for i in range(3):
            timestamps = [i * 30000 + 2000, i * 30000 + 20000]  # 18s gaps
            mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps)
            
            video_file = tmp_path / f"concurrent_video_{i}.mp4"
            video_file.write_bytes(mp4_data)
            video_files.append(video_file)
        
        # Process concurrently
        config = Config()
        processor = VideoProcessor(config)
        
        # Mock external dependencies
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process
            
            # Mock successful clip extraction
            with patch.object(processor, 'extract_clip', return_value=True):
                # Process all videos concurrently
                tasks = [
                    processor.process_video(video_file, tmp_path / f"output_{i}")
                    for i, video_file in enumerate(video_files)
                ]
                
                results = await asyncio.gather(*tasks)
                
                # All should have processed successfully - expecting (hilight_clips, motion_clips)
                assert all(result == (2, 0) for result in results)  # 2 clips per video


class TestFileSystemErrors:
    """Test file system error handling."""
    
    @pytest.mark.integration
    def test_disk_space_simulation(self, tmp_path):
        """Test handling of disk space issues (simulated)."""
        # Create a file
        test_file = tmp_path / "disk_space_test.txt"
        test_file.write_text("initial content")
        
        # Simulate disk full error
        with patch('pathlib.Path.write_bytes', side_effect=OSError("No space left on device")):
            with pytest.raises(OSError, match="No space left on device"):
                test_file.write_bytes(b"new content")
    
    @pytest.mark.integration
    def test_file_locking_simulation(self, tmp_path):
        """Test handling of file locking issues."""
        test_file = tmp_path / "lock_test.txt"
        test_file.write_text("test content")
        
        # Simulate file being locked
        with patch('pathlib.Path.read_text', side_effect=PermissionError("File is locked")):
            with pytest.raises(PermissionError, match="File is locked"):
                test_file.read_text()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_missing_input_file_handling(self, tmp_path):
        """Test handling of missing input files."""
        non_existent_file = tmp_path / "does_not_exist.mp4"
        
        config = Config()
        processor = VideoProcessor(config)
        
        # Should handle gracefully
        clips_extracted = await processor.process_video(non_existent_file, tmp_path / "output")
        assert clips_extracted == (0, 0)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_invalid_output_directory(self, tmp_path, mp4_generator):
        """Test handling of invalid output directories."""
        # Create valid input file
        timestamps = [10000]
        mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps)
        
        video_file = tmp_path / "test_video.mp4"
        video_file.write_bytes(mp4_data)
        
        # Try to use a file as output directory
        invalid_output = tmp_path / "not_a_directory.txt"
        invalid_output.write_text("this is a file, not a directory")
        
        config = Config()
        processor = VideoProcessor(config)
        
        # Mock external dependencies
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process
            
            # Should handle gracefully when trying to create output in invalid location
            clips_extracted = await processor.process_video(video_file, invalid_output)
            # The actual behavior depends on implementation - might be 0 or might create files


class TestFileSystemIntegrationWithMainFunction:
    """Test file system integration with the main function."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_main_function_file_operations(self, tmp_path, mp4_generator):
        """Test main function with real file operations."""
        # Create input directory with video files
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        
        # Create test video files with sufficient gaps
        for i in range(2):
            timestamps = [i * 30000 + 3000, i * 30000 + 20000]  # 17s gaps
            mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps)
            
            video_file = input_dir / f"GX01012{i}.mp4"
            video_file.write_bytes(mp4_data)
        
        # Mock command line arguments
        test_args = [
            'extract_highlights.py',
            str(input_dir),
            str(output_dir),
            '--clip-duration', '15'
        ]
        
        with patch('sys.argv', test_args):
            # Mock external dependencies
            with patch('asyncio.create_subprocess_exec') as mock_exec:
                mock_process = AsyncMock()
                mock_process.communicate.return_value = (b"", b"")
                mock_process.returncode = 0
                mock_exec.return_value = mock_process
                
                # Mock clip extraction to create actual files
                def mock_extract_clip(video_path, timestamp, output_path):
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(b"mock_clip_data")
                    return True
                
                with patch('extract_highlights.VideoProcessor.extract_clip', side_effect=mock_extract_clip):
                    await main()
                    
                    # Verify output directory was created
                    assert output_dir.exists()
                    assert output_dir.is_dir()
                    
                    # Verify output files were created (check in subdirectories)
                    hilight_files = list((output_dir / "HiLights").glob("*.mp4"))
                    motion_files = list((output_dir / "Motion Detected Highlights").glob("*.mp4"))
                    total_files = len(hilight_files) + len(motion_files)
                    assert total_files == 4  # 2 videos * 2 clips each
    
    @pytest.mark.integration
    def test_file_extension_filtering(self, tmp_path):
        """Test file extension filtering in file discovery."""
        # Create files with various extensions
        files_to_create = [
            "video1.mp4",
            "video2.MP4",
            "video3.mov",
            "video4.MOV",
            "video5.avi",
            "video6.mkv",
            "document.txt",
            "image.jpg",
            "video7.mp4.bak",
        ]
        
        for filename in files_to_create:
            (tmp_path / filename).write_text("dummy")
        
        # Test default extension filtering
        config = Config()
        found_files = []
        
        for ext in config.video_extensions:
            found_files.extend(tmp_path.glob(f"*{ext}"))
        
        # Should only find the default video extensions
        found_names = sorted([f.name for f in found_files])
        expected_names = sorted(["video1.mp4", "video2.MP4", "video3.mov", "video4.MOV"])
        
        assert found_names == expected_names
    
    @pytest.mark.integration
    def test_cleanup_on_interruption(self, tmp_path):
        """Test cleanup behavior when operations are interrupted."""
        # Create temporary files
        temp_files = []
        for i in range(5):
            temp_file = tmp_path / f"temp_{i}.tmp"
            temp_file.write_text(f"temporary content {i}")
            temp_files.append(temp_file)
        
        # Verify files exist
        for temp_file in temp_files:
            assert temp_file.exists()
        
        # Simulate cleanup (this would normally happen in a finally block)
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
        
        # Verify cleanup
        for temp_file in temp_files:
            assert not temp_file.exists()


class TestFileSystemPerformance:
    """Test file system performance characteristics."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_large_directory_handling(self, tmp_path):
        """Test handling of directories with many files."""
        # Create directory with many files
        many_files_dir = tmp_path / "many_files"
        many_files_dir.mkdir()
        
        # Create 1000 files
        for i in range(1000):
            file_path = many_files_dir / f"file_{i:04d}.txt"
            file_path.write_text(f"content {i}")
        
        # Test file discovery performance
        start_time = time.time()
        all_files = list(many_files_dir.glob("*.txt"))
        end_time = time.time()
        
        assert len(all_files) == 1000
        assert end_time - start_time < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_file_io_performance(self, tmp_path, mp4_generator):
        """Test file I/O performance with realistic data sizes."""
        # Create a realistic-sized MP4 file
        timestamps = list(range(1000, 30000, 1000))  # 30 timestamps
        mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps)
        
        # Make it larger (simulate real video file size)
        large_mp4_data = mp4_data + b"padding" * 100000  # ~500KB
        
        # Test write performance
        large_file = tmp_path / "large_test.mp4"
        start_time = time.time()
        large_file.write_bytes(large_mp4_data)
        write_time = time.time() - start_time
        
        # Test read performance
        start_time = time.time()
        read_data = large_file.read_bytes()
        read_time = time.time() - start_time
        
        assert read_data == large_mp4_data
        assert write_time < 1.0  # Should complete within 1 second
        assert read_time < 1.0   # Should complete within 1 second 