"""
Integration tests for the full GoPro highlight extraction pipeline.

These tests validate end-to-end workflows with real component interactions,
data flow between components, error propagation, and async coordination.
"""

import pytest
import asyncio
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, AsyncMock, MagicMock
import struct
import logging

from config import Config
from video_processor import VideoProcessor
from gopro_hilight_extractor import GoProHiLightExtractor
from ui_components import ProgressBar, SpinnerContext
from extract_highlights import main


class TestFullPipelineIntegration:
    """Test end-to-end pipeline with real component interactions."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_pipeline_with_hilight_tags(self, tmp_path, mp4_generator):
        """Test complete pipeline from MP4 parsing to clip extraction."""
        # Create test data with sufficient gaps (default min_gap_seconds=10)
        timestamps = [5000, 20000, 40000]  # 5s, 20s, 40s in milliseconds (15s+ gaps)
        mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps, use_new_format=True)
        
        # Set up directories
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        
        # Create test video file
        video_file = input_dir / "GX010123.mp4"
        video_file.write_bytes(mp4_data)
        
        # Create processor with real components
        config = Config(clip_duration=10, prefer_hilight_tags=True)
        processor = VideoProcessor(config)
        
        # Mock only the ffmpeg calls (external dependencies)
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            # Mock ffprobe for duration
            mock_ffprobe = AsyncMock()
            mock_ffprobe.communicate.return_value = (b"120.0\n", b"")
            mock_ffprobe.returncode = 0
            
            # Mock ffmpeg for clip extraction
            mock_ffmpeg = AsyncMock()
            mock_ffmpeg.communicate.return_value = (b"", b"")
            mock_ffmpeg.returncode = 0
            
            # Configure mock to return different processes based on command
            def mock_subprocess(*args, **kwargs):
                if args[0] == 'ffprobe':
                    return mock_ffprobe
                elif args[0] == 'ffmpeg':
                    return mock_ffmpeg
                return AsyncMock()
            
            mock_exec.side_effect = mock_subprocess
            
            # Mock file creation for clip extraction
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.stat') as mock_stat:
                    mock_stat.return_value.st_size = 5000  # 5KB file
                    
                    # Process the video
                    clips_extracted = await processor.process_video(video_file, output_dir)
                    
                    # Verify results
                    assert clips_extracted == (3, 0)  # 3 hilight clips, 0 motion clips
                    
                    # Verify HiLight extraction worked
                    extractor = GoProHiLightExtractor()
                    highlights = await extractor.extract_hilight_tags(video_file)
                    expected_highlights = [5.0, 20.0, 40.0]  # Converted to seconds
                    assert highlights == expected_highlights
                    
                    # Verify ffmpeg was called for each clip + scene detection
                    ffmpeg_calls = [call for call in mock_exec.call_args_list if call[0][0] == 'ffmpeg']
                    assert len(ffmpeg_calls) == 4  # 3 clips + 1 scene detection
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_fallback_to_motion_analysis(self, tmp_path, mp4_generator):
        """Test pipeline fallback when no HiLight tags are found."""
        # Create MP4 without HiLight tags
        mp4_data = mp4_generator.create_mp4_without_hilights()
        
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        
        video_file = input_dir / "GX010124.mp4"
        video_file.write_bytes(mp4_data)
        
        config = Config(prefer_hilight_tags=True, motion_threshold=0.3)
        processor = VideoProcessor(config)
        
        # Mock ffmpeg calls
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_ffprobe = AsyncMock()
            mock_ffprobe.communicate.return_value = (b"60.0\n", b"")
            mock_ffprobe.returncode = 0
            
            # Mock motion analysis with stderr output
            mock_motion_ffmpeg = AsyncMock()
            motion_stderr = b"pts_time:10.5\npts_time:25.8\npts_time:45.2\n"
            mock_motion_ffmpeg.communicate.return_value = (b"", motion_stderr)
            mock_motion_ffmpeg.returncode = 0
            
            # Mock clip extraction ffmpeg
            mock_clip_ffmpeg = AsyncMock()
            mock_clip_ffmpeg.communicate.return_value = (b"", b"")
            mock_clip_ffmpeg.returncode = 0
            
            def mock_subprocess(*args, **kwargs):
                if args[0] == 'ffprobe':
                    return mock_ffprobe
                elif args[0] == 'ffmpeg' and 'select=' in str(args):
                    return mock_motion_ffmpeg
                elif args[0] == 'ffmpeg':
                    return mock_clip_ffmpeg
                return AsyncMock()
            
            mock_exec.side_effect = mock_subprocess
            
            # Mock file system operations
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.stat') as mock_stat:
                    mock_stat.return_value.st_size = 3000
                    
                    # Mock the temporary file creation for motion analysis
                    def mock_write_text(content):
                        pass  # Mock the write operation
                    
                    def mock_read_text():
                        return "pts_time:10.5\npts_time:25.8\npts_time:45.2\n"
                    
                    with patch('pathlib.Path.write_text', side_effect=mock_write_text):
                        with patch('pathlib.Path.read_text', side_effect=mock_read_text):
                            clips_extracted = await processor.process_video(video_file, output_dir)
                        
                        # Should have fallen back to motion analysis
                        assert clips_extracted == (0, 3)  # 0 hilight clips, 3 motion clips
                        
                        # Verify motion analysis was called
                        motion_calls = [call for call in mock_exec.call_args_list 
                                      if len(call[0]) > 2 and 'select=' in str(call[0])]
                        assert len(motion_calls) == 1
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_propagation_across_components(self, tmp_path, mp4_generator):
        """Test that errors propagate correctly across component boundaries."""
        # Create valid MP4 with HiLight tags
        timestamps = [10000, 20000]
        mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps)
        
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        video_file = input_dir / "test_video.mp4"
        video_file.write_bytes(mp4_data)
        
        config = Config()
        processor = VideoProcessor(config)
        
        # Test ffprobe failure propagation
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_ffprobe = AsyncMock()
            mock_ffprobe.communicate.return_value = (b"", b"File not found")
            mock_ffprobe.returncode = 1
            
            mock_exec.return_value = mock_ffprobe
            
            # Should handle error gracefully and return 0 clips
            clips_extracted = await processor.process_video(video_file, tmp_path / "output")
            assert clips_extracted == (0, 0)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_data_flow_validation(self, tmp_path, mp4_generator):
        """Test data flow between MP4 parser → HiLight extractor → VideoProcessor."""
        # Create MP4 with specific timestamps with sufficient gaps
        original_timestamps = [3000, 15000, 30000, 45000]  # milliseconds with 12s+ gaps
        mp4_data = mp4_generator.create_complete_mp4_with_hilights(original_timestamps)
        
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        video_file = input_dir / "test_data_flow.mp4"
        video_file.write_bytes(mp4_data)
        
        # Step 1: Test MP4 parsing
        extractor = GoProHiLightExtractor()
        
        # Test box parsing
        with open(video_file, 'rb') as f:
            boxes = extractor.find_boxes(f)
            assert b'ftyp' in boxes
            assert b'moov' in boxes
        
        # Step 2: Test HiLight extraction
        extracted_timestamps = await extractor.extract_hilight_tags(video_file)
        expected_seconds = [ts / 1000.0 for ts in original_timestamps]
        assert extracted_timestamps == expected_seconds
        
        # Step 3: Test VideoProcessor integration
        config = Config(min_gap_seconds=3)  # Filter some timestamps
        processor = VideoProcessor(config)
        
        # Mock external dependencies
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process
            
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.stat') as mock_stat:
                    mock_stat.return_value.st_size = 4000
                    
                    # Process video and verify filtering
                    clips_extracted = await processor.process_video(video_file, tmp_path / "output")
                    
                    # Should have filtered some timestamps due to min_gap_seconds=3
                    # Original: [3.0, 15.0, 30.0, 45.0] -> After filtering: [3.0, 15.0, 30.0, 45.0]
                    # (all are > 3 seconds apart, so all should be kept)
                    assert clips_extracted == (4, 0)  # 4 hilight clips, 0 motion clips
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_async_coordination_multiple_components(self, tmp_path, mp4_generator):
        """Test async coordination between multiple real components."""
        # Create multiple video files with sufficient gaps
        video_files = []
        for i in range(3):
            timestamps = [i * 30000 + 5000, i * 30000 + 20000]  # Different timestamps per file, 15s gaps
            mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps)
            
            input_dir = tmp_path / "input"
            input_dir.mkdir(exist_ok=True)
            video_file = input_dir / f"video_{i}.mp4"
            video_file.write_bytes(mp4_data)
            video_files.append(video_file)
        
        config = Config(max_workers=2)
        processor = VideoProcessor(config)
        
        # Track processing order and timing
        processing_order = []
        
        async def track_processing(original_method):
            async def wrapper(video_path, output_dir):
                processing_order.append(video_path.name)
                return await original_method(video_path, output_dir)
            return wrapper
        
        # Mock external dependencies
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_process = AsyncMock()
            mock_process.communicate.return_value = (b"", b"")
            mock_process.returncode = 0
            mock_exec.return_value = mock_process
            
            with patch('pathlib.Path.stat') as mock_stat:
                mock_stat_result = MagicMock()
                mock_stat_result.st_size = 2000
                mock_stat_result.st_mode = 0o755  # Regular file mode
                mock_stat.return_value = mock_stat_result
                
                # Process all videos
                results = []
                for video_file in video_files:
                    result = await processor.process_video(video_file, tmp_path / "output")
                    results.append(result)
                
                # All should have processed successfully - expecting (hilight_clips, motion_clips)
                assert all(result == (2, 0) for result in results)
                
                # Verify each video was processed with its own HiLight tags
                for i, video_file in enumerate(video_files):
                    extractor = GoProHiLightExtractor()
                    highlights = await extractor.extract_hilight_tags(video_file)
                    expected = [(i * 30 + 5), (i * 30 + 20)]  # Converted to seconds
                    assert highlights == expected


class TestProgressIndicatorIntegration:
    """Test progress indicators in real pipeline context."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_progress_bar_with_real_processing(self, tmp_path, mp4_generator):
        """Test ProgressBar with actual processing workflow."""
        # Create test videos
        video_files = []
        for i in range(5):
            timestamps = [i * 5000 + 1000]  # One highlight per video
            mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps)
            
            input_dir = tmp_path / "input"
            input_dir.mkdir(exist_ok=True)
            video_file = input_dir / f"progress_test_{i}.mp4"
            video_file.write_bytes(mp4_data)
            video_files.append(video_file)
        
        # Test ProgressBar with real updates
        progress = ProgressBar(total=len(video_files))
        
        with patch('sys.stdout') as mock_stdout:
            for i, video_file in enumerate(video_files):
                progress.update(i)
                # Simulate processing delay
                await asyncio.sleep(0.01)
            
            progress.update(len(video_files))
            progress.finish()
            
            # Verify progress was updated
            assert mock_stdout.write.call_count > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_spinner_context_with_real_operations(self, tmp_path, mp4_generator):
        """Test SpinnerContext with real async operations."""
        timestamps = [10000, 20000]
        mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps)
        
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        video_file = input_dir / "spinner_test.mp4"
        video_file.write_bytes(mp4_data)
        
        extractor = GoProHiLightExtractor()
        
        with patch('sys.stdout') as mock_stdout:
            async with SpinnerContext("Extracting HiLight tags"):
                highlights = await extractor.extract_hilight_tags(video_file)
                # Add some processing time
                await asyncio.sleep(0.1)
            
            # Verify spinner output
            assert mock_stdout.write.call_count > 0
            assert highlights == [10.0, 20.0]


class TestErrorHandlingIntegration:
    """Test error handling across the integrated pipeline."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_corrupted_mp4_handling(self, tmp_path, mp4_generator):
        """Test handling of corrupted MP4 files."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        
        # Create corrupted MP4
        corrupted_file = input_dir / "corrupted.mp4"
        corrupted_data = mp4_generator.create_invalid_mp4()
        corrupted_file.write_bytes(corrupted_data)
        
        config = Config()
        processor = VideoProcessor(config)
        
        # Should handle gracefully without crashing
        clips_extracted = await processor.process_video(corrupted_file, tmp_path / "output")
        assert clips_extracted == (0, 0)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_missing_dependencies_handling(self, tmp_path, mp4_generator):
        """Test handling when external dependencies are missing."""
        timestamps = [5000, 15000]
        mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps)
        
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        video_file = input_dir / "test_deps.mp4"
        video_file.write_bytes(mp4_data)
        
        config = Config()
        processor = VideoProcessor(config)
        
        # Mock ffmpeg not found
        with patch('asyncio.create_subprocess_exec', side_effect=FileNotFoundError("ffmpeg not found")):
            clips_extracted = await processor.process_video(video_file, tmp_path / "output")
            assert clips_extracted == (0, 0)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_permission_errors_handling(self, tmp_path, mp4_generator):
        """Test handling of permission errors."""
        timestamps = [10000]
        mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps)
        
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        video_file = input_dir / "test_perms.mp4"
        video_file.write_bytes(mp4_data)
        
        config = Config()
        processor = VideoProcessor(config)
        
        # Mock permission error during clip extraction
        with patch('asyncio.create_subprocess_exec') as mock_exec:
            mock_ffprobe = AsyncMock()
            mock_ffprobe.communicate.return_value = (b"60.0\n", b"")
            mock_ffprobe.returncode = 0
            
            mock_ffmpeg = AsyncMock()
            mock_ffmpeg.communicate.side_effect = PermissionError("Permission denied")
            
            def mock_subprocess(*args, **kwargs):
                if args[0] == 'ffprobe':
                    return mock_ffprobe
                elif args[0] == 'ffmpeg':
                    return mock_ffmpeg
                return AsyncMock()
            
            mock_exec.side_effect = mock_subprocess
            
            clips_extracted = await processor.process_video(video_file, tmp_path / "output")
            assert clips_extracted == (0, 0)


class TestLoggingIntegration:
    """Test logging integration across components."""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_logging_throughout_pipeline(self, tmp_path, mp4_generator, caplog):
        """Test that logging works throughout the pipeline."""
        timestamps = [7000, 14000]
        mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps)
        
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        video_file = input_dir / "logging_test.mp4"
        video_file.write_bytes(mp4_data)
        
        config = Config()
        processor = VideoProcessor(config)
        
        with caplog.at_level(logging.INFO):
            # Mock external dependencies
            with patch('asyncio.create_subprocess_exec') as mock_exec:
                mock_process = AsyncMock()
                mock_process.communicate.return_value = (b"", b"")
                mock_process.returncode = 0
                mock_exec.return_value = mock_process
                
                with patch('pathlib.Path.exists', return_value=True):
                    with patch('pathlib.Path.stat') as mock_stat:
                        mock_stat.return_value.st_size = 3000
                        
                        await processor.process_video(video_file, tmp_path / "output")
        
        # Verify logging occurred
        log_messages = [record.message for record in caplog.records]
        assert any("Processing:" in msg for msg in log_messages)
        assert any("Extracted" in msg and "HiLight tags" in msg for msg in log_messages) 