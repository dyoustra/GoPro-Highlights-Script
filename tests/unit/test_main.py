"""
Unit tests for main function and command line argument parsing.
"""

import pytest
import asyncio
import sys
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call
import argparse

from config import Config, get_optimal_workers
from video_processor import VideoProcessor
from extract_highlights import main


class TestMainFunction:
    """Test the main function."""
    
    @pytest.mark.asyncio
    async def test_main_successful_processing(self):
        """Test successful main function execution."""
        test_args = [
            'extract_highlights.py',
            '/fake/input/dir',
            '/fake/output/dir'
        ]
        
        with patch('sys.argv', test_args):
            with patch('extract_highlights.Path') as mock_path:
                # Mock input directory exists
                mock_input_dir = MagicMock()
                mock_input_dir.exists.return_value = True
                # Mock glob to return files for each extension
                def mock_glob(pattern):
                    if pattern == "*.mp4":
                        return [Path('video1.mp4')]
                    elif pattern == "*.MP4":
                        return [Path('video2.MP4')]
                    else:
                        return []
                mock_input_dir.glob.side_effect = mock_glob
                
                # Mock output directory
                mock_output_dir = MagicMock()
                mock_output_dir.mkdir = MagicMock()
                
                # Configure Path constructor to return appropriate mocks
                def path_constructor(path_str):
                    if path_str == '/fake/input/dir':
                        return mock_input_dir
                    elif path_str == '/fake/output/dir':
                        return mock_output_dir
                    return MagicMock()
                
                mock_path.side_effect = path_constructor
                
                # Mock VideoProcessor
                mock_processor = MagicMock()
                mock_processor.process_video = AsyncMock(return_value=(2, 0))
                
                with patch('extract_highlights.VideoProcessor', return_value=mock_processor):
                    await main()
                    
                    # Should have processed 2 videos
                    assert mock_processor.process_video.call_count == 2
    
    @pytest.mark.asyncio
    async def test_main_input_directory_not_exists(self):
        """Test main function when input directory doesn't exist."""
        test_args = [
            'extract_highlights.py',
            '/fake/nonexistent/dir',
            '/fake/output/dir'
        ]
        
        with patch('sys.argv', test_args):
            with patch('extract_highlights.Path') as mock_path:
                # Mock input directory doesn't exist
                mock_input_dir = MagicMock()
                mock_input_dir.exists.return_value = False
                
                mock_path.side_effect = lambda path_str: mock_input_dir if path_str == '/fake/nonexistent/dir' else MagicMock()
                
                with patch('sys.exit') as mock_exit:
                    await main()
                    
                    # Should exit with code 1 (may be called twice - once for missing dir, once for no files)
                    mock_exit.assert_called_with(1)
    
    @pytest.mark.asyncio
    async def test_main_no_video_files_found(self):
        """Test main function when no video files are found."""
        test_args = [
            'extract_highlights.py',
            '/fake/input/dir',
            '/fake/output/dir'
        ]
        
        with patch('sys.argv', test_args):
            with patch('extract_highlights.Path') as mock_path:
                # Mock input directory exists but has no video files
                mock_input_dir = MagicMock()
                mock_input_dir.exists.return_value = True
                mock_input_dir.glob.return_value = []  # No video files for any extension
                
                mock_output_dir = MagicMock()
                
                def path_constructor(path_str):
                    if path_str == '/fake/input/dir':
                        return mock_input_dir
                    elif path_str == '/fake/output/dir':
                        return mock_output_dir
                    return MagicMock()
                
                mock_path.side_effect = path_constructor
                
                with patch('sys.exit') as mock_exit:
                    await main()
                    
                    # Should exit with code 1
                    mock_exit.assert_called_with(1)
    
    @pytest.mark.asyncio
    async def test_main_with_custom_arguments(self):
        """Test main function with custom command line arguments."""
        test_args = [
            'extract_highlights.py',
            '/fake/input/dir',
            '/fake/output/dir',
            '--clip-duration', '30',
            '--motion-threshold', '0.5',
            '--min-gap', '20',
            '--max-workers', '6',
            '--disable-hilight-tags'
        ]
        
        with patch('sys.argv', test_args):
            with patch('extract_highlights.Path') as mock_path:
                # Mock directories and files
                mock_input_dir = MagicMock()
                mock_input_dir.exists.return_value = True
                # Mock glob to return one file for .mp4 extension
                def mock_glob(pattern):
                    if pattern == "*.mp4":
                        return [Path('video1.mp4')]
                    else:
                        return []
                mock_input_dir.glob.side_effect = mock_glob
                
                mock_output_dir = MagicMock()
                
                def path_constructor(path_str):
                    if path_str == '/fake/input/dir':
                        return mock_input_dir
                    elif path_str == '/fake/output/dir':
                        return mock_output_dir
                    return MagicMock()
                
                mock_path.side_effect = path_constructor
                
                # Mock VideoProcessor to capture config
                captured_config = None
                
                def mock_processor_init(config):
                    nonlocal captured_config
                    captured_config = config
                    mock_processor = MagicMock()
                    mock_processor.process_video = AsyncMock(return_value=(1, 0))
                    return mock_processor
                
                with patch('extract_highlights.VideoProcessor', side_effect=mock_processor_init):
                    await main()
                    
                    # Check that config was created with custom values
                    assert captured_config.clip_duration == 30
                    assert captured_config.motion_threshold == 0.5
                    assert captured_config.min_gap_seconds == 20
                    assert captured_config.max_workers == 6
                    assert captured_config.prefer_hilight_tags is False  # --disable-hilight-tags
    
    @pytest.mark.asyncio
    async def test_main_output_directory_creation(self):
        """Test that main function creates output directory."""
        test_args = [
            'extract_highlights.py',
            '/fake/input/dir',
            '/fake/output/dir'
        ]
        
        with patch('sys.argv', test_args):
            with patch('extract_highlights.Path') as mock_path:
                mock_input_dir = MagicMock()
                mock_input_dir.exists.return_value = True
                # Mock glob to return one file for .mp4 extension
                def mock_glob(pattern):
                    if pattern == "*.mp4":
                        return [Path('video1.mp4')]
                    else:
                        return []
                mock_input_dir.glob.side_effect = mock_glob
                
                mock_output_dir = MagicMock()
                
                def path_constructor(path_str):
                    if path_str == '/fake/input/dir':
                        return mock_input_dir
                    elif path_str == '/fake/output/dir':
                        return mock_output_dir
                    return MagicMock()
                
                mock_path.side_effect = path_constructor
                
                with patch('extract_highlights.VideoProcessor') as mock_processor_class:
                    mock_processor = MagicMock()
                    mock_processor.process_video = AsyncMock(return_value=(1, 0))
                    mock_processor_class.return_value = mock_processor
                    
                    await main()
                    
                    # Should have called mkdir on output directory
                    mock_output_dir.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    @pytest.mark.asyncio
    async def test_main_video_file_discovery(self):
        """Test that main function discovers video files correctly."""
        test_args = [
            'extract_highlights.py',
            '/fake/input/dir',
            '/fake/output/dir'
        ]
        
        with patch('sys.argv', test_args):
            with patch('extract_highlights.Path') as mock_path:
                mock_input_dir = MagicMock()
                mock_input_dir.exists.return_value = True
                
                # Mock glob calls for different extensions
                def mock_glob(pattern):
                    if pattern == '*.mp4':
                        return [Path('video1.mp4'), Path('video2.mp4')]
                    elif pattern == '*.MP4':
                        return [Path('VIDEO3.MP4')]
                    elif pattern == '*.mov':
                        return [Path('video4.mov')]
                    elif pattern == '*.MOV':
                        return []
                    return []
                
                mock_input_dir.glob.side_effect = mock_glob
                
                mock_output_dir = MagicMock()
                
                def path_constructor(path_str):
                    if path_str == '/fake/input/dir':
                        return mock_input_dir
                    elif path_str == '/fake/output/dir':
                        return mock_output_dir
                    return MagicMock()
                
                mock_path.side_effect = path_constructor
                
                with patch('extract_highlights.VideoProcessor') as mock_processor_class:
                    mock_processor = MagicMock()
                    mock_processor.process_video = AsyncMock(return_value=(1, 0))
                    mock_processor_class.return_value = mock_processor
                    
                    await main()
                    
                    # Should have processed 4 videos total
                    assert mock_processor.process_video.call_count == 4
    
    @pytest.mark.asyncio
    async def test_main_argument_parsing_defaults(self):
        """Test that main function uses correct default arguments."""
        test_args = [
            'extract_highlights.py',
            '/fake/input/dir',
            '/fake/output/dir'
        ]
        
        with patch('sys.argv', test_args):
            with patch('extract_highlights.Path') as mock_path:
                mock_input_dir = MagicMock()
                mock_input_dir.exists.return_value = True
                # Mock glob to return one file for .mp4 extension
                def mock_glob(pattern):
                    if pattern == "*.mp4":
                        return [Path('video1.mp4')]
                    else:
                        return []
                mock_input_dir.glob.side_effect = mock_glob
                
                mock_output_dir = MagicMock()
                
                def path_constructor(path_str):
                    if path_str == '/fake/input/dir':
                        return mock_input_dir
                    elif path_str == '/fake/output/dir':
                        return mock_output_dir
                    return MagicMock()
                
                mock_path.side_effect = path_constructor
                
                captured_config = None
                
                def mock_processor_init(config):
                    nonlocal captured_config
                    captured_config = config
                    mock_processor = MagicMock()
                    mock_processor.process_video = AsyncMock(return_value=(1, 0))
                    return mock_processor
                
                with patch('extract_highlights.VideoProcessor', side_effect=mock_processor_init):
                    await main()
                    
                    # Check default values
                    assert captured_config.clip_duration == 15
                    assert captured_config.motion_threshold == 0.2
                    assert captured_config.min_gap_seconds == 10
                    assert captured_config.max_workers == get_optimal_workers()
                    assert captured_config.prefer_hilight_tags is True


class TestArgumentParsing:
    """Test command line argument parsing."""
    
    def test_argument_parser_required_arguments(self):
        """Test that required arguments are parsed correctly."""
        parser = argparse.ArgumentParser(description='Extract highlights from GoPro videos')
        parser.add_argument('input_dir', type=Path, help='Directory containing GoPro videos')
        parser.add_argument('output_dir', type=Path, help='Directory to save highlight clips')
        
        args = parser.parse_args(['/input', '/output'])
        
        assert args.input_dir == Path('/input')
        assert args.output_dir == Path('/output')
    
    def test_argument_parser_optional_arguments(self):
        """Test that optional arguments are parsed correctly."""
        parser = argparse.ArgumentParser(description='Extract highlights from GoPro videos')
        parser.add_argument('input_dir', type=Path)
        parser.add_argument('output_dir', type=Path)
        parser.add_argument('--clip-duration', type=int, default=15)
        parser.add_argument('--motion-threshold', type=float, default=0.2)
        parser.add_argument('--min-gap', type=int, default=10)
        parser.add_argument('--max-workers', type=int, default=get_optimal_workers())
        parser.add_argument('--disable-hilight-tags', action='store_true')
        
        args = parser.parse_args([
            '/input', '/output',
            '--clip-duration', '30',
            '--motion-threshold', '0.5',
            '--min-gap', '20',
            '--max-workers', '8',
            '--disable-hilight-tags'
        ])
        
        assert args.clip_duration == 30
        assert args.motion_threshold == 0.5
        assert args.min_gap == 20
        assert args.max_workers == 8
        assert args.disable_hilight_tags is True
    
    def test_argument_parser_defaults(self):
        """Test that default values are set correctly."""
        parser = argparse.ArgumentParser(description='Extract highlights from GoPro videos')
        parser.add_argument('input_dir', type=Path)
        parser.add_argument('output_dir', type=Path)
        parser.add_argument('--clip-duration', type=int, default=15)
        parser.add_argument('--motion-threshold', type=float, default=0.2)
        parser.add_argument('--min-gap', type=int, default=10)
        parser.add_argument('--max-workers', type=int, default=get_optimal_workers())
        parser.add_argument('--disable-hilight-tags', action='store_true')
        
        args = parser.parse_args(['/input', '/output'])
        
        assert args.clip_duration == 15
        assert args.motion_threshold == 0.2
        assert args.min_gap == 10
        assert args.max_workers == get_optimal_workers()
        assert args.disable_hilight_tags is False


class TestDependencyChecking:
    """Test dependency checking functionality."""
    
    def test_ffmpeg_dependency_check_success(self):
        """Test successful ffmpeg dependency check."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            
            # This should not raise an exception
            try:
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
                subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                pytest.fail("Should not raise exception when ffmpeg is available")
    
    def test_ffmpeg_dependency_check_failure(self):
        """Test ffmpeg dependency check failure."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError("ffmpeg not found")
            
            with pytest.raises(FileNotFoundError):
                subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    
    def test_ffprobe_dependency_check_failure(self):
        """Test ffprobe dependency check failure."""
        with patch('subprocess.run') as mock_run:
            # ffmpeg succeeds, ffprobe fails
            mock_run.side_effect = [
                MagicMock(returncode=0),  # ffmpeg success
                subprocess.CalledProcessError(1, 'ffprobe')  # ffprobe failure
            ]
            
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            
            with pytest.raises(subprocess.CalledProcessError):
                subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)


class TestMainIntegration:
    """Test main function integration aspects."""
    
    @pytest.mark.asyncio
    async def test_main_prints_summary_information(self):
        """Test that main function prints summary information."""
        test_args = [
            'extract_highlights.py',
            '/fake/input/dir',
            '/fake/output/dir'
        ]
        
        with patch('sys.argv', test_args):
            with patch('extract_highlights.Path') as mock_path:
                mock_input_dir = MagicMock()
                mock_input_dir.exists.return_value = True
                # Mock glob to return files for .mp4 extension
                def mock_glob(pattern):
                    if pattern == "*.mp4":
                        return [Path('video1.mp4'), Path('video2.mp4')]
                    else:
                        return []
                mock_input_dir.glob.side_effect = mock_glob
                
                mock_output_dir = MagicMock()
                
                def path_constructor(path_str):
                    if path_str == '/fake/input/dir':
                        return mock_input_dir
                    elif path_str == '/fake/output/dir':
                        return mock_output_dir
                    return MagicMock()
                
                mock_path.side_effect = path_constructor
                
                with patch('extract_highlights.VideoProcessor') as mock_processor_class:
                    mock_processor = MagicMock()
                    mock_processor.process_video = AsyncMock(return_value=(2, 0))
                    mock_processor_class.return_value = mock_processor
                    
                    # Capture print output
                    with patch('builtins.print') as mock_print:
                        await main()
                        
                        # Should print summary information
                        print_calls = [call.args[0] if call.args else str(call) for call in mock_print.call_args_list]
                        summary_prints = [call for call in print_calls if 'clips extracted' in str(call) or 'videos to process' in str(call)]
                        
                        assert len(summary_prints) > 0, "Should print summary information"
    
    @pytest.mark.asyncio
    async def test_main_processes_videos_sequentially(self):
        """Test that main function processes videos in sequence."""
        test_args = [
            'extract_highlights.py',
            '/fake/input/dir',
            '/fake/output/dir'
        ]
        
        with patch('sys.argv', test_args):
            with patch('extract_highlights.Path') as mock_path:
                mock_input_dir = MagicMock()
                mock_input_dir.exists.return_value = True
                # Mock glob to return files for .mp4 extension
                def mock_glob(pattern):
                    if pattern == "*.mp4":
                        return [Path('video1.mp4'), Path('video2.mp4'), Path('video3.mp4')]
                    else:
                        return []
                mock_input_dir.glob.side_effect = mock_glob
                
                mock_output_dir = MagicMock()
                
                def path_constructor(path_str):
                    if path_str == '/fake/input/dir':
                        return mock_input_dir
                    elif path_str == '/fake/output/dir':
                        return mock_output_dir
                    return MagicMock()
                
                mock_path.side_effect = path_constructor
                
                with patch('extract_highlights.VideoProcessor') as mock_processor_class:
                    mock_processor = MagicMock()
                    
                    # Track call order
                    call_order = []
                    
                    async def mock_process_video(video_path, output_dir):
                        call_order.append(video_path.name)
                        return (1, 0)
                    
                    mock_processor.process_video = mock_process_video
                    mock_processor_class.return_value = mock_processor
                    
                    await main()
                    
                    # Should have processed all videos
                    assert len(call_order) == 3
                    assert 'video1.mp4' in call_order
                    assert 'video2.mp4' in call_order
                    assert 'video3.mp4' in call_order 