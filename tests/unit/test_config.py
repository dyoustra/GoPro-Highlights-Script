"""
Unit tests for configuration and utility functions.
"""

import pytest
import os
from unittest.mock import patch, MagicMock

from extract_highlights import Config, get_optimal_workers


class TestGetOptimalWorkers:
    """Test the get_optimal_workers function."""
    
    def test_get_optimal_workers_normal_cpu_count(self):
        """Test optimal workers calculation with normal CPU count."""
        with patch('os.cpu_count', return_value=8):
            result = get_optimal_workers()
            # Should be 75% of 8 = 6
            assert result == 6
    
    def test_get_optimal_workers_high_cpu_count(self):
        """Test optimal workers calculation with high CPU count."""
        with patch('os.cpu_count', return_value=16):
            result = get_optimal_workers()
            # Should be capped at 8 (max limit)
            assert result == 8
    
    def test_get_optimal_workers_low_cpu_count(self):
        """Test optimal workers calculation with low CPU count."""
        with patch('os.cpu_count', return_value=2):
            result = get_optimal_workers()
            # Should be minimum of 2
            assert result == 2
    
    def test_get_optimal_workers_single_cpu(self):
        """Test optimal workers calculation with single CPU."""
        with patch('os.cpu_count', return_value=1):
            result = get_optimal_workers()
            # Should be minimum of 2
            assert result == 2
    
    def test_get_optimal_workers_cpu_count_none(self):
        """Test optimal workers calculation when cpu_count returns None."""
        with patch('os.cpu_count', return_value=None):
            result = get_optimal_workers()
            # Should fallback to 4, then calculate 75% = 3, but minimum is 2
            assert result == 3
    
    def test_get_optimal_workers_edge_cases(self):
        """Test edge cases for optimal workers calculation."""
        # Test various CPU counts
        test_cases = [
            (3, 2),   # 3 * 0.75 = 2.25 -> 2 (minimum)
            (4, 3),   # 4 * 0.75 = 3
            (6, 4),   # 6 * 0.75 = 4.5 -> 4
            (12, 8),  # 12 * 0.75 = 9 -> 8 (capped)
        ]
        
        for cpu_count, expected in test_cases:
            with patch('os.cpu_count', return_value=cpu_count):
                result = get_optimal_workers()
                assert result == expected, f"CPU count {cpu_count} should give {expected} workers, got {result}"


class TestConfig:
    """Test the Config dataclass."""
    
    def test_config_default_values(self):
        """Test Config with default values."""
        config = Config()
        
        assert config.clip_duration == 15
        assert config.motion_threshold == 0.2
        assert config.min_gap_seconds == 10
        assert config.max_workers == get_optimal_workers()
        assert config.video_extensions == ['.mp4', '.MP4', '.mov', '.MOV']
        assert config.prefer_hilight_tags is True
    
    def test_config_custom_values(self):
        """Test Config with custom values."""
        config = Config(
            clip_duration=30,
            motion_threshold=0.5,
            min_gap_seconds=20,
            max_workers=4,
            video_extensions=['.mp4', '.avi'],
            prefer_hilight_tags=False
        )
        
        assert config.clip_duration == 30
        assert config.motion_threshold == 0.5
        assert config.min_gap_seconds == 20
        assert config.max_workers == 4
        assert config.video_extensions == ['.mp4', '.avi']
        assert config.prefer_hilight_tags is False
    
    def test_config_post_init_video_extensions(self):
        """Test that video_extensions is set correctly in __post_init__."""
        # Test with None (should use defaults)
        config = Config(video_extensions=None)
        assert config.video_extensions == ['.mp4', '.MP4', '.mov', '.MOV']
        
        # Test with custom extensions
        custom_extensions = ['.mkv', '.avi']
        config = Config(video_extensions=custom_extensions)
        assert config.video_extensions == custom_extensions
    
    def test_config_post_init_max_workers(self):
        """Test that max_workers is set correctly in __post_init__."""
        # Test with None (should use get_optimal_workers)
        with patch('extract_highlights.get_optimal_workers', return_value=6):
            config = Config(max_workers=None)
            assert config.max_workers == 6
        
        # Test with custom max_workers
        config = Config(max_workers=8)
        assert config.max_workers == 8
    
    def test_config_immutable_after_creation(self):
        """Test that Config behaves as expected after creation."""
        config = Config()
        
        # Should be able to access attributes
        assert isinstance(config.clip_duration, int)
        assert isinstance(config.motion_threshold, float)
        assert isinstance(config.min_gap_seconds, int)
        assert isinstance(config.max_workers, int)
        assert isinstance(config.video_extensions, list)
        assert isinstance(config.prefer_hilight_tags, bool)
    
    def test_config_validation_ranges(self):
        """Test Config with edge case values."""
        # Test minimum values
        config = Config(
            clip_duration=1,
            motion_threshold=0.0,
            min_gap_seconds=0,
            max_workers=1
        )
        
        assert config.clip_duration == 1
        assert config.motion_threshold == 0.0
        assert config.min_gap_seconds == 0
        assert config.max_workers == 1
        
        # Test maximum reasonable values
        config = Config(
            clip_duration=300,  # 5 minutes
            motion_threshold=1.0,
            min_gap_seconds=120,  # 2 minutes
            max_workers=32
        )
        
        assert config.clip_duration == 300
        assert config.motion_threshold == 1.0
        assert config.min_gap_seconds == 120
        assert config.max_workers == 32
    
    def test_config_with_different_video_extensions(self):
        """Test Config with various video extension formats."""
        test_cases = [
            ['.mp4'],
            ['.MP4', '.mov'],
            ['.mp4', '.MP4', '.mov', '.MOV', '.avi', '.mkv'],
            []  # Empty list
        ]
        
        for extensions in test_cases:
            config = Config(video_extensions=extensions)
            assert config.video_extensions == extensions
    
    def test_config_type_hints_compatibility(self):
        """Test that Config works with type hints as expected."""
        # This tests that the dataclass is properly typed
        config = Config()
        
        # These should not raise type errors in a type checker
        clip_duration: int = config.clip_duration
        motion_threshold: float = config.motion_threshold
        min_gap_seconds: int = config.min_gap_seconds
        max_workers: int = config.max_workers
        video_extensions: list = config.video_extensions
        prefer_hilight_tags: bool = config.prefer_hilight_tags
        
        # Verify types at runtime
        assert isinstance(clip_duration, int)
        assert isinstance(motion_threshold, float)
        assert isinstance(min_gap_seconds, int)
        assert isinstance(max_workers, int)
        assert isinstance(video_extensions, list)
        assert isinstance(prefer_hilight_tags, bool) 