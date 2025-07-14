"""
Unit tests for progress indicators (ProgressBar and SpinnerContext).
"""

import pytest
import asyncio
import io
import sys
from unittest.mock import patch, MagicMock

from ui_components import ProgressBar, SpinnerContext


class TestProgressBar:
    """Test the ProgressBar class."""
    
    def test_progress_bar_initialization(self):
        """Test ProgressBar initialization."""
        progress = ProgressBar(total=100)
        
        assert progress.total == 100
        assert progress.width == 50  # default width
        assert progress.current == 0
        
        # Test with custom width
        progress = ProgressBar(total=50, width=20)
        assert progress.total == 50
        assert progress.width == 20
        assert progress.current == 0
    
    def test_progress_bar_update_beginning(self):
        """Test ProgressBar update at the beginning."""
        progress = ProgressBar(total=100)
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            progress.update(0)
            output = mock_stdout.getvalue()
            
            # Should show 0% progress
            assert '0%' in output
            assert '(0/100)' in output
            assert '[' in output and ']' in output
    
    def test_progress_bar_update_middle(self):
        """Test ProgressBar update in the middle."""
        progress = ProgressBar(total=100)
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            progress.update(50)
            output = mock_stdout.getvalue()
            
            # Should show 50% progress
            assert '50%' in output
            assert '(50/100)' in output
            assert '=' in output  # Should have some filled bars
            assert '-' in output  # Should have some empty bars
    
    def test_progress_bar_update_complete(self):
        """Test ProgressBar update when complete."""
        progress = ProgressBar(total=100)
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            progress.update(100)
            output = mock_stdout.getvalue()
            
            # Should show 100% progress
            assert '100%' in output
            assert '(100/100)' in output
            # Should be all filled bars (no empty bars)
            assert '=' in output
    
    def test_progress_bar_update_over_total(self):
        """Test ProgressBar update when current exceeds total."""
        progress = ProgressBar(total=100)
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            progress.update(150)
            output = mock_stdout.getvalue()
            
            # Should handle gracefully (percentage calculation might be > 100%)
            assert '(150/100)' in output
    
    def test_progress_bar_finish(self):
        """Test ProgressBar finish method."""
        progress = ProgressBar(total=100)
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            progress.finish()
            output = mock_stdout.getvalue()
            
            # Should print a newline
            assert output == '\n'
    
    def test_progress_bar_different_widths(self):
        """Test ProgressBar with different widths."""
        test_cases = [
            (10, 20),   # Small width
            (50, 100),  # Large width
            (1, 100),   # Very small width
        ]
        
        for width, total in test_cases:
            progress = ProgressBar(total=total, width=width)
            
            with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
                progress.update(total // 2)  # 50% progress
                output = mock_stdout.getvalue()
                
                # Should contain progress elements
                assert '[' in output
                assert ']' in output
                assert '50%' in output
    
    def test_progress_bar_zero_total(self):
        """Test ProgressBar with zero total (edge case)."""
        progress = ProgressBar(total=0)
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            # This might cause division by zero, should handle gracefully
            try:
                progress.update(0)
                output = mock_stdout.getvalue()
                # Should not crash
                assert isinstance(output, str)
            except ZeroDivisionError:
                # If it raises ZeroDivisionError, that's also acceptable behavior
                pass
    
    def test_progress_bar_sequential_updates(self):
        """Test ProgressBar with sequential updates."""
        progress = ProgressBar(total=10)
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            for i in range(11):  # 0 to 10
                progress.update(i)
                
            output = mock_stdout.getvalue()
            
            # Should have multiple progress updates
            assert '0%' in output
            assert '100%' in output
            assert '(10/10)' in output


class TestSpinnerContext:
    """Test the SpinnerContext class."""
    
    def test_spinner_context_initialization(self):
        """Test SpinnerContext initialization."""
        spinner = SpinnerContext("Loading...")
        
        assert spinner.message == "Loading..."
        assert spinner.spinner_chars == '⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
        assert spinner.running is False
        assert spinner._spin_task is None
    
    @pytest.mark.asyncio
    async def test_spinner_context_as_context_manager(self):
        """Test SpinnerContext as async context manager."""
        spinner = SpinnerContext("Processing...")
        
        # Mock the spinning task to avoid actual spinning
        with patch.object(spinner, '_spin') as mock_spin:
            mock_spin.return_value = asyncio.create_task(asyncio.sleep(0.1))
            
            async with spinner:
                assert spinner.running is True
                assert spinner._spin_task is not None
            
            # After exiting context, should be stopped
            assert spinner.running is False
    
    @pytest.mark.asyncio
    async def test_spinner_context_completion_message(self):
        """Test SpinnerContext prints completion message."""
        spinner = SpinnerContext("Working...")
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            with patch.object(spinner, '_spin') as mock_spin:
                mock_spin.return_value = asyncio.create_task(asyncio.sleep(0.01))
                
                async with spinner:
                    await asyncio.sleep(0.02)  # Brief pause
                
                output = mock_stdout.getvalue()
                
                # Should print completion message
                assert '✓ Working...' in output
    
    @pytest.mark.asyncio
    async def test_spinner_context_exception_handling(self):
        """Test SpinnerContext handles exceptions properly."""
        spinner = SpinnerContext("Testing...")
        
        with patch.object(spinner, '_spin') as mock_spin:
            mock_spin.return_value = asyncio.create_task(asyncio.sleep(0.01))
            
            try:
                async with spinner:
                    raise ValueError("Test exception")
            except ValueError:
                pass  # Expected
            
            # Should still clean up properly
            assert spinner.running is False
    
    @pytest.mark.asyncio
    async def test_spinner_spin_method(self):
        """Test the internal _spin method."""
        spinner = SpinnerContext("Spinning...")
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            # Start spinning
            spinner.running = True
            spin_task = asyncio.create_task(spinner._spin())
            
            # Let it spin briefly
            await asyncio.sleep(0.15)  # Should be enough for at least one spin
            
            # Stop spinning
            spinner.running = False
            spin_task.cancel()
            
            try:
                await spin_task
            except asyncio.CancelledError:
                pass  # Expected
            
            output = mock_stdout.getvalue()
            
            # Should have some spinner output
            assert 'Spinning...' in output
            # Should contain some spinner characters
            assert any(char in output for char in spinner.spinner_chars)
    
    @pytest.mark.asyncio
    async def test_spinner_cancellation(self):
        """Test SpinnerContext task cancellation."""
        spinner = SpinnerContext("Cancelling...")
        
        # Test that the spinner properly manages task lifecycle
        task_created = False
        task_cancelled = False
        
        async def mock_spin():
            nonlocal task_created, task_cancelled
            task_created = True
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                task_cancelled = True
                raise
        
        with patch.object(spinner, '_spin', side_effect=mock_spin):
            async with spinner:
                # Brief delay to let the task start
                await asyncio.sleep(0.01)
                assert task_created
                assert not task_cancelled
            
            # After context exit, task should be cancelled
            await asyncio.sleep(0.01)
            assert task_cancelled
    
    @pytest.mark.asyncio
    async def test_spinner_multiple_characters(self):
        """Test that spinner cycles through different characters."""
        spinner = SpinnerContext("Cycling...")
        
        with patch('sys.stdout', new_callable=io.StringIO) as mock_stdout:
            spinner.running = True
            
            # Manually call _spin for controlled testing
            spin_task = asyncio.create_task(spinner._spin())
            
            # Let it run for enough time to cycle through characters
            await asyncio.sleep(0.25)
            
            spinner.running = False
            spin_task.cancel()
            
            try:
                await spin_task
            except asyncio.CancelledError:
                pass
            
            output = mock_stdout.getvalue()
            
            # Should have multiple different spinner characters
            spinner_chars_in_output = [char for char in spinner.spinner_chars if char in output]
            assert len(spinner_chars_in_output) > 1, "Should cycle through multiple spinner characters"
    
    def test_spinner_context_message_types(self):
        """Test SpinnerContext with different message types."""
        test_messages = [
            "Loading...",
            "Processing data",
            "🔄 Working",
            "Very long message that might wrap around the terminal width",
            "",  # Empty message
            "Special chars: éñ中文"
        ]
        
        for message in test_messages:
            spinner = SpinnerContext(message)
            assert spinner.message == message
    
    @pytest.mark.asyncio
    async def test_spinner_context_nested_usage(self):
        """Test that SpinnerContext works correctly when nested."""
        spinner1 = SpinnerContext("Outer task...")
        spinner2 = SpinnerContext("Inner task...")
        
        with patch.object(spinner1, '_spin') as mock_spin1:
            with patch.object(spinner2, '_spin') as mock_spin2:
                mock_spin1.return_value = asyncio.create_task(asyncio.sleep(0.01))
                mock_spin2.return_value = asyncio.create_task(asyncio.sleep(0.01))
                
                async with spinner1:
                    assert spinner1.running is True
                    
                    async with spinner2:
                        assert spinner2.running is True
                        assert spinner1.running is True  # Should still be running
                    
                    assert spinner2.running is False
                    assert spinner1.running is True  # Should still be running
                
                assert spinner1.running is False
                assert spinner2.running is False 