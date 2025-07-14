#!/usr/bin/env python3
"""
UI components for GoPro Highlights Script.
"""

import asyncio


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