# GoPro Highlights Script - Test Suite

This directory contains comprehensive tests for the GoPro highlight extraction script. The test suite is designed to validate both individual components and their interactions using modern testing practices.

## Test Structure

```
tests/
├── README.md              # This file
├── conftest.py            # Shared fixtures and test utilities
├── __init__.py            # Package marker
├── unit/                  # Unit tests for individual components
└── integration/           # Integration tests for component interactions
```

## Test Categories

### Unit Tests (107 tests)
- **`test_config.py`** Configuration validation, worker optimization, and dataclass behavior
- **`test_main.py`** Main function execution, CLI parsing, and dependency checking
- **`test_mp4_parsing.py`** MP4 box parsing, HiLight tag extraction, and format handling
- **`test_progress_indicators.py`** Progress bars, spinner contexts, and async UI components
- **`test_video_processor.py`** Video processing, motion analysis, and clip extraction

### Integration Tests (32 tests)
- **`test_file_handling.py`** Real file operations, permissions, concurrency, and error handling
- **`test_full_pipeline.py`** End-to-end workflows, component interactions, and data flow

## Key Testing Features

### Synthetic Test Data
The test suite uses a custom `MP4TestDataGenerator` class that creates synthetic MP4 data without requiring actual video files:

```python
# Create test MP4 with HiLight tags
timestamps = [5000, 15000, 30000]  # milliseconds
mp4_data = mp4_generator.create_complete_mp4_with_hilights(timestamps)
```

### Async Testing Support
Full support for testing async operations with proper event loop management:

```python
@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_operation()
    assert result == expected_value
```

### Comprehensive Mocking
External dependencies (ffmpeg, file system) are mocked to ensure tests are:
- Fast and reliable
- Independent of system configuration
- Focused on testing our code logic

## Running Tests

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Test Execution

#### Run All Tests
```bash
pytest
```

#### Run Only Unit Tests
```bash
pytest tests/unit/
```

#### Run Only Integration Tests
```bash
pytest tests/integration/
```

#### Run Specific Test File
```bash
pytest tests/unit/test_mp4_parsing.py
```

#### Run Specific Test Class
```bash
pytest tests/unit/test_mp4_parsing.py::TestMP4BoxParsing
```

#### Run Specific Test Method
```bash
pytest tests/unit/test_mp4_parsing.py::TestMP4BoxParsing::test_find_boxes_single_box
```

### Test Markers

The test suite uses pytest markers to categorize tests:

#### Available Markers
- `unit`: Unit tests (fast, isolated)
- `integration`: Integration tests (slower, real operations)
- `slow`: Tests that take longer to run
- `mp4`: Tests related to MP4 parsing
- `ffmpeg`: Tests that mock ffmpeg operations
- `asyncio`: Tests that use async/await

#### Run Tests by Marker
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only MP4-related tests
pytest -m mp4

# Run all tests except slow ones
pytest -m "not slow"

# Run unit tests that involve MP4 parsing
pytest -m "unit and mp4"
```

### Verbose Output

#### Show Test Names and Results
```bash
pytest -v
```

#### Show Print Statements
```bash
pytest -s
```

#### Show Both Verbose and Print Output
```bash
pytest -v -s
```

### Test Coverage

#### Run Tests with Coverage Report
```bash
pytest --cov=extract_highlights
```

#### Generate HTML Coverage Report
```bash
pytest --cov=extract_highlights --cov-report=html
```

#### Show Missing Lines
```bash
pytest --cov=extract_highlights --cov-report=term-missing
```

### Performance Testing

#### Run Only Fast Tests (Skip Slow Ones)
```bash
pytest -m "not slow"
```

#### Run Performance-Critical Tests
```bash
pytest tests/integration/test_file_handling.py::TestFileSystemPerformance
```

### Parallel Test Execution

#### Install pytest-xdist for Parallel Execution
```bash
pip install pytest-xdist
```

#### Run Tests in Parallel
```bash
# Use all available CPU cores
pytest -n auto

# Use specific number of workers
pytest -n 4
```

### Debugging Tests

#### Stop on First Failure
```bash
pytest -x
```

#### Enter Debugger on Failure
```bash
pytest --pdb
```

#### Show Local Variables on Failure
```bash
pytest -l
```

#### Run Last Failed Tests Only
```bash
pytest --lf
```

## Test Configuration

The test suite is configured via `pytest.ini` in the project root:

```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
markers =
    unit: Unit tests
    integration: Integration tests
    slow: Slow tests
    mp4: MP4 parsing tests
    ffmpeg: FFmpeg-related tests
    asyncio: Async tests
asyncio_mode = auto
```

## Common Test Patterns

### Testing Async Functions
```python
@pytest.mark.asyncio
async def test_async_operation():
    result = await async_function()
    assert result == expected
```

### Using Fixtures
```python
def test_with_temp_file(tmp_path, mp4_generator):
    # tmp_path provides a temporary directory
    # mp4_generator provides synthetic MP4 data
    video_file = tmp_path / "test.mp4"
    mp4_data = mp4_generator.create_complete_mp4_with_hilights([5000])
    video_file.write_bytes(mp4_data)
```

### Mocking External Dependencies
```python
@patch('asyncio.create_subprocess_exec')
async def test_with_mocked_ffmpeg(mock_exec):
    mock_process = AsyncMock()
    mock_process.communicate.return_value = (b"output", b"")
    mock_process.returncode = 0
    mock_exec.return_value = mock_process
    
    # Test code that uses ffmpeg
```

## Troubleshooting

### Common Issues

#### Tests Fail Due to Missing Dependencies
```bash
# Install test dependencies
pip install pytest pytest-asyncio
```

#### Async Tests Not Running Properly
Ensure you have the `asyncio` marker:
```python
@pytest.mark.asyncio
async def test_async_function():
    pass
```

#### File Permission Issues on Windows
Some integration tests may fail on Windows due to file permission differences. These tests are designed to handle such cases gracefully.

#### Slow Test Performance
```bash
# Skip slow tests for faster development
pytest -m "not slow"

# Run tests in parallel
pytest -n auto
```

### Getting Help

#### Show Available Markers
```bash
pytest --markers
```

#### Show Test Collection (Don't Run)
```bash
pytest --collect-only
```

#### Show Configuration
```bash
pytest --help
```

## Contributing to Tests

### Adding New Tests

1. **Unit Tests**: Add to appropriate file in `tests/unit/`
2. **Integration Tests**: Add to appropriate file in `tests/integration/`
3. **Use appropriate markers**: `@pytest.mark.unit`, `@pytest.mark.integration`, etc.
4. **Follow naming conventions**: `test_*` for functions, `Test*` for classes

### Test Best Practices

1. **Use descriptive test names** that explain what is being tested
2. **Keep tests focused** on a single behavior or scenario
3. **Use fixtures** for common setup code
4. **Mock external dependencies** to keep tests fast and reliable
5. **Test edge cases** and error conditions
6. **Use appropriate assertions** with clear failure messages

### Example Test Structure
```python
class TestFeatureName:
    """Test description for the feature."""
    
    def test_normal_case(self):
        """Test the normal, expected behavior."""
        # Arrange
        setup_code()
        
        # Act
        result = function_under_test()
        
        # Assert
        assert result == expected_value
    
    def test_edge_case(self):
        """Test edge case or error condition."""
        with pytest.raises(ExpectedException):
            function_that_should_fail()
    
    @pytest.mark.asyncio
    async def test_async_case(self):
        """Test async behavior."""
        result = await async_function()
        assert result == expected
```
