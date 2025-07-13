# GoPro Highlights Script

An automated Python script to extract highlight clips from GoPro videos using either built-in HiLight tags or motion analysis.

## Features

### 🏷️ GoPro HiLight Tag Extraction (Recommended)
- **Extracts user-created HiLight tags** from GoPro MP4 files
- **Works with tags created during recording** by pressing the mode button on your GoPro
- **Supports both newer (HERO6+) and older GoPro formats**
- **Much more accurate than motion analysis** - uses your actual highlights!
- **Automatic fallback** to motion analysis if no HiLight tags are found

### 📊 Motion Analysis (Fallback)
- Analyzes video for high-motion scenes using ffmpeg
- Configurable motion detection threshold
- Useful for non-GoPro videos or when HiLight tags aren't available

### ⚡ Performance & Usability
- **Async/await architecture** for efficient concurrent processing
- **Automatic worker optimization** based on your system's CPU cores
- **Modern progress indicators** with spinners and progress bars
- **Comprehensive logging** with both console and file output
- **Intelligent clip filtering** to avoid overlapping highlights

## How GoPro HiLight Tags Work

GoPro cameras allow users to create "HiLight" tags during recording by pressing the mode button. These tags are stored directly in the MP4 file's metadata and mark the exact moments you found interesting while filming.

**Benefits over motion analysis:**
- ✅ **User-controlled** - You decide what's a highlight
- ✅ **Perfectly accurate** - No false positives from camera shake or irrelevant motion
- ✅ **Instant processing** - No need to analyze the entire video
- ✅ **Works with any content** - Even slow-motion or static scenes you found interesting

**Supported GoPro models:**
- HERO6 Black and newer (GPMF format)
- HERO5 Black and older (HMMT format)
- Most GoPro models that support HiLight tagging

**Note:** HiLight tags created in GoPro Quik desktop software are NOT stored in the MP4 file and cannot be extracted.

## Installation

### Prerequisites
- Python 3.10 or later
- ffmpeg and ffprobe (for video processing)

### Install ffmpeg
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Install the script
```bash
git clone <repository-url>
cd gopro-highlights-script
# No additional Python packages required - uses only standard library!
```

## Usage

### Basic Usage
```bash
python extract_highlights.py /path/to/gopro/videos /path/to/output
```

### Advanced Options
```bash
python extract_highlights.py /path/to/gopro/videos /path/to/output \
    --clip-duration 20 \
    --motion-threshold 0.3 \
    --min-gap 15 \
    --max-workers 4
```

### Disable HiLight Tags (Use Motion Analysis Only)
```bash
python extract_highlights.py /path/to/gopro/videos /path/to/output \
    --disable-hilight-tags
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--clip-duration` | Duration of each highlight clip in seconds | 15 |
| `--motion-threshold` | Motion detection threshold (0.0-1.0) | 0.2 |
| `--min-gap` | Minimum gap between highlights in seconds | 10 |
| `--max-workers` | Maximum concurrent workers | Auto-detected |
| `--disable-hilight-tags` | Disable HiLight tag extraction | False |

## How It Works

### 1. HiLight Tag Extraction (Primary Method)
1. **MP4 Box Parsing**: Reads the MP4 container structure
2. **Metadata Location**: Finds the user data (`udta`) section
3. **Format Detection**: Detects newer GPMF or older HMMT format
4. **Tag Extraction**: Extracts timestamp data from HiLight tags
5. **Conversion**: Converts millisecond timestamps to seconds

### 2. Motion Analysis (Fallback Method)
1. **Scene Detection**: Uses ffmpeg to analyze motion vectors
2. **Threshold Filtering**: Identifies scenes above the motion threshold
3. **Timestamp Extraction**: Parses ffmpeg output for scene change times

### 3. Clip Generation
1. **Timestamp Filtering**: Removes overlapping highlights
2. **Concurrent Processing**: Extracts multiple clips simultaneously
3. **Quality Validation**: Ensures generated clips are valid

## Output

The script creates highlight clips with the following naming convention:
```
original_video_highlight_001.mp4
original_video_highlight_002.mp4
original_video_highlight_003.mp4
...
```

Each clip is `--clip-duration` seconds long (default: 15 seconds) and starts at the HiLight tag timestamp.

## Logging

The script provides comprehensive logging:
- **Console output**: Real-time progress and status
- **Log file**: Detailed information saved to `highlight_extraction.log`
- **Debug mode**: Use Python's `-v` flag for verbose output

## Troubleshooting

### No HiLight Tags Found
- **Cause**: Video doesn't contain HiLight tags or wasn't recorded with a compatible GoPro
- **Solution**: Script automatically falls back to motion analysis

### ffmpeg Not Found
- **Cause**: ffmpeg/ffprobe not installed or not in PATH
- **Solution**: Install ffmpeg using the instructions above

### Poor Motion Detection Results
- **Cause**: Motion threshold too high/low for your content
- **Solution**: Adjust `--motion-threshold` (lower = more sensitive)

### Overlapping Clips
- **Cause**: HiLight tags created too close together
- **Solution**: Increase `--min-gap` value

## Performance Tips

1. **Use HiLight Tags**: Much faster than motion analysis
2. **Adjust Workers**: Increase `--max-workers` on powerful systems
3. **SSD Storage**: Use fast storage for input/output directories
4. **Batch Processing**: Process multiple videos in one run

## Technical Details

### MP4 Box Structure
The script parses MP4 files using the following hierarchy:
```
ftyp (file type)
moov (movie metadata)
  └── udta (user data)
      ├── GPMF (GoPro Metadata Format - HERO6+)
      └── HMMT (HiLight Metadata - pre-HERO6)
```

### Supported Formats
- **Input**: MP4, MOV files from GoPro cameras
- **Output**: MP4 files with H.264 video codec
- **Metadata**: GPMF (newer) and HMMT (older) formats

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License.

## Credits

- GoPro HiLight tag parsing inspired by [icegoogles/GoPro-Highlight-Parser](https://github.com/icegoogles/GoPro-Highlight-Parser)
- MP4 box parsing techniques from various open-source GPMF parsers
- Motion analysis using ffmpeg's scene detection capabilities
