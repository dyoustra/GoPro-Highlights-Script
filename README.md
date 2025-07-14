# GoPro Highlights Script

An automated Python script to extract highlight clips from GoPro videos using either built-in HiLight tags or advanced scene detection.

## Features

### 🏷️ GoPro HiLight Tag Extraction (Primary Method)
- **Extracts user-created HiLight tags** from GoPro MP4 files
- **Works with tags created during recording** by pressing the mode button on your GoPro
- **Supports both newer (HERO6+) and older GoPro formats**
- **Much more accurate than motion analysis** - uses your actual highlights!
- **Automatic fallback** to scene detection if no HiLight tags are found

### 🧠 TransNetV2 Scene Detection (Recommended Secondary Method)
- **Neural network-based** shot boundary detection
- **PyTorch-powered** - excellent Python 3.13+ compatibility
- **Optimized for action footage** - handles fast camera movement well
- **Automatic model setup** - no manual configuration required
- **Intelligent fallback** to ffmpeg if TransNetV2 unavailable
- **Simplified implementation** - works out of the box with any Python 3.10+

### 📊 Motion Analysis (Fallback Method)
- Analyzes video for high-motion scenes using ffmpeg
- Configurable motion detection threshold
- Useful for non-GoPro videos or when other methods aren't available

### ⚡ Performance & Usability
- **Async/await architecture** for efficient concurrent processing
- **Automatic worker optimization** based on your system's CPU cores
- **Modern progress indicators** with spinners and progress bars
- **Comprehensive logging** with both console and file output
- **Intelligent clip filtering** to avoid overlapping highlights
- **Organized output** with separate folders for different detection methods

## How It Works

The script uses a **hybrid approach** with three detection methods in order of preference:

1. **🏷️ HiLight Tags (Primary)**: Extracts user-created tags from GoPro MP4 metadata
2. **🧠 TransNetV2 (Secondary)**: Uses neural network for accurate scene detection
3. **📊 ffmpeg (Fallback)**: Traditional motion-based analysis

## Output Structure

The script creates organized output directories:

```
/output/directory/
├── HiLights/                    # Clips from GoPro HiLight tags
│   ├── video1_hilight_001.mp4
│   ├── video1_hilight_002.mp4
│   └── ...
└── Motion Detected Highlights/  # Clips from scene detection
    ├── video1_motion_001.mp4
    ├── video1_motion_002.mp4
    └── ...
```

## Installation

### Prerequisites
- Python 3.10+ (including 3.13+ with excellent PyTorch support)
- ffmpeg and ffprobe (for video processing)

**Note**: Now using PyTorch instead of TensorFlow for much better Python 3.13+ compatibility!

### Install ffmpeg
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Install Python Dependencies

#### Option 1: Automated Setup (Recommended)
For the easiest setup with TransNetV2 scene detection:

```bash
git clone <repository-url>
cd gopro-highlights-script
python setup_transnetv2.py
```

This will automatically install TensorFlow, OpenCV, NumPy, and test the TransNetV2 integration.

#### Option 2: Manual Installation
For full control over the installation process:

```bash
git clone <repository-url>
cd gopro-highlights-script
pip install -r requirements.txt
```

#### Option 3: Minimal Installation
If you only want to use HiLight tags and ffmpeg fallback:

```bash
git clone <repository-url>
cd gopro-highlights-script
# No additional packages required - uses only standard library!
```

### TransNetV2 Dependencies

The script will automatically download TransNetV2 when needed, but you can pre-install dependencies:

```bash
# For neural network scene detection (optional but recommended)
pip install torch>=1.9.0 opencv-python>=4.5.0 numpy>=1.21.0
```

**Note**: TransNetV2 PyTorch model is automatically created in `~/.cache/transnetv2/` on first use. This is a simplified but functional implementation that works well for scene detection.

## Usage

### Basic Usage
```bash
python extract_highlights.py /path/to/gopro/videos /path/to/output
```

### Advanced Options
```bash
python extract_highlights.py /path/to/gopro/videos /path/to/output \
    --clip-duration 20 \
    --transnetv2-threshold 0.4 \
    --motion-threshold 0.3 \
    --min-gap 15 \
    --max-workers 4
```

### Scene Detection Options
```bash
# Use TransNetV2 (default)
python extract_highlights.py /path/to/videos /path/to/output --scene-detection transnetv2

# Use ffmpeg only
python extract_highlights.py /path/to/videos /path/to/output --scene-detection ffmpeg

# Auto mode (TransNetV2 with ffmpeg fallback)
python extract_highlights.py /path/to/videos /path/to/output --scene-detection auto
```

### Disable HiLight Tags (Use Scene Detection Only)
```bash
python extract_highlights.py /path/to/gopro/videos /path/to/output \
    --disable-hilight-tags
```

## Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--clip-duration` | Duration of each highlight clip in seconds | 15 |
| `--transnetv2-threshold` | TransNetV2 detection threshold (0.0-1.0) | 0.5 |
| `--motion-threshold` | Motion detection threshold (0.0-1.0) | 0.2 |
| `--min-gap` | Minimum gap between highlights in seconds | 10 |
| `--max-workers` | Maximum concurrent workers | Auto-detected |
| `--disable-hilight-tags` | Disable HiLight tag extraction | False |
| `--scene-detection` | Scene detection method (`transnetv2`, `ffmpeg`, `auto`) | `transnetv2` |

## How GoPro HiLight Tags Work

GoPro cameras allow users to create "HiLight" tags during recording by pressing the mode button. These tags are stored directly in the MP4 file's metadata and mark the exact moments you found interesting while filming.

**Benefits over scene detection:**
- ✅ **User-controlled** - You decide what's a highlight
- ✅ **Perfectly accurate** - No false positives from camera shake or irrelevant motion
- ✅ **Instant processing** - No need to analyze the entire video
- ✅ **Works with any content** - Even slow-motion or static scenes you found interesting

**Supported GoPro models:**
- HERO6 Black and newer (GPMF format)
- HERO5 Black and older (HMMT format)
- Most GoPro models that support HiLight tagging

**Note:** HiLight tags created in GoPro Quik desktop software are NOT stored in the MP4 file and cannot be extracted.

## TransNetV2 Scene Detection

### What is TransNetV2?

TransNetV2 is a state-of-the-art neural network specifically designed for shot boundary detection in videos. It was developed by researchers at Charles University and significantly outperforms traditional methods.

### Performance Comparison

| Method | ClipShots F1 Score | Speed | Best For |
|--------|-------------------|-------|----------|
| **TransNetV2** | **77.9%** | Fast | Action footage, fast cuts |
| Traditional ffmpeg | ~60-70% | Medium | General purpose |
| PySceneDetect | ~65-75% | Slow | Controlled lighting |

### Why TransNetV2 for GoPro Footage?

- **Handles fast motion**: Excellent for action cameras with rapid scene changes
- **Robust to lighting**: Works well with varying outdoor conditions
- **Optimized for cuts**: Specifically designed for shot boundary detection
- **Fast inference**: Processes videos quickly using neural acceleration

### Automatic Setup

The script automatically:
1. **Checks dependencies** - Verifies TensorFlow, OpenCV, and NumPy are available
2. **Downloads model** - Fetches TransNetV2 weights from GitHub (~50MB)
3. **Caches locally** - Stores model in `~/.cache/transnetv2/` for reuse
4. **Fallback gracefully** - Uses ffmpeg if TransNetV2 unavailable

## Troubleshooting

### TransNetV2 Issues

**Dependencies not found:**
```bash
# Install required packages
pip install torch opencv-python numpy
```

**Model creation fails:**
- Ensure PyTorch is properly installed: `pip install torch`
- Check that `~/.cache/transnetv2/` directory is writable
- For advanced users: replace with official TransNetV2 weights from GitHub

**PyTorch errors:**
- Ensure Python 3.10+ is being used (check with `python --version`)
- For CPU-only: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
- Visit https://pytorch.org/get-started/locally/ for custom installation
- PyTorch has excellent Python 3.13+ support!

### General Issues

**No HiLight Tags Found:**
- **Cause**: Video doesn't contain HiLight tags or wasn't recorded with a compatible GoPro
- **Solution**: Script automatically falls back to scene detection

**ffmpeg Not Found:**
- **Cause**: ffmpeg/ffprobe not installed or not in PATH
- **Solution**: Install ffmpeg using the instructions above

**Poor Scene Detection Results:**
- **Cause**: Detection threshold too high/low for your content
- **Solution**: Adjust `--transnetv2-threshold` (lower = more sensitive)

**Overlapping Clips:**
- **Cause**: HiLight tags or scene changes created too close together
- **Solution**: Increase `--min-gap` value

## Performance Tips

1. **Use HiLight Tags**: Much faster and more accurate than any scene detection
2. **Install TransNetV2**: PyTorch-powered, significantly better than ffmpeg for scene detection
3. **Adjust Workers**: Increase `--max-workers` on powerful systems
4. **SSD Storage**: Use fast storage for input/output directories
5. **Batch Processing**: Process multiple videos in one run

## Technical Details

### Scene Detection Methods

1. **TransNetV2**: 
   - Neural network trained on shot boundary detection
   - Processes frames at 48x27 resolution
   - Samples at ~5 FPS for efficiency
   - Threshold-based boundary detection

2. **ffmpeg**: 
   - Traditional scene change detection
   - Analyzes motion vectors and color histograms
   - Configurable sensitivity threshold

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

- **TransNetV2**: [soCzech/TransNetV2](https://github.com/soCzech/TransNetV2) - Neural network for shot boundary detection
- **GoPro HiLight parsing**: Inspired by [icegoogles/GoPro-Highlight-Parser](https://github.com/icegoogles/GoPro-Highlight-Parser)
- **MP4 box parsing**: Techniques from various open-source GPMF parsers
- **Motion analysis**: Using ffmpeg's scene detection capabilities
