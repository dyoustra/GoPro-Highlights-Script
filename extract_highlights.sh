#!/bin/bash

# Automated GoPro Highlight Extraction Script
# Usage: ./extract_highlights.sh /path/to/gopro/videos /path/to/output

INPUT_DIR="$1"
OUTPUT_DIR="$2"
CLIP_DURATION=15  # seconds
MOTION_THRESHOLD=0.2  # adjust based on your footage
MIN_GAP_SECONDS=10  # minimum gap between highlights to avoid duplicates

# Progress bar function
show_progress() {
    local current=$1
    local total=$2
    local width=50
    local percentage=$((current * 100 / total))
    local filled=$((width * current / total))
    local empty=$((width - filled))
    
    printf "\rProgress: ["
    printf "%${filled}s" | tr ' ' '='
    printf "%${empty}s" | tr ' ' '-'
    printf "] %d%% (%d/%d)" "$percentage" "$current" "$total"
}

# Spinner function for analysis phases
show_spinner() {
    local pid=$1
    local message=$2
    local spin='⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'
    local i=0
    
    while kill -0 $pid 2>/dev/null; do
        i=$(((i+1) % 10))
        printf "\r${spin:$i:1} $message"
        sleep 0.1
    done
    printf "\r✓ $message\n"
}

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to extract highlights from a single video
extract_highlights() {
    local input_file="$1"
    local base_name=$(basename "$input_file" .MP4)
    local temp_dir="/tmp/highlight_extraction_$$"
    
    mkdir -p "$temp_dir"
    
    echo "Processing: $input_file"
    
    # Get video duration for progress calculation
    local duration=$(ffprobe -v quiet -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$input_file")
    duration=${duration%.*}  # Remove decimal part
    
    # Step 1: Analyze motion vectors to find high-action scenes
    echo "🔍 Analyzing motion vectors..."
    ffmpeg -i "$input_file" -vf "select='gt(scene,0.3)',metadata=print:file=${temp_dir}/scenes.txt" -f null - 2>/dev/null &
    show_spinner $! "Analyzing motion vectors"
    
    # Step 2: Use motion detection to find exciting moments
    echo "🎬 Detecting motion events..."
    ffmpeg -i "$input_file" -vf "select='gt(scene,${MOTION_THRESHOLD})',showinfo" -f null - 2>"${temp_dir}/motion_analysis.txt" &
    show_spinner $! "Detecting motion events"
    
    # Step 3: Extract timestamps of high motion
    grep -o "pts_time:[0-9]*\.[0-9]*" "${temp_dir}/motion_analysis.txt" | cut -d: -f2 > "${temp_dir}/timestamps.txt"
    
    # Step 4: Filter out duplicate/overlapping timestamps
    local filtered_timestamps="${temp_dir}/filtered_timestamps.txt"
    
    # Sort timestamps and remove duplicates within MIN_GAP_SECONDS of each other
    sort -n "${temp_dir}/timestamps.txt" | awk -v gap="$MIN_GAP_SECONDS" '
    BEGIN { prev = -999 }
    {
        if ($1 - prev > gap) {
            print $1
            prev = $1
        }
    }' > "$filtered_timestamps"
    
    echo "Found $(wc -l < "${temp_dir}/timestamps.txt") motion events, filtered to $(wc -l < "$filtered_timestamps") unique highlights"
    
    # Step 5: Extract 15-second clips at filtered timestamps
    local clip_counter=1
    local total_clips=$(wc -l < "$filtered_timestamps")
    
    echo "🎥 Extracting $total_clips highlight clips..."
    
    while IFS= read -r timestamp; do
        if [[ -n "$timestamp" ]]; then
            local output_file="${OUTPUT_DIR}/${base_name}_highlight_${clip_counter}.mp4"
            
            # Show progress
            show_progress "$clip_counter" "$total_clips"
            
            # Extract 15-second clip
            ffmpeg -ss "$timestamp" -i "$input_file" -t $CLIP_DURATION -c copy -avoid_negative_ts make_zero "$output_file" 2>/dev/null
            
            # Check if clip was created successfully
            if [[ -f "$output_file" ]]; then
                ((clip_counter++))
            fi
        fi
    done < "$filtered_timestamps"
    
    printf "\n✓ Extracted $((clip_counter-1)) highlights from $base_name\n"
    
    # Cleanup temp files
    rm -rf "$temp_dir"
}

# Count total videos to process
total_videos=$(find "$INPUT_DIR" -name "*.MP4" -o -name "*.mp4" | wc -l)
current_video=0

echo "🎬 Found $total_videos videos to process"
echo "=================================="

# Process all MP4 files in input directory
for video_file in "$INPUT_DIR"/*.MP4 "$INPUT_DIR"/*.mp4; do
    if [[ -f "$video_file" ]]; then
        ((current_video++))
        echo "📹 Video $current_video of $total_videos: $(basename "$video_file")"
        extract_highlights "$video_file"
        echo ""
    fi
done

echo "🎉 Highlight extraction complete!"
echo "All clips saved to: $OUTPUT_DIR"