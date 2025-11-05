# Badminton Player-Shuttlecock Tracking System

A comprehensive tracking system that combines **TrackNetV3** (Original Implementation: https://github.com/qaz812345/TrackNetV3) for shuttlecock trajectory prediction and **YOLO11s + ByteTrack** for player detection and tracking. (Y.-J. Chen & Wang, 2024; Khanam & Hussain, 2024; Y. Zhang et al., 2022) The system processes badminton videos to generate output videos with both shuttlecock trajectories and player bounding boxes.

## System Overview

This system implements a two-stage sequential tracking approach:

1. **Stage 1: Shuttlecock Tracking** - Uses TrackNetV3 to predict and track shuttlecock trajectories
2. **Stage 2: Player Tracking** - Uses YOLO11s + ByteTrack to detect and track players with bounding boxes

The system takes a video input and produces two output videos:
- `{video_name}_shuttle_tracked.mp4` - Shuttlecock tracking only
- `{video_name}_player_shuttle_tracked.mp4` - Combined shuttlecock and player tracking

## Key Features

- **TrackNetV3**: Enhanced shuttlecock tracking with trajectory rectification and inpainting
- **YOLO11s + ByteTrack**: Real-time player detection and multi-object tracking
- **Sequential Processing**: Optimized workflow that processes shuttlecock tracking first, then player tracking
- **Dual Output**: Generates both individual and combined tracking videos
- **Robust Performance**: Handles occlusions and complex badminton scenarios

## Installation

### Prerequisites
* **Operating System**: Windows 10/11, Ubuntu 16.04.7 LTS, or macOS
* **Python**: 3.8.7 or higher
* **CUDA**: Compatible GPU with CUDA support (recommended for faster processing)

### Setup Instructions

1. **Clone the repository**
    ```bash
    git clone https://github.com/nixonnzh/Badminton-Player-Shuttlecock-Tracker.git
    cd Badminton-Player-Shuttlecock-Tracker
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Download model checkpoints**
    - Download the required model files and place them in the `ckpts/` directory:
      - `TrackNetV3_shuttle_best.pt` - TrackNetV3 model for shuttlecock tracking
      - `InpaintNet_shuttle_best.pt` - InpaintNet model for trajectory rectification
      - `yolov11s_player_best.pt` - YOLO11s model for player detection

## Quick Start

### Basic Usage

Run the complete player-shuttlecock tracking system on a video:

```bash
python main.py --video_file test.mp4 --tracknet_file ckpts/TrackNetV3_shuttle_best.pt --inpaintnet_file ckpts/InpaintNet_shuttle_best.pt
```

This command will:
1. **Process shuttlecock tracking** using TrackNetV3
2. **Generate intermediate video** with shuttlecock trajectory
3. **Process player tracking** using YOLO11s + ByteTrack
4. **Generate final combined video** with both shuttlecock and player tracking

### Output Files

The system generates the following output files in the `prediction/` directory:
- `{video_name}_ball.csv` - Shuttlecock trajectory coordinates (CSV format)
- `{video_name}_shuttle_tracked.mp4` - Video with shuttlecock tracking only
- `{video_name}_player_shuttle_tracked.mp4` - Video with both shuttlecock and player tracking

### Advanced Options

```bash
python main.py \
    --video_file test.mp4 \
    --tracknet_file ckpts/TrackNetV3_shuttle_best.pt \
    --inpaintnet_file ckpts/InpaintNet_shuttle_best.pt \
    --save_dir prediction \
    --batch_size 16 \
    --traj_len 8 \
    --large_video \
    --video_range 0,60
```

**Parameters:**
- `--video_file`: Path to input video file
- `--tracknet_file`: Path to TrackNetV3 model checkpoint
- `--inpaintnet_file`: Path to InpaintNet model checkpoint (optional)
- `--save_dir`: Output directory (default: `prediction`)
- `--batch_size`: Batch size for inference (default: 16)
- `--traj_len`: Length of trajectory trail to draw (default: 8)
- `--large_video`: Enable for large videos to prevent memory issues
- `--video_range`: Process specific time range (start_sec,end_sec)
- `--max_sample_num`: Maximum frames for background estimation (default: 1800)

### For Large Videos

For processing large videos that might cause memory issues:

```bash
python main.py \
    --video_file large_video.mp4 \
    --tracknet_file ckpts/TrackNetV3_shuttle_best.pt \
    --inpaintnet_file ckpts/InpaintNet_shuttle_best.pt \
    --large_video \
    --max_sample_num 1000 \
    --video_range 0,300
```

## System Workflow

The player-shuttlecock tracking system follows a sequential two-stage approach:

### Stage 1: Shuttlecock Tracking (TrackNetV3)
1. **Input Processing**: Load video frames and preprocess for TrackNetV3
2. **Background Estimation**: Generate median background image for better tracking
3. **Heatmap Prediction**: Use TrackNetV3 to predict shuttlecock location heatmaps
4. **Trajectory Rectification**: Apply InpaintNet to fix occluded or missing detections
5. **Coordinate Extraction**: Convert heatmaps to precise (x, y) coordinates
6. **Video Generation**: Create intermediate video with shuttlecock trajectory

### Stage 2: Player Tracking (YOLO11s + ByteTrack)
1. **Frame Loading**: Load frames from the shuttlecock-tracked video
2. **Player Detection**: Use YOLO11s to detect players in each frame
3. **Multi-Object Tracking**: Apply ByteTrack for consistent player ID assignment
4. **Bounding Box Drawing**: Draw player bounding boxes with tracking IDs
5. **Combined Video**: Generate final video with both shuttlecock and player tracking

### Key Components

- **TrackNetV3**: Deep learning model for shuttlecock trajectory prediction
- **InpaintNet**: Trajectory rectification module for handling occlusions
- **YOLO11s**: Real-time object detection for player identification
- **ByteTrack**: Multi-object tracking algorithm for consistent player IDs
- **Video Processing**: OpenCV-based video I/O and frame processing

## Troubleshooting

### Common Issues

**1. CUDA out of memory**
```bash
# Solution: Reduce batch size or use CPU
python main.py --video_file test.mp4 --batch_size 4  # Reduce batch size
# Or set CUDA_VISIBLE_DEVICES="" to use CPU
```

**2. Video processing errors**
```bash
# Solution: Check video format and codec support
ffmpeg -i input_video.mp4 -c:v libx264 -c:a aac output_video.mp4
```

**3. Model loading errors**
```bash
# Solution: Verify model file paths and formats
ls ckpts/*.pt  # Check if all model files exist
```

### Performance Optimization

**For faster processing:**
- Use GPU acceleration (CUDA)
- Reduce video resolution if possible
- Use `--large_video` flag for memory efficiency
- Adjust `--batch_size` based on available memory

**For better accuracy:**
- Use higher resolution videos
- Ensure good lighting conditions
- Use videos with clear shuttlecock visibility
- Train models on domain-specific data

## System Architecture

### File Structure
```
Badminton-Player-Shuttlecock-Tracker/
├── main.py                          # Main execution script
├── model.py                         # TrackNet and InpaintNet model definitions
├── ckpts/                          # Pre-trained model checkpoints
│   ├── TrackNetV3_shuttle_best.pt
│   ├── InpaintNet_shuttle_best.pt
│   └── yolov11s_player_best.pt
├── utils/
│   ├── tracknet/                    # TrackNet utilities
│   │   ├── general.py              # General functions
│   │   ├── dataset.py              # Dataset handling
│   │   ├── test.py                 # Testing utilities
│   │   └── visualize.py            # Visualization tools
│   └── bytetrack/                   # ByteTrack utilities
│       ├── yolo_byte_player_tracker.py
│       └── video_utils.py
├── train/                           # Training scripts
│   ├── train.py                    # Model training
│   └── player_detection/           # Player detection training
└── prediction/                     # Output directory
    ├── {video_name}_ball.csv
    ├── {video_name}_tracknet_bytetrack.mp4
    └── {video_name}_combined.mp4
```

### Dependencies
- **PyTorch**: Deep learning framework
- **OpenCV**: Video processing
- **Ultralytics YOLO**: Object detection
- **ByteTrack**: Multi-object tracking
- **NumPy, Pandas**: Data manipulation
- **PIL**: Image processing

## Reference
* **TrackNetV3 Paper**: [Enhancing ShuttleCock Tracking with Augmentations and Trajectory Rectification](https://dl.acm.org/doi/10.1145/3595916.3626370)
* **TrackNetV2**: https://nol.cs.nctu.edu.tw:234/open-source/TrackNetv2
* **Shuttlecock Trajectory Dataset**: https://hackmd.io/@TUIK/rJkRW54cU
* **YOLO11**: https://github.com/ultralytics/ultralytics
* **ByteTrack**: https://github.com/ifzhang/ByteTrack