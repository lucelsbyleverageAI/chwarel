# Face Detection Testing Framework

## Overview

This framework tests different face detection configurations using DeepFace to help you understand speed vs accuracy trade-offs. It analyzes:

**Settings Tested:**
- Image resolutions (Original, HD, SD)
- Batch processing sizes
- Number of parallel workers
- Face alignment options
- Confidence thresholds

**Outputs:**
- Processing speed measurements
- Detection accuracy vs ground truth
- System resource usage
- Visualization of results

## Quick Start

### 1. Clone repo and install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Test Images
Place test images in `data/test_dataset/`
- Supported: .png, .jpg, .jpeg
- Include variety: different lighting, angles, blur conditions

### 3. Set Ground Truth
Edit `configs/test_configs.yaml`:
```yaml
ground_truth:
  clear2.png: 2    # Image has 2 faces
  group.jpg: 5     # Image has 5 faces
  # Images not listed are assumed to have 1 face
```

### 4. Run Tests
```bash
python src/run_tests.py
```

### 5. View Results
Results are saved in:
- `data/results/detailed_results_[timestamp].csv` - Raw data
- `data/results/results_plot_[timestamp].png` - Visualizations
- `data/logs/face_detection_[timestamp].log` - Execution logs

### run_tests.bat Setup
```bash
# to activate the venv, install dependencies, and run the tests, run run_tests.bat
run_tests.bat
```

## Configuration Options
Example in `test_configs.yaml`:
```yaml
test_configurations:
  baseline:
    name: "Baseline"
    resize_dimensions: [0, 0]    # Original size
    batch_size: 1               # Images per batch
    num_workers: 1              # Parallel processes
    confidence_threshold: 0.5   # Detection threshold
```

## Common Issues

1. **Memory Issues**
   - Reduce batch_size
   - Close other applications

2. **Performance Issues**
   - Check logs for recommendations
   - Adjust num_workers based on CPU cores

## Project Structure
```
face_detection/
├── src/                    # Source code
├── data/
│   ├── test_dataset/      # Your test images
│   ├── results/           # Results and plots
│   └── logs/              # Execution logs
└── configs/
    └── test_configs.yaml  # Configuration file
```