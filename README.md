# YOLO OBB + Label Studio Pipeline

A complete pipeline for downloading images from Google Cloud Storage, running YOLO OBB (Oriented Bounding Box) inference, converting results to Label Studio format, and uploading for manual correction and review.

## 🎯 Overview

This pipeline enables you to:

1. **Download** images from Google Cloud Storage bucket
2. **Process** images with YOLO OBB model for object detection
3. **Convert** YOLO results to Label Studio predictions format
4. **Upload** predictions to Label Studio for review and correction
5. **Export** corrected annotations for training or evaluation

## 📋 Features

- **Batch Processing**: Handle thousands of images efficiently
- **Multiple Formats**: Support for both polygon and rectangle annotations
- **Parallel Processing**: Configurable parallel downloads and inference
- **Resume Capability**: Resume interrupted pipeline executions
- **Comprehensive Logging**: Detailed logs for monitoring and debugging
- **Flexible Configuration**: YAML-based configuration management
- **Error Handling**: Robust error handling and validation
- **Progress Tracking**: Visual progress bars for all operations

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Google Cloud SDK (for GCS access)
- YOLO OBB model weights
- Label Studio instance (local or cloud)

### Quick Setup

1. **Clone and setup the project:**
```bash
git clone <repository-url>
cd yolo-label-studio-pipeline
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download YOLO OBB model:**
```bash
# Download your YOLO OBB model weights to models/
# Example:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-obb.pt -O models/yolo_obb_model.pt
```

4. **Setup Google Cloud credentials:**
```bash
# Download your GCS service account key
cp /path/to/your/service-account-key.json config/gcs_credentials.json
```

5. **Configure the pipeline:**
```bash
cp config/config.yaml.template config/config.yaml
# Edit config/config.yaml with your settings
```

## ⚙️ Configuration

### Main Configuration (`config/config.yaml`)

```yaml
# Google Cloud Storage Settings
gcs:
  project_id: "your-gcs-project-id"
  bucket_name: "your-images-bucket"
  credentials_path: "config/gcs_credentials.json"
  image_prefix: ""  # Optional: filter images by prefix
  max_images: null  # Limit number of images (null = all)

# YOLO Model Settings
yolo:
  model_path: "models/yolo_obb_model.pt"
  confidence_threshold: 0.25
  iou_threshold: 0.7
  device: "auto"  # "auto", "cpu", "cuda", "mps"

# Class Mapping (YOLO class ID to Label Studio label)
class_mapping:
  0: "person"
  1: "car"
  2: "truck"
  # Add more classes as needed

# Label Studio Settings
label_studio:
  url: "http://localhost:8080"
  api_key: "your-label-studio-api-key"
  project_id: 1
```

### Required Environment Variables

For Label Studio local file serving:

```bash
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/absolute/path/to/data/raw_images
```

## 🚀 Usage

### Complete Pipeline

Run the entire pipeline:

```bash
# Using shell script (recommended)
./run_pipeline.sh

# Or directly with Python
python scripts/main_pipeline.py
```

### Individual Stages

Run specific stages:

```bash
# Download images from GCS
./run_pipeline.sh --stage download

# Run YOLO inference
./run_pipeline.sh --stage inference

# Convert to Label Studio format
./run_pipeline.sh --stage convert

# Upload to Label Studio
./run_pipeline.sh --stage upload
```

### Advanced Options

```bash
# Install dependencies and run
./run_pipeline.sh --install-deps

# Dry run (show what would be executed)
./run_pipeline.sh --dry-run

# Skip validation
./run_pipeline.sh --skip-validation

# Use custom config file
./run_pipeline.sh --config custom_config.yaml
```

## 📂 Project Structure

```
yolo-label-studio-pipeline/
├── config/
│   ├── config.yaml                 # Main configuration
│   ├── gcs_credentials.json        # GCS service account key
│   └── label_studio_config.yaml    # Label Studio project settings
├── scripts/
│   ├── 1_download_from_gcs.py      # Download images from GCS
│   ├── 2_yolo_inference.py         # YOLO model inference
│   ├── 3_convert_to_predictions.py # Convert to Label Studio format
│   ├── 4_upload_to_labelstudio.py  # Upload to Label Studio
│   └── main_pipeline.py            # Main orchestration script
├── utils/
│   ├── gcs_handler.py              # GCS operations
│   ├── yolo_utils.py               # YOLO utilities
│   ├── format_converter.py         # Format conversion
│   └── labelstudio_client.py       # Label Studio API client
├── data/
│   ├── raw_images/                 # Downloaded images
│   ├── yolo_results/               # YOLO inference results
│   └── predictions/                # Label Studio predictions
├── logs/
│   └── pipeline.log                # Pipeline logs
├── models/
│   └── yolo_obb_model.pt           # YOLO model weights
├── requirements.txt                # Python dependencies
├── run_pipeline.sh                 # Main execution script
└── README.md                       # This file
```

## 🔧 Label Studio Setup

### 1. Install and Start Label Studio

```bash
# Install Label Studio
pip install label-studio

# Start Label Studio
label-studio

# Access at http://localhost:8080
```

### 2. Create Project

1. Open Label Studio in your browser
2. Create a new project
3. Use this labeling configuration for oriented bounding boxes:

```xml
<View>
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  <PolygonLabels name="label" toName="image" strokeWidth="3" pointSize="small">
    <Label value="person" background="red"/>
    <Label value="car" background="blue"/>
    <Label value="truck" background="green"/>
    <!-- Add more labels as needed -->
  </PolygonLabels>
</View>
```

### 3. Setup Local File Serving

Configure Label Studio to serve local files:

```bash
# Set environment variables
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true
export LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/absolute/path/to/your/data/raw_images

# Restart Label Studio
label-studio
```

### 4. Get API Key

1. Go to Label Studio settings
2. Navigate to "Account & Settings"
3. Copy your API token
4. Add it to `config/config.yaml`

## 📊 Output Formats

The pipeline generates predictions in multiple formats:

### Polygon Format (Main)
- File: `data/predictions/predictions_polygon.json`
- Uses oriented bounding boxes as polygons
- Best for preserving rotation information

### Rectangle Format (Alternative)
- File: `data/predictions/predictions_rectangle_labels.json`
- Converts OBB to axis-aligned rectangles
- Compatible with standard rectangle labeling

### Sample Format
- File: `data/predictions/predictions_sample.json`
- Contains first 5 predictions for testing

## 🔍 Monitoring and Logging

### Log Files

- **Pipeline logs**: `logs/pipeline.log`
- **Stage-specific logs**: Integrated into main log
- **Error details**: Captured with full stack traces

### Progress Monitoring

- Visual progress bars for all operations
- Real-time status updates
- Detailed summaries after each stage

### Result Files

Each stage creates summary files:

- **Download**: `data/raw_images/download_results.yaml`
- **Inference**: `data/yolo_results/inference_summary.json`
- **Conversion**: `data/predictions/conversion_summary.json`
- **Upload**: `data/predictions/upload_summary.json`

## 🛠️ Troubleshooting

### Common Issues

#### 1. GCS Connection Issues
```bash
# Check credentials
export GOOGLE_APPLICATION_CREDENTIALS=config/gcs_credentials.json
gsutil ls gs://your-bucket-name
```

#### 2. YOLO Model Not Found
```bash
# Download YOLO OBB model
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-obb.pt -O models/yolo_obb_model.pt
```

#### 3. Label Studio Connection Failed
```bash
# Check Label Studio is running
curl http://localhost:8080/api/projects/

# Verify API key in config
```

#### 4. Local File Serving Issues
```bash
# Check environment variables
echo $LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED
echo $LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT

# Test file access
curl "http://localhost:8080/data/local-files/?d=test_image.jpg"
```

### Debug Mode

Enable debug logging:

```yaml
# In config/config.yaml
logging:
  level: "DEBUG"
  console_output: true
```

### Validation

Run validation without execution:

```bash
./run_pipeline.sh --dry-run
```

## 📈 Performance Optimization

### Parallel Processing

Configure parallel workers:

```yaml
processing:
  parallel_downloads: true
  parallel_inference: true
  max_workers: 4
```

### Batch Sizes

Optimize batch sizes for your hardware:

```yaml
yolo:
  batch_size: 16  # Increase for better GPU utilization

label_studio:
  upload_batch_size: 100  # Adjust based on network
```

### Memory Management

For large datasets:

```yaml
gcs:
  download_batch_size: 50  # Limit concurrent downloads
  max_images: 1000  # Process in chunks
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the YOLO implementation
- [Label Studio](https://github.com/heartexlabs/label-studio) for the annotation platform
- [Google Cloud SDK](https://cloud.google.com/sdk) for GCS integration

## 📞 Support

For issues and questions:

1. Check the [troubleshooting section](#-troubleshooting)
2. Review the logs in `logs/pipeline.log`
3. Open an issue on GitHub
4. Check the [Label Studio documentation](https://labelstud.io/guide/)

---

**Happy Annotating! 🎯** 