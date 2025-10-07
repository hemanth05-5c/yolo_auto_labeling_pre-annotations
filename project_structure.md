# YOLO OBB + Label Studio Pipeline

## Project Structure
```
yolo-label-studio-pipeline/
├── config/
│   ├── config.yaml                 # Main configuration file
│   ├── gcs_credentials.json        # GCS service account key
│   └── label_studio_config.yaml    # Label Studio project settings
├── scripts/
│   ├── 1_download_from_gcs.py      # Download images from GCS
│   ├── 2_yolo_inference.py         # Run YOLO OBB model inference
│   ├── 3_convert_to_predictions.py # Convert YOLO OBB to Label Studio format
│   ├── 4_upload_to_labelstudio.py  # Upload predictions to Label Studio
│   └── main_pipeline.py            # Main orchestration script
├── utils/
│   ├── __init__.py
│   ├── gcs_handler.py              # GCS operations utilities
│   ├── yolo_utils.py               # YOLO model utilities
│   ├── format_converter.py         # Format conversion utilities
│   └── labelstudio_client.py       # Label Studio API client
├── models/
│   └── yolo_obb_model.pt           # YOLO OBB model weights
├── data/
│   ├── raw_images/                 # Downloaded images from GCS
│   ├── yolo_results/               # YOLO inference results
│   └── predictions/                # Label Studio predictions JSON
├── logs/
│   └── pipeline.log                # Pipeline execution logs
├── requirements.txt                # Python dependencies
├── setup.py                        # Package setup
├── README.md                       # Setup and usage instructions
└── run_pipeline.sh                 # Bash script to run complete pipeline
```

## Pipeline Flow

1. **Download Phase**: Download images from GCS bucket to local storage
2. **Inference Phase**: Run YOLO OBB model on downloaded images
3. **Conversion Phase**: Convert YOLO OBB results to Label Studio predictions format
4. **Upload Phase**: Upload predictions to Label Studio project
5. **Cleanup Phase**: Optional cleanup of temporary files

## Key Features

- Batch processing of images from GCS
- Support for YOLO OBB (Oriented Bounding Box) format
- Automatic conversion to Label Studio predictions
- Configurable pipeline parameters
- Comprehensive logging and error handling
- Resume capability for interrupted processes
- Parallel processing support

## Configuration Management

- YAML-based configuration for easy parameter tuning
- Separate credentials management for security
- Environment-specific settings support

## Dependencies

- Google Cloud Storage SDK
- Ultralytics YOLO
- Label Studio SDK
- OpenCV, NumPy, Pillow for image processing
- PyYAML for configuration management 