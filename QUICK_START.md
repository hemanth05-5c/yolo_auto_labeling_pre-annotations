# Quick Start Guide

## Streamlined Label Studio + YOLO Workflow

This pipeline downloads images from an existing Label Studio project, runs YOLO inference, and uploads predictions back for refinement.

### Prerequisites

1. **Python Environment**
   ```bash
   cd auto_label_pre-annotations
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Google Cloud Authentication** (for private GCS buckets)
   ```bash
   # Set up Google Cloud credentials
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
   ```

3. **Label Studio Setup**
   - Ensure you have an existing Label Studio project with images
   - Get your API key and project ID

### Configuration

Edit `config/config.yaml`:

```yaml
# Label Studio Settings
label_studio:
  url: "https://your-labelstudio-instance.com"
  api_key: "your-api-key"
  project_id: 123  # Your project ID

# YOLO Model
yolo:
  model_path: "/path/to/your/yolo/model.pt"
  confidence_threshold: 0.25

# Stages (all enabled for streamlined workflow)
stages:
  get_existing_images: true
  inference: true
  convert: true
  upload_predictions: true
```

### Run the Pipeline

```bash
# Full pipeline
./run_pipeline.sh

# Or run directly with Python
python scripts/main_pipeline.py

# Dry run to see what will be executed
python scripts/main_pipeline.py --dry-run
```

### Workflow Steps

1. **Get Images**: Downloads images from existing Label Studio project
2. **YOLO Inference**: Runs object detection on downloaded images
3. **Convert**: Converts YOLO results to Label Studio prediction format
4. **Upload**: Uploads predictions to existing Label Studio tasks for refinement

### Output

- Downloaded images: `data/raw_images/`
- YOLO results: `data/yolo_results/`
- Formatted predictions: `data/predictions/`
- Logs: `logs/pipeline.log` 