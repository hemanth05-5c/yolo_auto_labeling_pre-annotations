# Label Studio Setup Guide

## Setting up Label Studio Labelling Interface for Object Detection

### 1. Label Studio Installation & Setup

```bash
# Install Label Studio
pip install label-studio

# Start Label Studio
label-studio start

# Access at: http://localhost:8080
```

### 2. Project Configuration

#### Create New Project:
1. Go to Label Studio (http://localhost:8080)
2. Click "Create Project"
3. Enter project name (e.g., "Medical Device Detection")
4. Choose "Computer Vision" → "Object Detection with Bounding Boxes"

#### Import Your Images:
1. Go to "Data Import" tab
2. Upload your images or connect to cloud storage
3. For GCS integration, use the GCS import option

### 3. Labelling Interface Configuration

Use this XML configuration for your labelling interface:

```xml
<View>
  <Image name="image" value="$image"/>
  
  <RectangleLabels name="label" toName="image">
    <Label value="NG Tube" background="red"/>
    <Label value="Pacemaker" background="blue"/>
    <Label value="CV Line" background="green"/>
    <Label value="Tracheostomy" background="yellow"/>
    <Label value="ICD Tube" background="purple"/>
    <Label value="ET Tube" background="orange"/>
    <Label value="Chemoport" background="pink"/>
    <Label value="Cardiac valve" background="brown"/>
    <Label value="Sternal Sutures" background="gray"/>
    <Label value="Spinal fusion" background="cyan"/>
    <Label value="chest leads" background="magenta"/>
    <Label value="Pigtail Catheter" background="lime"/>
  </RectangleLabels>
</View>
```

### 4. Advanced Interface Configuration (Optional)

For more detailed annotations with confidence scores and additional metadata:

```xml
<View>
  <Header value="Medical Device Detection"/>
  
  <Image name="image" value="$image" zoom="true" zoomControl="true"/>
  
  <RectangleLabels name="bbox" toName="image" strokeWidth="3" canRotate="true">
    <Label value="NG Tube" background="#FF6B6B" hotkey="1"/>
    <Label value="Pacemaker" background="#4ECDC4" hotkey="2"/>
    <Label value="CV Line" background="#45B7D1" hotkey="3"/>
    <Label value="Tracheostomy" background="#96CEB4" hotkey="4"/>
    <Label value="ICD Tube" background="#FFEAA7" hotkey="5"/>
    <Label value="ET Tube" background="#DDA0DD" hotkey="6"/>
    <Label value="Chemoport" background="#98D8C8" hotkey="7"/>
    <Label value="Cardiac valve" background="#F7DC6F" hotkey="8"/>
    <Label value="Sternal Sutures" background="#BB8FCE" hotkey="9"/>
    <Label value="Spinal fusion" background="#85C1E9" hotkey="0"/>
    <Label value="chest leads" background="#F8C471" hotkey="q"/>
    <Label value="Pigtail Catheter" background="#82E0AA" hotkey="w"/>
  </RectangleLabels>
  
  <Choices name="visibility" toName="image" choice="single">
    <Choice value="Clear"/>
    <Choice value="Partially Obscured"/>
    <Choice value="Difficult to See"/>
  </Choices>
  
  <Rating name="confidence" toName="image" maxRating="5" icon="star" size="medium"/>
  
  <TextArea name="notes" toName="image" placeholder="Additional notes about this image..." rows="3"/>
</View>
```

### 5. API Configuration

Update your `config/config.yaml` with your Label Studio details:

```yaml
label_studio:
  url: "http://localhost:8080"  # Or your Label Studio URL
  api_key: "your-api-key-here"  # Get from Account & Settings > Access Token
  project_id: 123  # Your project ID (visible in URL)
  model_version: "supportive_devices_v1"
  upload_batch_size: 100
```

### 6. Getting Your API Key

1. Go to Label Studio
2. Click your profile (top right)
3. Go to "Account & Settings"
4. Click "Access Token" tab
5. Copy your token and paste it in the config

### 7. Finding Your Project ID

1. Open your Label Studio project
2. Look at the URL: `http://localhost:8080/projects/123/data`
3. The number after `/projects/` is your project ID (123 in this example)

### 8. Pre-annotation Workflow

Once you run the pipeline:

1. **Images Download**: All images from your project will be downloaded locally
2. **YOLO Inference**: Object detection will run on all images
3. **Predictions Upload**: Results will appear as pre-annotations in Label Studio
4. **Manual Review**: You can then review, edit, and approve the predictions

### 9. Tips for Efficient Labelling

- **Use Hotkeys**: Each label has a hotkey for faster annotation
- **Zoom**: Use zoom controls for detailed inspection
- **Batch Processing**: Review predictions in batches
- **Quality Control**: Use the confidence rating to mark difficult cases
- **Notes**: Add comments for edge cases or unusual findings

### 10. Troubleshooting

**No images showing up?**
- Check your image URLs in the tasks
- Ensure images are accessible from Label Studio
- Verify your API key and project ID

**Predictions not uploading?**
- Check API permissions
- Verify project ID matches
- Check logs for detailed error messages

**Performance issues?**
- Reduce batch sizes in config
- Use fewer parallel workers
- Consider processing in smaller chunks

### 11. Data Format

Your images should be accessible via URLs. The pipeline supports:
- Google Cloud Storage URLs (`gs://bucket/path/image.jpg`)
- Direct HTTP URLs (`https://example.com/image.jpg`)
- Local file paths (if Label Studio can access them)

The pipeline will automatically:
- Download images from GCS (with authentication)
- Run YOLO inference
- Convert to Label Studio prediction format
- Upload as pre-annotations for review 