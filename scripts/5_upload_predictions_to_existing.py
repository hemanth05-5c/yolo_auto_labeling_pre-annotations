#!/usr/bin/env python3
"""
Upload Predictions to Existing Label Studio Tasks
Uploads YOLO predictions as predictions to existing Label Studio tasks (not new tasks)
"""

import os
import sys
import logging
import yaml
import json
import requests
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import time

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_level = getattr(logging, config['logging']['level'].upper())
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = []
    if config['logging']['console_output']:
        handlers.append(logging.StreamHandler())
    if config['logging']['log_file']:
        handlers.append(logging.FileHandler(config['logging']['log_file']))
    
    logging.basicConfig(level=log_level, format=log_format, handlers=handlers)


def load_config(config_path: str = "config/config.yaml") -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing configuration file: {e}")
        sys.exit(1)


def load_task_mapping(raw_images_dir: str) -> Dict[str, str]:
    """
    Load task mapping from previous download step
    
    Args:
        raw_images_dir: Directory where task mapping was saved
        
    Returns:
        Dictionary mapping local_path -> task_id
    """
    mapping_file = os.path.join(raw_images_dir, 'task_mapping.json')
    
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(
            f"Task mapping file not found: {mapping_file}. "
            "Please run the get existing images script first."
        )
    
    with open(mapping_file, 'r') as f:
        return json.load(f)


def load_yolo_results(yolo_results_dir: str) -> List[Dict[str, Any]]:
    """
    Load YOLO detection results
    
    Args:
        yolo_results_dir: Directory containing YOLO results
        
    Returns:
        List of YOLO detection results
    """
    results_file = os.path.join(yolo_results_dir, 'detections.json')
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(
            f"YOLO results not found: {results_file}. "
            "Please run YOLO inference first."
        )
    
    with open(results_file, 'r') as f:
        return json.load(f)


def convert_yolo_to_prediction_format(
    yolo_result: Dict[str, Any],
    task_id: str,
    class_mapping: Dict[int, str],
    model_version: str = "supportive_devices_v1"
) -> Dict[str, Any]:
    """
    Convert single YOLO result to Label Studio prediction format
    
    Args:
        yolo_result: YOLO detection result
        task_id: Label Studio task ID
        class_mapping: Mapping from YOLO class IDs to labels
        model_version: Model version string
        
    Returns:
        Label Studio prediction
    """
    image_width = yolo_result['image_width']
    image_height = yolo_result['image_height']
    detections = yolo_result['detections']
    
    # Convert detections to Label Studio format
    prediction_results = []
    
    for i, detection in enumerate(detections):
        # Convert YOLO class ID to Label Studio label
        class_id = detection['class_id']
        if class_id in class_mapping:
            label = class_mapping[class_id]
        else:
            label = f"class_{class_id}"
        
        # Check if this is OBB (oriented bounding box) or regular bbox
        if 'corners' in detection and detection.get('bbox_type') == 'obb':
            # Handle OBB - convert corners to axis-aligned rectangle
            corners = detection["corners"]
            
            # Find bounding box from corners (min/max x and y)
            x_coords = [corner[0] for corner in corners]
            y_coords = [corner[1] for corner in corners]
            
            min_x = min(x_coords)
            max_x = max(x_coords)
            min_y = min(y_coords)
            max_y = max(y_coords)
            
            # Convert to percentages
            x_percent = (min_x / image_width) * 100
            y_percent = (min_y / image_height) * 100
            width_percent = ((max_x - min_x) / image_width) * 100
            height_percent = ((max_y - min_y) / image_height) * 100
            
            # Clamp values
            x_percent = max(0, min(100, x_percent))
            y_percent = max(0, min(100, y_percent))
            width_percent = max(0, min(100 - x_percent, width_percent))
            height_percent = max(0, min(100 - y_percent, height_percent))
            
            # Create Label Studio rectangle result for OBB
            prediction_result = {
                "id": f"prediction_{i}",
                "type": "rectanglelabels",
                "from_name": "ai_supportive_devices",
                "to_name": "image",
                "original_width": image_width,
                "original_height": image_height,
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "x": x_percent,
                    "y": y_percent,
                    "width": width_percent,
                    "height": height_percent,
                    "rectanglelabels": [label]
                },
                "score": detection["confidence"]
            }
            
        else:
            # Handle regular bounding box (assume bbox format: [x, y, width, height])
            if 'bbox' in detection:
                bbox = detection['bbox']
                x, y, w, h = bbox
            else:
                # If no bbox, try to extract from corners or skip
                continue
            
            # Normalize to percentages
            x_percent = (x / image_width) * 100
            y_percent = (y / image_height) * 100
            width_percent = (w / image_width) * 100
            height_percent = (h / image_height) * 100
            
            # Clamp values
            x_percent = max(0, min(100, x_percent))
            y_percent = max(0, min(100, y_percent))
            width_percent = max(0, min(100 - x_percent, width_percent))
            height_percent = max(0, min(100 - y_percent, height_percent))
            
            # Create Label Studio rectangle result
            prediction_result = {
                "id": f"prediction_{i}",
                "type": "rectanglelabels",
                "from_name": "ai_supportive_devices",
                "to_name": "image",
                "original_width": image_width,
                "original_height": image_height,
                "image_rotation": 0,
                "value": {
                    "rotation": 0,
                    "x": x_percent,
                    "y": y_percent,
                    "width": width_percent,
                    "height": height_percent,
                    "rectanglelabels": [label]
                },
                "score": detection['confidence']
            }
        
        prediction_results.append(prediction_result)
    
    # Calculate overall prediction score
    overall_score = sum(d['confidence'] for d in detections) / len(detections) if detections else 0.0
    
    # Create Label Studio prediction
    prediction = {
        "task": int(task_id),
        "model_version": model_version,
        "score": overall_score,
        "result": prediction_results
    }
    
    return prediction


class PredictionUploader:
    """Upload predictions to existing Label Studio tasks"""
    
    def __init__(self, url: str, api_key: str):
        """
        Initialize prediction uploader
        
        Args:
            url: Label Studio server URL
            api_key: API key for authentication
        """
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
        
        # Setup headers
        self.headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def upload_single_prediction(self, prediction: Dict[str, Any]) -> bool:
        """
        Upload a single prediction to Label Studio
        
        Args:
            prediction: Prediction data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.url}/api/predictions/"
            response = requests.post(url, json=prediction, headers=self.headers)
            
            if response.status_code in [200, 201]:
                return True
            else:
                self.logger.error(f"Failed to upload prediction for task {prediction.get('task')}: "
                                f"{response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error uploading prediction for task {prediction.get('task')}: {e}")
            return False
    
    def upload_predictions_batch(
        self,
        predictions: List[Dict[str, Any]],
        delay_between_requests: float = 0.1
    ) -> Dict[str, Any]:
        """
        Upload multiple predictions with rate limiting
        
        Args:
            predictions: List of predictions to upload
            delay_between_requests: Delay between API requests
            
        Returns:
            Upload results summary
        """
        results = {
            'successful': 0,
            'failed': 0,
            'errors': []
        }
        
        self.logger.info(f"Uploading {len(predictions)} predictions...")
        
        with tqdm(total=len(predictions), desc="Uploading predictions") as pbar:
            for i, prediction in enumerate(predictions):
                success = self.upload_single_prediction(prediction)
                
                if success:
                    results['successful'] += 1
                else:
                    results['failed'] += 1
                    results['errors'].append(f"Task {prediction.get('task', 'unknown')}")
                
                pbar.update(1)
                
                # Rate limiting
                if delay_between_requests > 0 and i < len(predictions) - 1:
                    time.sleep(delay_between_requests)
        
        self.logger.info(f"Upload completed: {results['successful']} successful, {results['failed']} failed")
        return results


def upload_predictions_to_existing(config: dict) -> bool:
    """
    Upload predictions to existing Label Studio tasks
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Load task mapping (from previous download)
        logger.info("Loading task mapping...")
        task_mapping = load_task_mapping(config['paths']['raw_images'])
        logger.info(f"Loaded mapping for {len(task_mapping)} images")
        
        # Load YOLO results
        logger.info("Loading YOLO results...")
        yolo_results = load_yolo_results(config['paths']['yolo_results'])
        logger.info(f"Loaded {len(yolo_results)} YOLO results")
        
        # Convert YOLO results to predictions
        logger.info("Converting YOLO results to Label Studio predictions...")
        predictions = []
        
        for yolo_result in yolo_results:
            image_path = yolo_result['image_path']
            
            # Find corresponding task ID
            task_id = None
            for local_path, tid in task_mapping.items():
                if os.path.basename(local_path) == os.path.basename(image_path):
                    task_id = tid
                    break
            
            if task_id is None:
                logger.warning(f"No task ID found for image: {image_path}")
                continue
            
            # Convert to prediction format
            prediction = convert_yolo_to_prediction_format(
                yolo_result=yolo_result,
                task_id=task_id,
                class_mapping=config['class_mapping'],
                model_version=config['label_studio']['model_version']
            )
            
            predictions.append(prediction)
        
        logger.info(f"Created {len(predictions)} predictions")
        
        if not predictions:
            logger.error("No predictions created")
            return False
        
        # Upload predictions
        uploader = PredictionUploader(
            url=config['label_studio']['url'],
            api_key=config['label_studio']['api_key']
        )
        
        results = uploader.upload_predictions_batch(predictions)
        
        # Save upload summary
        summary = {
            'total_predictions': len(predictions),
            'successful_uploads': results['successful'],
            'failed_uploads': results['failed'],
            'errors': results['errors'],
            'upload_type': 'predictions_to_existing_tasks'
        }
        
        summary_file = os.path.join(config['paths']['predictions'], 'upload_summary.json')
        os.makedirs(config['paths']['predictions'], exist_ok=True)
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Upload summary saved to: {summary_file}")
        
        return results['successful'] > 0
        
    except Exception as e:
        logger.error(f"Error uploading predictions: {e}")
        return False


def main():
    """Main entry point"""
    print("Upload Predictions to Existing Label Studio Tasks")
    print("=" * 55)
    
    # Load configuration
    config = load_config()
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Check if upload predictions stage is enabled
    if not config['stages'].get('upload_predictions', False):
        logger.info("Upload predictions stage is disabled")
        return
    
    # Check if existing project mode is enabled
    if not config.get('existing_project', {}).get('enabled', False):
        logger.info("Existing project mode is disabled")
        return
    
    try:
        success = upload_predictions_to_existing(config)
        
        if success:
            logger.info("Predictions uploaded successfully!")
            print("\n✅ Predictions uploaded successfully!")
            
            # Show Label Studio URL
            project_url = f"{config['label_studio']['url']}/projects/{config['label_studio']['project_id']}"
            print(f"🔗 Label Studio Project: {project_url}")
            print(f"\n🎯 Next steps:")
            print(f"   1. Open Label Studio in your browser")
            print(f"   2. Review the predictions on your existing tasks")
            print(f"   3. Correct any inaccuracies")
            print(f"   4. Export the corrected annotations")
            
        else:
            logger.error("Upload failed!")
            print("\n❌ Upload failed!")
            print("Check the logs for detailed error information")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Upload process failed: {e}")
        print(f"\n❌ Upload process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 