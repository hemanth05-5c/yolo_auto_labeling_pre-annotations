"""
Format Converter Utilities
Converts YOLO OBB results to Label Studio predictions format
"""

import os
import logging
import uuid
from typing import List, Dict, Any, Optional
from pathlib import Path


class YOLOToLabelStudioConverter:
    """Convert YOLO OBB results to Label Studio predictions format"""
    
    def __init__(self, class_mapping: Dict[int, str], model_version: str = "yolo_obb_v1"):
        """
        Initialize converter
        
        Args:
            class_mapping: Mapping from YOLO class IDs to Label Studio labels
            model_version: Version string for the model
        """
        self.class_mapping = class_mapping
        self.model_version = model_version
        self.logger = logging.getLogger(__name__)
    
    def convert_single_result(
        self, 
        yolo_result: Dict[str, Any],
        image_url_prefix: str = "/data/local-files/?d="
    ) -> Dict[str, Any]:
        """
        Convert single YOLO result to Label Studio format
        
        Args:
            yolo_result: YOLO detection result
            image_url_prefix: URL prefix for accessing images in Label Studio
            
        Returns:
            Label Studio task with predictions
        """
        try:
            image_path = yolo_result['image_path']
            image_width = yolo_result['image_width']
            image_height = yolo_result['image_height']
            detections = yolo_result['detections']
            
            # Create image URL for Label Studio
            # Extract relative path from full path
            image_filename = os.path.basename(image_path)
            
            # Handle directory structure in GCS downloads
            if 'raw_images' in image_path:
                # Extract path after raw_images directory
                parts = Path(image_path).parts
                raw_images_idx = parts.index('raw_images')
                relative_path = '/'.join(parts[raw_images_idx + 1:])
            else:
                relative_path = image_filename
            
            image_url = f"{image_url_prefix}{relative_path}"
            
            # Convert detections to Label Studio format
            prediction_results = []
            
            for detection in detections:
                # Convert YOLO class ID to Label Studio label
                class_id = detection['class_id']
                if class_id in self.class_mapping:
                    label = self.class_mapping[class_id]
                else:
                    self.logger.warning(f"Unknown class ID: {class_id}")
                    label = f"class_{class_id}"
                
                # Convert OBB corners to normalized polygon points
                corners = detection['corners']
                normalized_points = self._normalize_obb_points(
                    corners, image_width, image_height
                )
                
                # Create Label Studio polygon result
                result_id = str(uuid.uuid4())
                
                prediction_result = {
                    "id": result_id,
                    "type": "polygonlabels",
                    "from_name": "ai_supportive_devices",
                    "to_name": "image",
                    "original_width": image_width,
                    "original_height": image_height,
                    "image_rotation": 0,
                    "value": {
                        "points": normalized_points,
                        "polygonlabels": [label]
                    },
                    "score": detection['confidence']
                }
                
                prediction_results.append(prediction_result)
            
            # Create Label Studio task format
            task = {
                "data": {
                    "image": image_url
                },
                "predictions": [{
                    "model_version": self.model_version,
                    "score": self._calculate_overall_score(detections),
                    "result": prediction_results
                }]
            }
            
            return task
            
        except Exception as e:
            self.logger.error(f"Error converting YOLO result: {e}")
            return None
    
    def convert_batch_results(
        self,
        yolo_results: List[Dict[str, Any]],
        image_url_prefix: str = "/data/local-files/?d="
    ) -> List[Dict[str, Any]]:
        """
        Convert batch of YOLO results to Label Studio format
        
        Args:
            yolo_results: List of YOLO detection results
            image_url_prefix: URL prefix for accessing images
            
        Returns:
            List of Label Studio tasks with predictions
        """
        tasks = []
        
        for yolo_result in yolo_results:
            task = self.convert_single_result(yolo_result, image_url_prefix)
            if task:
                tasks.append(task)
            else:
                self.logger.warning(f"Failed to convert result for {yolo_result.get('image_path', 'unknown')}")
        
        self.logger.info(f"Converted {len(tasks)} tasks from {len(yolo_results)} YOLO results")
        return tasks
    
    def _normalize_obb_points(
        self, 
        corners: List[List[float]], 
        image_width: int, 
        image_height: int
    ) -> List[List[float]]:
        """
        Normalize OBB corner points to percentage coordinates (0-100)
        
        Args:
            corners: List of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            image_width: Image width in pixels
            image_height: Image height in pixels
            
        Returns:
            Normalized points as percentages
        """
        normalized_points = []
        
        for corner in corners:
            x_percent = (corner[0] / image_width) * 100
            y_percent = (corner[1] / image_height) * 100
            
            # Clamp values to 0-100 range
            x_percent = max(0, min(100, x_percent))
            y_percent = max(0, min(100, y_percent))
            
            normalized_points.append([x_percent, y_percent])
        
        return normalized_points
    
    def _calculate_overall_score(self, detections: List[Dict[str, Any]]) -> float:
        """
        Calculate overall prediction score for the image
        
        Args:
            detections: List of detections
            
        Returns:
            Overall score (average confidence)
        """
        if not detections:
            return 0.0
        
        total_confidence = sum(d['confidence'] for d in detections)
        return total_confidence / len(detections)
    
    def save_predictions(
        self, 
        tasks: List[Dict[str, Any]], 
        output_file: str
    ):
        """
        Save Label Studio predictions to JSON file
        
        Args:
            tasks: List of Label Studio tasks
            output_file: Output file path
        """
        import json
        
        try:
            # Create output directory
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Save predictions
            with open(output_file, 'w') as f:
                json.dump(tasks, f, indent=2)
            
            self.logger.info(f"Saved {len(tasks)} predictions to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving predictions: {e}")
            raise


class AlternativeFormatConverter:
    """Alternative converter for different Label Studio configurations"""
    
    @staticmethod
    def convert_to_rectangle_labels(
        yolo_results: List[Dict[str, Any]],
        class_mapping: Dict[int, str],
        model_version: str = "yolo_obb_v1",
        image_url_prefix: str = "/data/local-files/?d="
    ) -> List[Dict[str, Any]]:
        """
        Convert YOLO OBB to rectangle labels format
        (For when using RectangleLabels instead of PolygonLabels)
        """
        tasks = []
        
        for yolo_result in yolo_results:
            try:
                image_path = yolo_result['image_path']
                image_width = yolo_result['image_width']
                image_height = yolo_result['image_height']
                detections = yolo_result['detections']
                
                # Create image URL
                image_filename = os.path.basename(image_path)
                if 'raw_images' in image_path:
                    parts = Path(image_path).parts
                    raw_images_idx = parts.index('raw_images')
                    relative_path = '/'.join(parts[raw_images_idx + 1:])
                else:
                    relative_path = image_filename
                
                image_url = f"{image_url_prefix}{relative_path}"
                
                # Convert to axis-aligned rectangles
                prediction_results = []
                
                for detection in detections:
                    class_id = detection['class_id']
                    label = class_mapping.get(class_id, f"class_{class_id}")
                    
                    # Convert OBB to axis-aligned bounding box
                    corners = detection['corners']
                    x_coords = [c[0] for c in corners]
                    y_coords = [c[1] for c in corners]
                    
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    # Normalize to percentages
                    x_norm = (x_min / image_width) * 100
                    y_norm = (y_min / image_height) * 100
                    width_norm = ((x_max - x_min) / image_width) * 100
                    height_norm = ((y_max - y_min) / image_height) * 100
                    
                    result_id = str(uuid.uuid4())
                    
                    prediction_result = {
                        "id": result_id,
                        "type": "rectanglelabels",
                        "from_name": "ai_supportive_devices",
                        "to_name": "image",
                        "original_width": image_width,
                        "original_height": image_height,
                        "image_rotation": 0,
                        "value": {
                            "rotation": 0,
                            "x": x_norm,
                            "y": y_norm,
                            "width": width_norm,
                            "height": height_norm,
                            "rectanglelabels": [label]
                        },
                        "score": detection['confidence']
                    }
                    
                    prediction_results.append(prediction_result)
                
                # Create task
                task = {
                    "data": {
                        "image": image_url
                    },
                    "predictions": [{
                        "model_version": model_version,
                        "score": sum(d['confidence'] for d in detections) / len(detections) if detections else 0.0,
                        "result": prediction_results
                    }]
                }
                
                tasks.append(task)
                
            except Exception as e:
                logging.getLogger(__name__).error(f"Error converting result: {e}")
                continue
        
        return tasks


def validate_predictions(tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Validate Label Studio predictions format
    
    Args:
        tasks: List of Label Studio tasks
        
    Returns:
        Validation results
    """
    validation_results = {
        'valid_tasks': 0,
        'invalid_tasks': 0,
        'errors': [],
        'warnings': []
    }
    
    required_fields = ['data', 'predictions']
    required_prediction_fields = ['model_version', 'result']
    
    for i, task in enumerate(tasks):
        try:
            # Check required task fields
            for field in required_fields:
                if field not in task:
                    validation_results['errors'].append(f"Task {i}: Missing field '{field}'")
                    continue
            
            # Check data field
            if 'image' not in task['data']:
                validation_results['errors'].append(f"Task {i}: Missing 'image' in data")
            
            # Check predictions
            predictions = task['predictions']
            if not isinstance(predictions, list) or len(predictions) == 0:
                validation_results['errors'].append(f"Task {i}: Invalid predictions format")
                continue
            
            for j, prediction in enumerate(predictions):
                for field in required_prediction_fields:
                    if field not in prediction:
                        validation_results['errors'].append(
                            f"Task {i}, Prediction {j}: Missing field '{field}'"
                        )
            
            validation_results['valid_tasks'] += 1
            
        except Exception as e:
            validation_results['invalid_tasks'] += 1
            validation_results['errors'].append(f"Task {i}: Validation error - {e}")
    
    return validation_results 