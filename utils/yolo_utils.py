"""
YOLO Utilities
Handles YOLO OBB model inference and result processing
"""

import os
import logging
import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import torch

try:
    from ultralytics import YOLO
except ImportError:
    raise ImportError("ultralytics package required. Install with: pip install ultralytics")


class YOLOOBBProcessor:
    """Handle YOLO OBB (Oriented Bounding Box) model inference"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize YOLO OBB processor
        
        Args:
            model_path: Path to YOLO OBB model weights
            device: Device to run inference on ("auto", "cpu", "cuda", "mps")
        """
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.logger = logging.getLogger(__name__)
        
        # Load model
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            self.logger.info(f"YOLO OBB model loaded successfully on {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def _setup_device(self, device: str) -> str:
        """Setup and validate device for inference"""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def predict_single_image(
        self, 
        image_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_det: int = 1000,
        img_size: int = 640
    ) -> Optional[Dict[str, Any]]:
        """
        Run inference on a single image
        
        Args:
            image_path: Path to input image
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            max_det: Maximum number of detections
            img_size: Input image size for model
            
        Returns:
            Dictionary with detection results or None if failed
        """
        try:
            # Run inference
            results = self.model.predict(
                source=image_path,
                conf=conf_threshold,
                iou=iou_threshold,
                max_det=max_det,
                imgsz=img_size,
                verbose=False
            )
            
            if not results:
                return None
            
            result = results[0]  # Get first (and only) result
            
            # Extract oriented bounding boxes
            detections = []
            if hasattr(result, 'obb') and result.obb is not None:
                # OBB format: [x_center, y_center, width, height, rotation, confidence, class]
                obb_data = result.obb
                
                for i in range(len(obb_data.xyxyxyxy)):
                    # Get 4 corner points of oriented bounding box
                    corners = obb_data.xyxyxyxy[i].cpu().numpy()
                    
                    # Get confidence and class
                    confidence = float(obb_data.conf[i])
                    class_id = int(obb_data.cls[i])
                    
                    detection = {
                        'class_id': class_id,
                        'confidence': confidence,
                        'corners': corners.tolist(),  # 4 corner points
                        'bbox_type': 'obb'
                    }
                    detections.append(detection)
            
            # Get image dimensions
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not read image: {image_path}")
                return None
            
            height, width = image.shape[:2]
            
            return {
                'image_path': image_path,
                'image_width': width,
                'image_height': height,
                'detections': detections,
                'model_info': {
                    'conf_threshold': conf_threshold,
                    'iou_threshold': iou_threshold,
                    'model_path': self.model_path
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def predict_batch(
        self,
        image_paths: List[str],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        max_det: int = 1000,
        img_size: int = 640,
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a batch of images
        
        Args:
            image_paths: List of image paths
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            max_det: Maximum detections per image
            img_size: Input image size
            batch_size: Batch size for inference
            
        Returns:
            List of detection results
        """
        results = []
        
        # Process images in batches
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            try:
                # Run batch inference
                batch_results = self.model.predict(
                    source=batch_paths,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=max_det,
                    imgsz=img_size,
                    verbose=False
                )
                
                # Process each result in batch
                for j, result in enumerate(batch_results):
                    image_path = batch_paths[j]
                    
                    # Extract oriented bounding boxes
                    detections = []
                    if hasattr(result, 'obb') and result.obb is not None:
                        obb_data = result.obb
                        
                        for k in range(len(obb_data.xyxyxyxy)):
                            corners = obb_data.xyxyxyxy[k].cpu().numpy()
                            confidence = float(obb_data.conf[k])
                            class_id = int(obb_data.cls[k])
                            
                            detection = {
                                'class_id': class_id,
                                'confidence': confidence,
                                'corners': corners.tolist(),
                                'bbox_type': 'obb'
                            }
                            detections.append(detection)
                    
                    # Get image dimensions
                    image = cv2.imread(image_path)
                    if image is not None:
                        height, width = image.shape[:2]
                        
                        result_dict = {
                            'image_path': image_path,
                            'image_width': width,
                            'image_height': height,
                            'detections': detections,
                            'model_info': {
                                'conf_threshold': conf_threshold,
                                'iou_threshold': iou_threshold,
                                'model_path': self.model_path
                            }
                        }
                        results.append(result_dict)
                    else:
                        self.logger.warning(f"Could not read image: {image_path}")
                        
            except Exception as e:
                self.logger.error(f"Error processing batch {i}-{i+batch_size}: {e}")
                continue
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], output_dir: str):
        """
        Save YOLO results to files
        
        Args:
            results: List of detection results
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary
        summary = {
            'total_images': len(results),
            'total_detections': sum(len(r['detections']) for r in results),
            'model_path': self.model_path,
            'device': self.device
        }
        
        import json
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        with open(os.path.join(output_dir, 'detections.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_dir}")
    
    def visualize_detections(
        self,
        image_path: str,
        detections: List[Dict[str, Any]],
        output_path: str,
        class_names: Optional[Dict[int, str]] = None
    ):
        """
        Visualize detections on image
        
        Args:
            image_path: Path to input image
            detections: List of detections
            output_path: Path to save visualization
            class_names: Mapping of class IDs to names
        """
        try:
            image = cv2.imread(image_path)
            if image is None:
                self.logger.error(f"Could not read image: {image_path}")
                return
            
            for detection in detections:
                corners = np.array(detection['corners'], dtype=np.int32)
                class_id = detection['class_id']
                confidence = detection['confidence']
                
                # Draw oriented bounding box
                cv2.polylines(image, [corners], True, (0, 255, 0), 2)
                
                # Add label
                label = f"Class {class_id}"
                if class_names and class_id in class_names:
                    label = class_names[class_id]
                label += f" {confidence:.2f}"
                
                # Find top-left corner for label placement
                x_min = int(np.min(corners[:, 0]))
                y_min = int(np.min(corners[:, 1]))
                
                cv2.putText(image, label, (x_min, y_min - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save visualization
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)
            
        except Exception as e:
            self.logger.error(f"Error visualizing detections: {e}")


def obb_to_polygon_points(corners: List[List[float]]) -> List[Tuple[float, float]]:
    """
    Convert OBB corners to polygon points for Label Studio
    
    Args:
        corners: List of 4 corner points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        
    Returns:
        List of (x, y) tuples representing polygon points
    """
    return [(float(corner[0]), float(corner[1])) for corner in corners]


def normalize_coordinates(
    corners: List[List[float]], 
    image_width: int, 
    image_height: int
) -> List[List[float]]:
    """
    Normalize pixel coordinates to percentages (0-100)
    
    Args:
        corners: Corner points in pixel coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
        
    Returns:
        Normalized corner points (0-100 range)
    """
    normalized = []
    for corner in corners:
        x_norm = (corner[0] / image_width) * 100
        y_norm = (corner[1] / image_height) * 100
        normalized.append([x_norm, y_norm])
    
    return normalized