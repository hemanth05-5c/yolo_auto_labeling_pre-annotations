#!/usr/bin/env python3
"""
Script 2: YOLO OBB Model Inference
Runs YOLO OBB model inference on downloaded images
"""

import os
import sys
import logging
import yaml
import json
from pathlib import Path
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.yolo_utils import YOLOOBBProcessor


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


def get_downloaded_images(config: dict) -> List[str]:
    """
    Get list of downloaded images from previous step
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of image paths
    """
    logger = logging.getLogger(__name__)
    
    # Check if download results exist
    results_file = os.path.join(config['paths']['raw_images'], 'download_results.yaml')
    
    if os.path.exists(results_file):
        # Load from download results
        with open(results_file, 'r') as f:
            download_results = yaml.safe_load(f)
        
        image_paths = [item['local_path'] for item in download_results['successful']]
        logger.info(f"Loaded {len(image_paths)} images from download results")
        
    else:
        # Scan directory for image files
        logger.warning("Download results not found, scanning directory for images")
        
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_paths = []
        
        for root, dirs, files in os.walk(config['paths']['raw_images']):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_paths.append(os.path.join(root, file))
        
        logger.info(f"Found {len(image_paths)} images by scanning directory")
    
    return image_paths


def run_yolo_inference(config: dict, image_paths: List[str]) -> bool:
    """
    Run YOLO OBB inference on images
    
    Args:
        config: Configuration dictionary
        image_paths: List of image paths to process
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize YOLO processor
        yolo_processor = YOLOOBBProcessor(
            model_path=config['yolo']['model_path'],
            device=config['yolo']['device']
        )
        
        logger.info(f"Starting YOLO inference on {len(image_paths)} images...")
        
        # Run inference
        if config['processing']['parallel_inference'] and len(image_paths) > 1:
            # Parallel processing
            results = run_parallel_inference(
                yolo_processor, 
                image_paths, 
                config
            )
        else:
            # Sequential processing
            results = run_sequential_inference(
                yolo_processor,
                image_paths,
                config
            )
        
        # Save results
        yolo_processor.save_results(results, config['paths']['yolo_results'])
        
        # Generate summary statistics
        total_detections = sum(len(r['detections']) for r in results)
        avg_detections = total_detections / len(results) if results else 0
        
        logger.info(f"YOLO inference completed:")
        logger.info(f"  - Images processed: {len(results)}")
        logger.info(f"  - Total detections: {total_detections}")
        logger.info(f"  - Average detections per image: {avg_detections:.2f}")
        
        # Save processing summary
        summary = {
            'images_processed': len(results),
            'total_detections': total_detections,
            'average_detections_per_image': avg_detections,
            'model_config': config['yolo'],
            'processing_config': config['processing']
        }
        
        summary_file = os.path.join(config['paths']['yolo_results'], 'inference_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return len(results) > 0
        
    except Exception as e:
        logger.error(f"Error running YOLO inference: {e}")
        return False


def run_sequential_inference(
    yolo_processor: YOLOOBBProcessor,
    image_paths: List[str],
    config: dict
) -> List[dict]:
    """Run inference sequentially with progress bar"""
    logger = logging.getLogger(__name__)
    results = []
    
    with tqdm(total=len(image_paths), desc="YOLO Inference") as pbar:
        for image_path in image_paths:
            result = yolo_processor.predict_single_image(
                image_path=image_path,
                conf_threshold=config['yolo']['confidence_threshold'],
                iou_threshold=config['yolo']['iou_threshold'],
                max_det=config['yolo']['max_detections'],
                img_size=config['yolo']['image_size']
            )
            
            if result:
                results.append(result)
            else:
                logger.warning(f"Failed to process image: {image_path}")
            
            pbar.update(1)
    
    return results


def run_parallel_inference(
    yolo_processor: YOLOOBBProcessor,
    image_paths: List[str],
    config: dict
) -> List[dict]:
    """Run inference in parallel using batch processing"""
    logger = logging.getLogger(__name__)
    
    # Use batch processing for better efficiency
    return yolo_processor.predict_batch(
        image_paths=image_paths,
        conf_threshold=config['yolo']['confidence_threshold'],
        iou_threshold=config['yolo']['iou_threshold'],
        max_det=config['yolo']['max_detections'],
        img_size=config['yolo']['image_size'],
        batch_size=config['yolo']['batch_size']
    )


def create_visualizations(config: dict, results: List[dict]):
    """
    Create visualization images with detections
    
    Args:
        config: Configuration dictionary
        results: YOLO inference results
    """
    if not config['yolo']['save_visualizations']:
        return
    
    logger = logging.getLogger(__name__)
    logger.info("Creating visualization images...")
    
    # Create visualization directory
    viz_dir = os.path.join(config['paths']['yolo_results'], 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # Initialize YOLO processor for visualization
    yolo_processor = YOLOOBBProcessor(
        model_path=config['yolo']['model_path'],
        device=config['yolo']['device']
    )
    
    class_names = config.get('class_mapping', {})
    
    for result in tqdm(results, desc="Creating visualizations"):
        image_path = result['image_path']
        detections = result['detections']
        
        # Create output path
        image_name = os.path.basename(image_path)
        name, ext = os.path.splitext(image_name)
        viz_path = os.path.join(viz_dir, f"{name}_detected{ext}")
        
        # Create visualization
        yolo_processor.visualize_detections(
            image_path=image_path,
            detections=detections,
            output_path=viz_path,
            class_names=class_names
        )
    
    logger.info(f"Visualizations saved to: {viz_dir}")


def main():
    """Main entry point"""
    print("YOLO OBB + Label Studio Pipeline - Step 2: YOLO Inference")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Check if inference stage is enabled
    if not config['stages']['inference']:
        logger.info("Inference stage is disabled in configuration")
        return
    
    # Check if YOLO model exists
    if not os.path.exists(config['yolo']['model_path']):
        logger.error(f"YOLO model not found: {config['yolo']['model_path']}")
        print(f"\n❌ YOLO model not found: {config['yolo']['model_path']}")
        print("Please download or specify the correct path to your YOLO OBB model")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs(config['paths']['yolo_results'], exist_ok=True)
    
    # Get list of downloaded images
    image_paths = get_downloaded_images(config)
    
    if not image_paths:
        logger.error("No images found for processing")
        print("\n❌ No images found for processing")
        print("Make sure to run step 1 (download from GCS) first")
        sys.exit(1)
    
    logger.info(f"Starting YOLO inference on {len(image_paths)} images...")
    
    # Run YOLO inference
    success = run_yolo_inference(config, image_paths)
    
    if success:
        # Load results for visualization
        results_file = os.path.join(config['paths']['yolo_results'], 'detections.json')
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Create visualizations if enabled
        create_visualizations(config, results)
        
        logger.info("YOLO inference completed successfully!")
        print("\n✅ YOLO inference completed successfully!")
        print(f"📁 Results saved to: {config['paths']['yolo_results']}")
        
        # Show summary
        summary_file = os.path.join(config['paths']['yolo_results'], 'inference_summary.json')
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            print(f"📊 Summary:")
            print(f"   - Images processed: {summary['images_processed']}")
            print(f"   - Total detections: {summary['total_detections']}")
            print(f"   - Avg detections/image: {summary['average_detections_per_image']:.2f}")
    else:
        logger.error("YOLO inference failed!")
        print("\n❌ YOLO inference failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 