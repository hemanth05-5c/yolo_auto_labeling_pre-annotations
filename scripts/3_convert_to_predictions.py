#!/usr/bin/env python3
"""
Script 3: Convert YOLO OBB Results to Label Studio Predictions
Converts YOLO OBB detection results to Label Studio predictions format
"""

import os
import sys
import logging
import yaml
import json
from pathlib import Path
from typing import List, Dict, Any

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.format_converter import YOLOToLabelStudioConverter, AlternativeFormatConverter, validate_predictions


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


def load_yolo_results(config: dict) -> List[Dict[str, Any]]:
    """
    Load YOLO detection results from previous step
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of YOLO detection results
    """
    logger = logging.getLogger(__name__)
    
    # Load YOLO results
    results_file = os.path.join(config['paths']['yolo_results'], 'detections.json')
    
    if not os.path.exists(results_file):
        logger.error(f"YOLO results not found: {results_file}")
        raise FileNotFoundError(f"YOLO results not found. Please run step 2 (YOLO inference) first.")
    
    try:
        with open(results_file, 'r') as f:
            yolo_results = json.load(f)
        
        logger.info(f"Loaded {len(yolo_results)} YOLO results")
        return yolo_results
        
    except Exception as e:
        logger.error(f"Error loading YOLO results: {e}")
        raise


def convert_to_label_studio_format(
    config: dict, 
    yolo_results: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Convert YOLO results to Label Studio predictions format
    
    Args:
        config: Configuration dictionary
        yolo_results: List of YOLO detection results
        
    Returns:
        List of Label Studio tasks with predictions
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize converter
        converter = YOLOToLabelStudioConverter(
            class_mapping=config['class_mapping'],
            model_version=config['label_studio']['model_version']
        )
        
        # Convert results
        logger.info("Converting YOLO results to Label Studio format...")
        
        label_studio_tasks = converter.convert_batch_results(
            yolo_results=yolo_results,
            image_url_prefix=config['label_studio']['image_url_prefix']
        )
        
        logger.info(f"Successfully converted {len(label_studio_tasks)} tasks")
        return label_studio_tasks
        
    except Exception as e:
        logger.error(f"Error converting to Label Studio format: {e}")
        raise


def create_alternative_formats(
    config: dict,
    yolo_results: List[Dict[str, Any]]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Create alternative format conversions for different labeling configurations
    
    Args:
        config: Configuration dictionary
        yolo_results: YOLO detection results
        
    Returns:
        Dictionary with different format conversions
    """
    logger = logging.getLogger(__name__)
    alternative_formats = {}
    
    try:
        # Rectangle labels format (axis-aligned bounding boxes)
        logger.info("Creating rectangle labels format...")
        rectangle_tasks = AlternativeFormatConverter.convert_to_rectangle_labels(
            yolo_results=yolo_results,
            class_mapping=config['class_mapping'],
            model_version=config['label_studio']['model_version'],
            image_url_prefix=config['label_studio']['image_url_prefix']
        )
        alternative_formats['rectangle_labels'] = rectangle_tasks
        
        logger.info(f"Created {len(rectangle_tasks)} tasks in rectangle format")
        
    except Exception as e:
        logger.warning(f"Error creating alternative formats: {e}")
    
    return alternative_formats


def validate_converted_predictions(tasks: List[Dict[str, Any]]) -> bool:
    """
    Validate the converted predictions
    
    Args:
        tasks: List of Label Studio tasks
        
    Returns:
        True if validation passes, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Validating converted predictions...")
    validation_results = validate_predictions(tasks)
    
    logger.info(f"Validation results:")
    logger.info(f"  - Valid tasks: {validation_results['valid_tasks']}")
    logger.info(f"  - Invalid tasks: {validation_results['invalid_tasks']}")
    
    if validation_results['errors']:
        logger.warning(f"Found {len(validation_results['errors'])} validation errors:")
        for error in validation_results['errors'][:10]:  # Show first 10 errors
            logger.warning(f"  - {error}")
        if len(validation_results['errors']) > 10:
            logger.warning(f"  ... and {len(validation_results['errors']) - 10} more errors")
    
    if validation_results['warnings']:
        logger.warning(f"Found {len(validation_results['warnings'])} validation warnings:")
        for warning in validation_results['warnings'][:5]:  # Show first 5 warnings
            logger.warning(f"  - {warning}")
    
    # Consider validation successful if we have more valid than invalid tasks
    return validation_results['valid_tasks'] > validation_results['invalid_tasks']


def save_predictions(
    config: dict, 
    tasks: List[Dict[str, Any]], 
    alternative_formats: Dict[str, List[Dict[str, Any]]]
):
    """
    Save predictions to files
    
    Args:
        config: Configuration dictionary
        tasks: Main Label Studio tasks
        alternative_formats: Alternative format tasks
    """
    logger = logging.getLogger(__name__)
    
    # Create predictions directory
    predictions_dir = config['paths']['predictions']
    os.makedirs(predictions_dir, exist_ok=True)
    
    # Save main predictions (polygon format)
    main_file = os.path.join(predictions_dir, 'predictions_polygon.json')
    try:
        with open(main_file, 'w') as f:
            json.dump(tasks, f, indent=2)
        logger.info(f"Saved main predictions to: {main_file}")
    except Exception as e:
        logger.error(f"Error saving main predictions: {e}")
        raise
    
    # Save alternative formats
    for format_name, format_tasks in alternative_formats.items():
        try:
            format_file = os.path.join(predictions_dir, f'predictions_{format_name}.json')
            with open(format_file, 'w') as f:
                json.dump(format_tasks, f, indent=2)
            logger.info(f"Saved {format_name} format to: {format_file}")
        except Exception as e:
            logger.warning(f"Error saving {format_name} format: {e}")
    
    # Save conversion summary
    summary = {
        'total_images': len(tasks),
        'total_predictions': sum(len(task['predictions']) for task in tasks),
        'formats_created': ['polygon'] + list(alternative_formats.keys()),
        'class_mapping': config['class_mapping'],
        'model_version': config['label_studio']['model_version']
    }
    
    summary_file = os.path.join(predictions_dir, 'conversion_summary.json')
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved conversion summary to: {summary_file}")
    except Exception as e:
        logger.warning(f"Error saving conversion summary: {e}")


def create_sample_predictions(config: dict, tasks: List[Dict[str, Any]], sample_size: int = 5):
    """
    Create a sample of predictions for testing
    
    Args:
        config: Configuration dictionary
        tasks: All Label Studio tasks
        sample_size: Number of sample tasks to create
    """
    if len(tasks) <= sample_size:
        return
    
    logger = logging.getLogger(__name__)
    
    # Create sample
    sample_tasks = tasks[:sample_size]
    
    # Save sample
    predictions_dir = config['paths']['predictions']
    sample_file = os.path.join(predictions_dir, 'predictions_sample.json')
    
    try:
        with open(sample_file, 'w') as f:
            json.dump(sample_tasks, f, indent=2)
        logger.info(f"Created sample predictions file with {len(sample_tasks)} tasks: {sample_file}")
    except Exception as e:
        logger.warning(f"Error creating sample predictions: {e}")


def main():
    """Main entry point"""
    print("YOLO OBB + Label Studio Pipeline - Step 3: Convert to Predictions")
    print("=" * 65)
    
    # Load configuration
    config = load_config()
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Check if conversion stage is enabled
    if not config['stages']['convert']:
        logger.info("Conversion stage is disabled in configuration")
        return
    
    try:
        # Load YOLO results
        yolo_results = load_yolo_results(config)
        
        if not yolo_results:
            logger.error("No YOLO results found for conversion")
            print("\n❌ No YOLO results found for conversion")
            print("Make sure to run step 2 (YOLO inference) first")
            sys.exit(1)
        
        # Convert to Label Studio format
        label_studio_tasks = convert_to_label_studio_format(config, yolo_results)
        
        if not label_studio_tasks:
            logger.error("Conversion failed - no tasks created")
            print("\n❌ Conversion failed - no tasks created")
            sys.exit(1)
        
        # Create alternative formats
        alternative_formats = create_alternative_formats(config, yolo_results)
        
        # Validate predictions
        validation_passed = validate_converted_predictions(label_studio_tasks)
        
        if not validation_passed:
            logger.warning("Validation warnings found, but proceeding with save")
        
        # Save predictions
        save_predictions(config, label_studio_tasks, alternative_formats)
        
        # Create sample for testing
        create_sample_predictions(config, label_studio_tasks)
        
        # Success
        logger.info("Format conversion completed successfully!")
        print("\n✅ Format conversion completed successfully!")
        print(f"📁 Predictions saved to: {config['paths']['predictions']}")
        
        # Show summary
        total_detections = sum(
            len(pred['result']) for task in label_studio_tasks 
            for pred in task['predictions']
        )
        
        print(f"📊 Summary:")
        print(f"   - Images converted: {len(label_studio_tasks)}")
        print(f"   - Total detections: {total_detections}")
        print(f"   - Formats created: polygon, {', '.join(alternative_formats.keys())}")
        
        # Show files created
        predictions_dir = config['paths']['predictions']
        print(f"\n📄 Files created:")
        print(f"   - Main predictions: {os.path.join(predictions_dir, 'predictions_polygon.json')}")
        for format_name in alternative_formats.keys():
            print(f"   - {format_name.title()} format: {os.path.join(predictions_dir, f'predictions_{format_name}.json')}")
        print(f"   - Sample predictions: {os.path.join(predictions_dir, 'predictions_sample.json')}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        print(f"\n❌ Conversion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 