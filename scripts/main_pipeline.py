#!/usr/bin/env python3
"""
Main Pipeline Script
Orchestrates the complete YOLO OBB + Label Studio pipeline
"""

import os
import sys
import logging
import yaml
import time
import argparse
from typing import List, Optional
from pathlib import Path

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import essential pipeline scripts for existing Label Studio workflow
import importlib.util
import sys

# Helper function to import modules with numeric names
def import_script(script_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Import the required scripts
script_dir = os.path.dirname(__file__)
get_existing_script = import_script(os.path.join(script_dir, "0_get_existing_images.py"), "get_existing_script")
inference_script = import_script(os.path.join(script_dir, "2_yolo_inference.py"), "inference_script")
convert_script = import_script(os.path.join(script_dir, "3_convert_to_predictions.py"), "convert_script")
upload_existing_script = import_script(os.path.join(script_dir, "5_upload_predictions_to_existing.py"), "upload_existing_script")


def setup_logging(config: dict):
    """Setup logging configuration"""
    log_level = getattr(logging, config['logging']['level'].upper())
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create logs directory
    os.makedirs(config['paths']['logs'], exist_ok=True)
    
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


def create_directories(config: dict):
    """Create necessary directories for the pipeline"""
    directories = [
        config['paths']['raw_images'],
        config['paths']['yolo_results'],
        config['paths']['predictions'],
        config['paths']['logs'],
        config['paths']['models']
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def validate_prerequisites(config: dict) -> bool:
    """
    Validate that all prerequisites are met
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if all prerequisites are met, False otherwise
    """
    logger = logging.getLogger(__name__)
    validation_errors = []
    
    # Check YOLO model
    if config['stages']['inference']:
        model_path = config['yolo']['model_path']
        if not os.path.exists(model_path):
            validation_errors.append(f"YOLO model not found: {model_path}")
    
    # Check class mapping
    if not config.get('class_mapping'):
        validation_errors.append("Class mapping is empty or not defined")
    
    # Check Label Studio configuration
    if config['stages']['upload_predictions']:
        if not config['label_studio']['api_key']:
            validation_errors.append("Label Studio API key is not set")
        if not config['label_studio']['project_id']:
            validation_errors.append("Label Studio project ID is not set")
    
    # Check existing project configuration
    if config['stages']['get_existing_images']:
        if not config.get('existing_project', {}).get('enabled', False):
            validation_errors.append("Existing project mode is not enabled in config")
    
    # Log validation results
    if validation_errors:
        logger.error("Validation failed:")
        for error in validation_errors:
            logger.error(f"  - {error}")
        return False
    else:
        logger.info("All prerequisites validated successfully")
        return True


# Download stage removed - using existing Label Studio project workflow


def run_inference_stage(config: dict) -> bool:
    """Run the YOLO inference stage"""
    logger = logging.getLogger(__name__)
    
    if not config['stages']['inference']:
        logger.info("Inference stage is disabled, skipping...")
        return True
    
    logger.info("Starting YOLO inference stage...")
    try:
        # Get downloaded images
        image_paths = inference_script.get_downloaded_images(config)
        if not image_paths:
            logger.error("No images found for inference")
            return False
        
        success = inference_script.run_yolo_inference(config, image_paths)
        if success:
            logger.info("Inference stage completed successfully")
        else:
            logger.error("Inference stage failed")
        return success
    except Exception as e:
        logger.error(f"Inference stage failed with exception: {e}")
        return False


def run_conversion_stage(config: dict) -> bool:
    """Run the format conversion stage"""
    logger = logging.getLogger(__name__)
    
    if not config['stages']['convert']:
        logger.info("Conversion stage is disabled, skipping...")
        return True
    
    logger.info("Starting format conversion stage...")
    try:
        # Load YOLO results
        yolo_results = convert_script.load_yolo_results(config)
        if not yolo_results:
            logger.error("No YOLO results found for conversion")
            return False
        
        # Convert to Label Studio format
        label_studio_tasks = convert_script.convert_to_label_studio_format(config, yolo_results)
        if not label_studio_tasks:
            logger.error("Conversion failed - no tasks created")
            return False
        
        # Create alternative formats
        alternative_formats = convert_script.create_alternative_formats(config, yolo_results)
        
        # Validate and save predictions
        validation_passed = convert_script.validate_converted_predictions(label_studio_tasks)
        convert_script.save_predictions(config, label_studio_tasks, alternative_formats)
        convert_script.create_sample_predictions(config, label_studio_tasks)
        
        logger.info("Conversion stage completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Conversion stage failed with exception: {e}")
        return False


# Upload stage removed - using upload_predictions_to_existing for streamlined workflow


def run_cleanup_stage(config: dict):
    """Run the cleanup stage"""
    logger = logging.getLogger(__name__)
    
    if not config['stages']['cleanup']:
        logger.info("Cleanup stage is disabled, skipping...")
        return
    
    logger.info("Starting cleanup stage...")
    
    try:
        # Cleanup temporary files if specified
        if config['processing']['cleanup_temp_files']:
            # Remove YOLO visualization files if they exist
            viz_dir = os.path.join(config['paths']['yolo_results'], 'visualizations')
            if os.path.exists(viz_dir):
                import shutil
                shutil.rmtree(viz_dir)
                logger.info("Removed YOLO visualization files")
            
            # Remove sample predictions
            sample_file = os.path.join(config['paths']['predictions'], 'predictions_sample.json')
            if os.path.exists(sample_file):
                os.remove(sample_file)
                logger.info("Removed sample predictions file")
        
        logger.info("Cleanup stage completed")
        
    except Exception as e:
        logger.warning(f"Cleanup stage had issues: {e}")


def run_get_existing_images_stage(config: dict) -> bool:
    """Run the get existing images stage"""
    logger = logging.getLogger(__name__)
    
    if not config['stages'].get('get_existing_images', False):
        logger.info("Get existing images stage is disabled, skipping...")
        return True
    
    if not config.get('existing_project', {}).get('enabled', False):
        logger.info("Existing project mode is disabled, skipping...")
        return True
    
    if get_existing_script is None:
        logger.error("Get existing images script not available")
        return False
    
    logger.info("Starting get existing images stage...")
    try:
        success = get_existing_script.get_existing_images(config)
        if success:
            logger.info("Get existing images stage completed successfully")
        else:
            logger.error("Get existing images stage failed")
        return success
    except Exception as e:
        logger.error(f"Get existing images stage failed with exception: {e}")
        return False


def run_upload_predictions_stage(config: dict) -> bool:
    """Run the upload predictions to existing tasks stage"""
    logger = logging.getLogger(__name__)
    
    if not config['stages'].get('upload_predictions', False):
        logger.info("Upload predictions stage is disabled, skipping...")
        return True
    
    if not config.get('existing_project', {}).get('enabled', False):
        logger.info("Existing project mode is disabled, skipping...")
        return True
    
    if upload_existing_script is None:
        logger.error("Upload predictions to existing script not available")
        return False
    
    logger.info("Starting upload predictions to existing stage...")
    try:
        success = upload_existing_script.upload_predictions_to_existing(config)
        if success:
            logger.info("Upload predictions stage completed successfully")
        else:
            logger.error("Upload predictions stage failed")
        return success
    except Exception as e:
        logger.error(f"Upload predictions stage failed with exception: {e}")
        return False


def print_pipeline_summary(config: dict, stage_results: dict, total_time: float):
    """Print a summary of the pipeline execution"""
    print("\n" + "="*80)
    print("🎯 PIPELINE EXECUTION SUMMARY")
    print("="*80)
    
    # Stage results
    print("\n📊 Stage Results:")
    stages = ['get_existing_images', 'inference', 'convert', 'upload_predictions']
    stage_names = ['Get Images from Label Studio', 'YOLO Inference', 'Format Conversion', 'Upload Predictions']
    
    for stage, name in zip(stages, stage_names):
        if config['stages'].get(stage, False):
            status = "✅ SUCCESS" if stage_results.get(stage, False) else "❌ FAILED"
            print(f"   {name}: {status}")
        else:
            print(f"   {name}: ⏭️  SKIPPED")
    
    # Time and performance
    print(f"\n⏱️  Total Pipeline Time: {total_time:.2f} seconds")
    
    # Files created
    print(f"\n📁 Output Locations:")
    if config['stages'].get('get_existing_images', False):
        print(f"   - Downloaded Images: {config['paths']['raw_images']}")
    if config['stages']['inference']:
        print(f"   - YOLO Results: {config['paths']['yolo_results']}")
    if config['stages']['convert']:
        print(f"   - Predictions: {config['paths']['predictions']}")
    
    # Next steps
    if stage_results.get('upload', False):
        print(f"\n🎯 Next Steps:")
        project_url = f"{config['label_studio']['url']}/projects/{config['label_studio']['project_id']}"
        print(f"   1. Open Label Studio: {project_url}")
        print(f"   2. Review and correct the predictions")
        print(f"   3. Export corrected annotations")
    
    print("\n" + "="*80)


def main():
    """Main pipeline entry point"""
    parser = argparse.ArgumentParser(description="YOLO OBB + Label Studio Pipeline")
    parser.add_argument("--config", "-c", default="config/config.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--stage", "-s", choices=['get_existing_images', 'inference', 'convert', 'upload_predictions'], 
                       help="Run only specific stage")
    parser.add_argument("--skip-validation", action="store_true", 
                       help="Skip prerequisite validation")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Show what would be executed without running")
    
    args = parser.parse_args()
    
    print("🚀 YOLO OBB + Label Studio Pipeline")
    print("="*50)
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Create directories
    create_directories(config)
    
    # Validate prerequisites
    if not args.skip_validation:
        if not validate_prerequisites(config):
            print("\n❌ Prerequisite validation failed!")
            print("Fix the issues above or use --skip-validation to proceed anyway")
            sys.exit(1)
    
    # Override stages if specific stage requested
    if args.stage:
        logger.info(f"Running only stage: {args.stage}")
        for stage in config['stages']:
            config['stages'][stage] = (stage == args.stage)
    
    # Dry run mode
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - Showing what would be executed:")
        stages = ['get_existing_images', 'inference', 'convert', 'upload_predictions']
        stage_names = ['Get Images from Label Studio', 'YOLO Inference', 'Format Conversion', 'Upload Predictions']
        
        for stage, name in zip(stages, stage_names):
            status = "✅ ENABLED" if config['stages'].get(stage, False) else "⏭️  DISABLED"
            print(f"   {name}: {status}")
        
        print("\nStreamlined workflow: Label Studio → YOLO → Predictions → Label Studio")
        print("Run without --dry-run to execute the pipeline")
        return
    
    # Execute pipeline
    start_time = time.time()
    stage_results = {}
    
    logger.info("Starting pipeline execution...")
    
    try:
        # Stage 1: Get existing images from Label Studio
        stage_results['get_existing_images'] = run_get_existing_images_stage(config)
        if config['stages'].get('get_existing_images', False) and not stage_results['get_existing_images']:
            if not config['processing']['resume_on_failure']:
                logger.error("Get existing images stage failed, stopping pipeline")
                sys.exit(1)
        
        # Stage 2: YOLO Inference
        stage_results['inference'] = run_inference_stage(config)
        if config['stages']['inference'] and not stage_results['inference']:
            if not config['processing']['resume_on_failure']:
                logger.error("Inference stage failed, stopping pipeline")
                sys.exit(1)
        
        # Stage 3: Format Conversion
        stage_results['convert'] = run_conversion_stage(config)
        if config['stages']['convert'] and not stage_results['convert']:
            if not config['processing']['resume_on_failure']:
                logger.error("Conversion stage failed, stopping pipeline")
                sys.exit(1)
        
        # Stage 4: Upload predictions to existing Label Studio tasks
        stage_results['upload_predictions'] = run_upload_predictions_stage(config)
        if config['stages'].get('upload_predictions', False) and not stage_results['upload_predictions']:
            if not config['processing']['resume_on_failure']:
                logger.error("Upload predictions stage failed, stopping pipeline")
                sys.exit(1)
        
        total_time = time.time() - start_time
        
        # Print summary
        print_pipeline_summary(config, stage_results, total_time)
        
        # Check overall success
        enabled_stages = [stage for stage, enabled in config['stages'].items() 
                         if enabled and stage != 'cleanup']
        successful_stages = [stage for stage, success in stage_results.items() if success]
        
        if len(successful_stages) == len(enabled_stages):
            logger.info("Pipeline completed successfully!")
            print("\n🎉 Pipeline completed successfully!")
        else:
            logger.warning("Pipeline completed with some failures")
            print("\n⚠️  Pipeline completed with some failures")
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\n⏹️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}")
        print(f"\n💥 Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 