#!/usr/bin/env python3
"""
Get Images from Existing Label Studio Project
Downloads images from an existing Label Studio project for local YOLO inference
"""

import os
import sys
import logging
import yaml
import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import urllib.parse

# Optional Google Cloud Storage support
try:
    from google.cloud import storage
    from google.auth.exceptions import DefaultCredentialsError
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False
    storage = None

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


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


class LabelStudioImageDownloader:
    """Download images from existing Label Studio project"""
    
    def __init__(self, url: str, api_key: str, project_id: int):
        """
        Initialize Label Studio image downloader
        
        Args:
            url: Label Studio server URL
            api_key: API key for authentication  
            project_id: Project ID to work with
        """
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.project_id = project_id
        self.logger = logging.getLogger(__name__)
        
        # Setup headers
        self.headers = {"Authorization": f"Token {self.api_key}"}
        
        # Initialize GCS client if available
        self.gcs_client = None
        if GCS_AVAILABLE:
            try:
                self.gcs_client = storage.Client()
                self.logger.info("Google Cloud Storage client initialized successfully")
            except DefaultCredentialsError:
                self.logger.warning("GCS credentials not found. Private GCS buckets will not be accessible.")
            except Exception as e:
                self.logger.warning(f"Failed to initialize GCS client: {e}")
    
    def get_project_tasks(self) -> List[Dict[str, Any]]:
        """
        Get ALL tasks from the Label Studio project (handles pagination)
        
        Returns:
            List of task dictionaries
        """
        try:
            all_tasks = []
            page = 1
            page_size = 100  # Maximum allowed by Label Studio API
            
            while True:
                url = f"{self.url}/api/projects/{self.project_id}/tasks/"
                params = {
                    'page': page,
                    'page_size': page_size
                }
                
                self.logger.info(f"Fetching page {page} (up to {page_size} tasks per page)...")
                response = requests.get(url, headers=self.headers, params=params)
                
                # Handle 404 as end of pagination gracefully
                if response.status_code == 404:
                    self.logger.info(f"Reached end of pagination at page {page}")
                    break
                    
                response.raise_for_status()
                
                data = response.json()
                
                # Label Studio returns a simple list, not paginated format
                if isinstance(data, list):
                    # Direct list response
                    tasks = data
                    all_tasks.extend(tasks)
                    
                    self.logger.info(f"Page {page}: Got {len(tasks)} tasks (total so far: {len(all_tasks)})")
                    
                    # If we got fewer tasks than page_size, we've reached the end
                    if len(tasks) < page_size:
                        self.logger.info(f"Reached end of results (got {len(tasks)} < {page_size})")
                        break
                        
                    page += 1
                    
                elif isinstance(data, dict) and 'results' in data:
                    # Paginated response (newer versions)
                    tasks = data['results']
                    total_count = data.get('count', 0)
                    
                    all_tasks.extend(tasks)
                    
                    self.logger.info(f"Page {page}: Got {len(tasks)} tasks (total so far: {len(all_tasks)}/{total_count})")
                    
                    # Check if we have more pages
                    if len(tasks) < page_size or len(all_tasks) >= total_count:
                        break
                        
                    page += 1
                else:
                    # Unknown format
                    self.logger.warning(f"Unexpected API response format: {type(data)}")
                    break
            
            self.logger.info(f"Found {len(all_tasks)} total tasks in project {self.project_id}")
            return all_tasks
            
        except Exception as e:
            self.logger.error(f"Error getting project tasks: {e}")
            raise
    
    def extract_image_urls(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Extract image URLs from tasks
        
        Args:
            tasks: List of Label Studio tasks
            
        Returns:
            List of dictionaries with task_id, image_url, and filename
        """
        image_info = []
        
        for task in tasks:
            try:
                task_id = task['id']
                data = task.get('data', {})
                
                # Look for image field (common field names)
                image_url = None
                
                # Extended list of possible field names
                possible_fields = [
                    'image', 'img', 'picture', 'photo', 'file', 'data', 'url', 'src',
                    'image_url', 'img_url', 'file_url', 'image_path', 'img_path',
                    'attachment', 'asset', 'media', 'resource'
                ]
                
                for field in possible_fields:
                    if field in data and data[field]:
                        image_url = data[field]
                        self.logger.debug(f"Found image URL in field '{field}': {image_url}")
                        break
                
                # If no image URL found, log the available fields for debugging
                if not image_url:
                    available_fields = list(data.keys()) if data else []
                    self.logger.debug(f"Task {task_id} data fields: {available_fields}")
                    if data:
                        # Look for any field that might contain a URL or file path
                        for key, value in data.items():
                            if isinstance(value, str) and (
                                value.startswith('http') or 
                                value.startswith('gs://') or 
                                value.startswith('/') or
                                any(ext in value.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'])
                            ):
                                image_url = value
                                self.logger.info(f"Found potential image URL in field '{key}': {image_url}")
                                break
                
                if image_url:
                    # Extract filename from URL
                    parsed_url = urllib.parse.urlparse(image_url)
                    filename = os.path.basename(parsed_url.path)
                    
                    # Clean up filename for GCS URLs
                    if 'storage.googleapis.com' in image_url or 'gcs' in image_url:
                        # Extract actual filename from GCS URL
                        if '/' in parsed_url.path:
                            filename = parsed_url.path.split('/')[-1]
                    
                    # If no filename in path, create one from task ID
                    if not filename or '.' not in filename:
                        filename = f"task_{task_id}.jpg"
                    
                    image_info.append({
                        'task_id': task_id,
                        'image_url': image_url,
                        'filename': filename
                    })
                else:
                    self.logger.warning(f"No image URL found in task {task_id}")
                    
            except Exception as e:
                self.logger.error(f"Error processing task {task.get('id', 'unknown')}: {e}")
        
        self.logger.info(f"Extracted {len(image_info)} image URLs")
        return image_info
    
    def download_single_image(self, image_info: Dict[str, str], output_dir: str) -> bool:
        """
        Download a single image
        
        Args:
            image_info: Dictionary with image information
            output_dir: Output directory for downloaded images
            
        Returns:
            True if successful, False otherwise
        """
        try:
            image_url = image_info['image_url']
            filename = image_info['filename']
            task_id = image_info['task_id']
            
            # Create output path
            output_path = os.path.join(output_dir, filename)
            
            # Skip if file already exists
            if os.path.exists(output_path):
                return True
            
            # Handle different URL types
            if image_url.startswith('/'):
                # Relative URL - prepend Label Studio URL
                full_url = f"{self.url}{image_url}"
                headers = self.headers
                
                # Download image
                response = requests.get(full_url, headers=headers, stream=True, timeout=30)
                response.raise_for_status()
            elif image_url.startswith('http') and self.url in image_url:
                # Label Studio hosted URL
                full_url = image_url
                headers = self.headers
                
                # Download image
                response = requests.get(full_url, headers=headers, stream=True, timeout=30)
                response.raise_for_status()
            elif image_url.startswith('gs://'):
                # Try authenticated GCS download first
                if self.download_from_gcs(image_url, output_path):
                    return True
                    
                # Fallback to public HTTPS access
                gcs_path = image_url[5:]  # Remove 'gs://' prefix
                full_url = f"https://storage.googleapis.com/{gcs_path}"
                headers = {}
                
                # Download image
                response = requests.get(full_url, headers=headers, stream=True, timeout=30)
                response.raise_for_status()
            elif 'storage.googleapis.com' in image_url or 'gcs' in image_url:
                # Direct GCS URL - no auth needed
                full_url = image_url
                headers = {}
                
                # Download image
                response = requests.get(full_url, headers=headers, stream=True, timeout=30)
                response.raise_for_status()
            else:
                # Other external URLs
                full_url = image_url
                headers = {}
                
                # Download image
                response = requests.get(full_url, headers=headers, stream=True, timeout=30)
                response.raise_for_status()
            
            # Save image (for non-GCS cases)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading image for task {image_info.get('task_id', 'unknown')}: {e}")
            return False
    
    def download_images_batch(
        self,
        image_info_list: List[Dict[str, str]],
        output_dir: str,
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """
        Download images in parallel
        
        Args:
            image_info_list: List of image information dictionaries
            output_dir: Output directory
            max_workers: Number of parallel download threads
            
        Returns:
            Download results summary
        """
        results = {
            'successful': [],
            'failed': [],
            'total': len(image_info_list)
        }
        
        os.makedirs(output_dir, exist_ok=True)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_info = {
                executor.submit(self.download_single_image, info, output_dir): info
                for info in image_info_list
            }
            
            # Process completed downloads with progress bar
            with tqdm(total=len(image_info_list), desc="Downloading images") as pbar:
                for future in as_completed(future_to_info):
                    info = future_to_info[future]
                    success = future.result()
                    
                    if success:
                        output_path = os.path.join(output_dir, info['filename'])
                        results['successful'].append({
                            'task_id': info['task_id'],
                            'filename': info['filename'],
                            'local_path': output_path,
                            'blob_name': info['filename']  # For compatibility with GCS format
                        })
                    else:
                        results['failed'].append(info)
                    
                    pbar.update(1)
        
        self.logger.info(
            f"Download complete: {len(results['successful'])} successful, "
            f"{len(results['failed'])} failed"
        )
        
        return results
    
    def create_task_mapping(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Create mapping from local image paths to Label Studio task IDs
        
        Args:
            results: Download results from download_images_batch
            
        Returns:
            Dictionary mapping local_path -> task_id
        """
        mapping = {}
        for item in results['successful']:
            mapping[item['local_path']] = str(item['task_id'])
        
        return mapping
    
    def download_from_gcs(self, gcs_url: str, output_path: str) -> bool:
        """
        Download file from Google Cloud Storage using authenticated client
        
        Args:
            gcs_url: GCS URL (gs://bucket/path/to/file)
            output_path: Local output path
            
        Returns:
            True if successful, False otherwise
        """
        if not self.gcs_client:
            return False
            
        try:
            # Parse GCS URL: gs://bucket-name/path/to/file
            if not gcs_url.startswith('gs://'):
                return False
                
            gcs_path = gcs_url[5:]  # Remove 'gs://' prefix
            bucket_name, blob_path = gcs_path.split('/', 1)
            
            # Get bucket and blob
            bucket = self.gcs_client.bucket(bucket_name)
            blob = bucket.blob(blob_path)
            
            # Create output directory
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Download blob to file
            blob.download_to_filename(output_path)
            return True
            
        except Exception as e:
            self.logger.error(f"Error downloading from GCS {gcs_url}: {e}")
            return False


def get_existing_images(config: dict) -> bool:
    """
    Download images from existing Label Studio project
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize downloader
        downloader = LabelStudioImageDownloader(
            url=config['label_studio']['url'],
            api_key=config['label_studio']['api_key'],
            project_id=config['label_studio']['project_id']
        )
        
        # Get tasks from Label Studio project
        logger.info("Fetching tasks from Label Studio project...")
        tasks = downloader.get_project_tasks()
        
        if not tasks:
            logger.error("No tasks found in project")
            return False
        
        # Extract image URLs
        image_info_list = downloader.extract_image_urls(tasks)
        
        if not image_info_list:
            logger.error("No images found in tasks")
            return False
        
        # Download images
        logger.info(f"Starting download of {len(image_info_list)} images...")
        output_dir = config['paths']['raw_images']
        
        results = downloader.download_images_batch(
            image_info_list=image_info_list,
            output_dir=output_dir,
            max_workers=config['processing']['max_workers']
        )
        
        # Create task mapping for later use
        task_mapping = downloader.create_task_mapping(results)
        
        # Save results and mapping
        results_file = os.path.join(output_dir, 'download_results.yaml')
        mapping_file = os.path.join(output_dir, 'task_mapping.json')
        
        with open(results_file, 'w') as f:
            yaml.dump(results, f)
        
        with open(mapping_file, 'w') as f:
            json.dump(task_mapping, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        logger.info(f"Task mapping saved to: {mapping_file}")
        
        return len(results['successful']) > 0
        
    except Exception as e:
        logger.error(f"Error downloading images from Label Studio: {e}")
        return False


def main():
    """Main entry point"""
    print("Get Images from Existing Label Studio Project")
    print("=" * 50)
    
    # Load configuration
    config = load_config()
    
    # Setup logging
    setup_logging(config)
    logger = logging.getLogger(__name__)
    
    # Check if stage is enabled
    if not config['stages'].get('get_existing_images', False):
        logger.info("Get existing images stage is disabled")
        return
    
    # Check if existing project mode is enabled
    if not config.get('existing_project', {}).get('enabled', False):
        logger.info("Existing project mode is disabled")
        return
    
    # Create directories
    os.makedirs(config['paths']['raw_images'], exist_ok=True)
    
    logger.info("Starting image download from existing Label Studio project...")
    
    # Download images
    success = get_existing_images(config)
    
    if success:
        logger.info("Images downloaded successfully from Label Studio project!")
        print("\n✅ Images downloaded successfully from Label Studio project!")
        print(f"📁 Images saved to: {config['paths']['raw_images']}")
    else:
        logger.error("Failed to download images from Label Studio project!")
        print("\n❌ Failed to download images from Label Studio project!")
        sys.exit(1)


if __name__ == "__main__":
    main() 