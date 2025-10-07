"""
Google Cloud Storage Handler
Handles downloading images from GCS bucket
"""

import os
import logging
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from google.cloud import storage
from google.cloud.exceptions import NotFound, GoogleCloudError
from tqdm import tqdm


class GCSHandler:
    """Handle Google Cloud Storage operations for image downloading"""
    
    def __init__(self, project_id: str, bucket_name: str, credentials_path: str):
        """
        Initialize GCS handler
        
        Args:
            project_id: Google Cloud project ID
            bucket_name: GCS bucket name
            credentials_path: Path to GCS credentials JSON file
        """
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.credentials_path = credentials_path
        self.logger = logging.getLogger(__name__)
        
        # Initialize GCS client
        try:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
            self.client = storage.Client(project=project_id)
            self.bucket = self.client.bucket(bucket_name)
            self.logger.info(f"Successfully connected to GCS bucket: {bucket_name}")
        except Exception as e:
            self.logger.error(f"Failed to initialize GCS client: {e}")
            raise
    
    def list_images(self, prefix: str = "", extensions: List[str] = None) -> List[str]:
        """
        List all image files in the bucket
        
        Args:
            prefix: Optional prefix to filter blobs
            extensions: List of allowed file extensions
            
        Returns:
            List of blob names (image paths)
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        extensions = [ext.lower() for ext in extensions]
        image_paths = []
        
        try:
            blobs = self.bucket.list_blobs(prefix=prefix)
            for blob in blobs:
                # Check if file has image extension
                if any(blob.name.lower().endswith(ext) for ext in extensions):
                    image_paths.append(blob.name)
            
            self.logger.info(f"Found {len(image_paths)} images in bucket")
            return image_paths
            
        except Exception as e:
            self.logger.error(f"Error listing images: {e}")
            raise
    
    def download_image(self, blob_name: str, local_path: str) -> bool:
        """
        Download a single image from GCS
        
        Args:
            blob_name: Name of the blob in GCS
            local_path: Local path to save the image
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download the blob
            blob = self.bucket.blob(blob_name)
            blob.download_to_filename(local_path)
            
            return True
            
        except NotFound:
            self.logger.warning(f"Blob not found: {blob_name}")
            return False
        except Exception as e:
            self.logger.error(f"Error downloading {blob_name}: {e}")
            return False
    
    def download_images_batch(
        self, 
        blob_names: List[str], 
        local_dir: str,
        max_workers: int = 4,
        preserve_structure: bool = True
    ) -> Dict[str, Any]:
        """
        Download multiple images in parallel
        
        Args:
            blob_names: List of blob names to download
            local_dir: Local directory to save images
            max_workers: Number of parallel download threads
            preserve_structure: Whether to preserve GCS directory structure
            
        Returns:
            Dictionary with download results
        """
        results = {
            'successful': [],
            'failed': [],
            'total': len(blob_names)
        }
        
        # Create local directory
        os.makedirs(local_dir, exist_ok=True)
        
        def download_single(blob_name: str) -> tuple:
            """Download single image and return result"""
            if preserve_structure:
                # Preserve directory structure
                local_path = os.path.join(local_dir, blob_name)
            else:
                # Flatten structure - use only filename
                filename = os.path.basename(blob_name)
                local_path = os.path.join(local_dir, filename)
            
            success = self.download_image(blob_name, local_path)
            return blob_name, local_path, success
        
        # Download images in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_blob = {
                executor.submit(download_single, blob_name): blob_name 
                for blob_name in blob_names
            }
            
            # Process completed downloads with progress bar
            with tqdm(total=len(blob_names), desc="Downloading images") as pbar:
                for future in as_completed(future_to_blob):
                    blob_name, local_path, success = future.result()
                    
                    if success:
                        results['successful'].append({
                            'blob_name': blob_name,
                            'local_path': local_path
                        })
                    else:
                        results['failed'].append(blob_name)
                    
                    pbar.update(1)
        
        self.logger.info(
            f"Download complete: {len(results['successful'])} successful, "
            f"{len(results['failed'])} failed"
        )
        
        return results
    
    def get_image_metadata(self, blob_name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an image blob
        
        Args:
            blob_name: Name of the blob
            
        Returns:
            Dictionary with metadata or None if not found
        """
        try:
            blob = self.bucket.blob(blob_name)
            blob.reload()
            
            return {
                'name': blob.name,
                'size': blob.size,
                'content_type': blob.content_type,
                'time_created': blob.time_created,
                'updated': blob.updated,
                'md5_hash': blob.md5_hash,
                'crc32c': blob.crc32c
            }
            
        except NotFound:
            self.logger.warning(f"Blob not found: {blob_name}")
            return None
        except Exception as e:
            self.logger.error(f"Error getting metadata for {blob_name}: {e}")
            return None
    
    def check_bucket_exists(self) -> bool:
        """Check if the bucket exists and is accessible"""
        try:
            self.bucket.reload()
            return True
        except NotFound:
            self.logger.error(f"Bucket {self.bucket_name} not found")
            return False
        except Exception as e:
            self.logger.error(f"Error accessing bucket: {e}")
            return False 