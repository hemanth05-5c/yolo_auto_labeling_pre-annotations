"""
Label Studio Client Utilities
Handles Label Studio API interactions and prediction uploads
"""

import os
import logging
import requests
import json
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time

try:
    from label_studio_sdk import Client
except ImportError:
    raise ImportError("label-studio-sdk package required. Install with: pip install label-studio-sdk")


class LabelStudioClient:
    """Handle Label Studio API operations and prediction uploads"""
    
    def __init__(self, url: str, api_key: str, project_id: int):
        """
        Initialize Label Studio client
        
        Args:
            url: Label Studio server URL
            api_key: API key for authentication
            project_id: Project ID to work with
        """
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.project_id = project_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize SDK client
        try:
            self.client = Client(url=url, api_key=api_key)
            self.project = self.client.get_project(project_id)
            self.logger.info(f"Successfully connected to Label Studio project {project_id}")
        except Exception as e:
            self.logger.error(f"Failed to initialize Label Studio client: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test connection to Label Studio
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            # Test basic connection
            response = requests.get(f"{self.url}/api/projects/{self.project_id}/", 
                                  headers={"Authorization": f"Token {self.api_key}"})
            
            if response.status_code == 200:
                project_info = response.json()
                self.logger.info(f"Connected to project: {project_info.get('title', 'Unknown')}")
                return True
            else:
                self.logger.error(f"Connection test failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Connection test failed: {e}")
            return False
    
    def get_project_info(self) -> Optional[Dict[str, Any]]:
        """
        Get project information
        
        Returns:
            Project information dictionary or None if failed
        """
        try:
            project_info = self.project.get_params()
            return project_info
        except Exception as e:
            self.logger.error(f"Error getting project info: {e}")
            return None
    
    def import_tasks_batch(
        self,
        tasks: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Import tasks with predictions in batches
        
        Args:
            tasks: List of Label Studio tasks with predictions
            batch_size: Number of tasks to import per batch
            
        Returns:
            Import results summary
        """
        results = {
            'successful_batches': 0,
            'failed_batches': 0,
            'total_tasks_imported': 0,
            'errors': []
        }
        
        # Split tasks into batches
        batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]
        
        self.logger.info(f"Importing {len(tasks)} tasks in {len(batches)} batches...")
        
        with tqdm(total=len(batches), desc="Uploading batches") as pbar:
            for i, batch in enumerate(batches):
                try:
                    # Import batch using SDK
                    imported_tasks = self.project.import_tasks(batch)
                    
                    results['successful_batches'] += 1
                    results['total_tasks_imported'] += len(batch)
                    
                    self.logger.debug(f"Imported batch {i+1}/{len(batches)} ({len(batch)} tasks)")
                    
                    # Add small delay to avoid overwhelming the server
                    time.sleep(0.1)
                    
                except Exception as e:
                    results['failed_batches'] += 1
                    error_msg = f"Batch {i+1} failed: {e}"
                    results['errors'].append(error_msg)
                    self.logger.error(error_msg)
                
                pbar.update(1)
        
        self.logger.info(f"Import completed: {results['successful_batches']} successful, "
                        f"{results['failed_batches']} failed batches")
        
        return results
    
    def import_tasks_sequential(
        self,
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Import tasks sequentially (more reliable for large datasets)
        
        Args:
            tasks: List of Label Studio tasks
            
        Returns:
            Import results summary
        """
        results = {
            'successful_tasks': 0,
            'failed_tasks': 0,
            'errors': []
        }
        
        self.logger.info(f"Importing {len(tasks)} tasks sequentially...")
        
        with tqdm(total=len(tasks), desc="Uploading tasks") as pbar:
            for i, task in enumerate(tasks):
                try:
                    # Import single task
                    imported_task = self.project.import_tasks([task])
                    results['successful_tasks'] += 1
                    
                except Exception as e:
                    results['failed_tasks'] += 1
                    error_msg = f"Task {i+1} failed: {e}"
                    results['errors'].append(error_msg)
                    self.logger.error(error_msg)
                
                pbar.update(1)
                
                # Small delay to avoid rate limiting
                if i % 10 == 0:
                    time.sleep(0.1)
        
        self.logger.info(f"Sequential import completed: {results['successful_tasks']} successful, "
                        f"{results['failed_tasks']} failed tasks")
        
        return results
    
    def create_predictions_from_tasks(
        self,
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create predictions from tasks that contain prediction data
        
        Args:
            tasks: List of tasks with embedded predictions
            
        Returns:
            Creation results summary
        """
        results = {
            'predictions_created': 0,
            'failed_predictions': 0,
            'errors': []
        }
        
        self.logger.info(f"Creating predictions for {len(tasks)} tasks...")
        
        # First, we need to import tasks without predictions to get task IDs
        tasks_without_predictions = []
        predictions_data = []
        
        for task in tasks:
            # Extract predictions
            if 'predictions' in task and task['predictions']:
                predictions_data.append(task['predictions'])
            else:
                predictions_data.append([])
            
            # Create task without predictions
            task_data = {'data': task['data']}
            tasks_without_predictions.append(task_data)
        
        try:
            # Import tasks first
            imported_tasks = self.project.import_tasks(tasks_without_predictions)
            self.logger.info(f"Imported {len(imported_tasks)} tasks")
            
            # Now create predictions for each task
            with tqdm(total=len(imported_tasks), desc="Creating predictions") as pbar:
                for i, (imported_task, predictions) in enumerate(zip(imported_tasks, predictions_data)):
                    if predictions:
                        try:
                            for prediction in predictions:
                                # Add task ID to prediction
                                prediction['task'] = imported_task['id']
                                
                                # Create prediction via API
                                self._create_prediction_api(prediction)
                                results['predictions_created'] += 1
                                
                        except Exception as e:
                            results['failed_predictions'] += 1
                            error_msg = f"Prediction creation failed for task {imported_task['id']}: {e}"
                            results['errors'].append(error_msg)
                            self.logger.error(error_msg)
                    
                    pbar.update(1)
            
        except Exception as e:
            self.logger.error(f"Error in prediction creation workflow: {e}")
            results['errors'].append(str(e))
        
        self.logger.info(f"Prediction creation completed: {results['predictions_created']} successful, "
                        f"{results['failed_predictions']} failed")
        
        return results
    
    def _create_prediction_api(self, prediction: Dict[str, Any]):
        """
        Create a single prediction via API
        
        Args:
            prediction: Prediction data
        """
        url = f"{self.url}/api/predictions/"
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, json=prediction, headers=headers)
        
        if response.status_code not in [200, 201]:
            raise Exception(f"API request failed: {response.status_code} - {response.text}")
    
    def get_tasks_count(self) -> int:
        """
        Get number of tasks in the project
        
        Returns:
            Number of tasks
        """
        try:
            tasks = self.project.get_tasks()
            return len(tasks)
        except Exception as e:
            self.logger.error(f"Error getting tasks count: {e}")
            return 0
    
    def get_predictions_count(self) -> int:
        """
        Get number of predictions in the project
        
        Returns:
            Number of predictions
        """
        try:
            # Get predictions via API
            url = f"{self.url}/api/predictions/"
            headers = {"Authorization": f"Token {self.api_key}"}
            params = {"project": self.project_id}
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'results' in data:
                    return len(data['results'])
                elif isinstance(data, list):
                    return len(data)
            
            return 0
            
        except Exception as e:
            self.logger.error(f"Error getting predictions count: {e}")
            return 0
    
    def setup_local_file_serving(self, local_files_root: str) -> bool:
        """
        Verify local file serving setup
        
        Args:
            local_files_root: Root directory for local files
            
        Returns:
            True if setup is correct, False otherwise
        """
        try:
            # Check if local files serving is enabled
            test_file = "test_image.jpg"  # This would be an actual image in your dataset
            test_url = f"{self.url}/data/local-files/?d={test_file}"
            
            response = requests.head(test_url)
            
            if response.status_code == 200:
                self.logger.info("Local file serving is properly configured")
                return True
            else:
                self.logger.warning(f"Local file serving test failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.warning(f"Could not verify local file serving: {e}")
            return False
    
    def validate_project_config(self, expected_label_config: str = None) -> Dict[str, Any]:
        """
        Validate project configuration
        
        Args:
            expected_label_config: Expected labeling configuration XML
            
        Returns:
            Validation results
        """
        validation_results = {
            'config_valid': False,
            'warnings': [],
            'errors': []
        }
        
        try:
            project_info = self.get_project_info()
            
            if not project_info:
                validation_results['errors'].append("Could not retrieve project information")
                return validation_results
            
            # Check label configuration
            current_config = project_info.get('label_config', '')
            
            if expected_label_config:
                if expected_label_config.strip() == current_config.strip():
                    validation_results['config_valid'] = True
                else:
                    validation_results['warnings'].append("Label configuration differs from expected")
            
            # Check if project accepts predictions
            if 'PolygonLabels' in current_config or 'RectangleLabels' in current_config:
                validation_results['config_valid'] = True
            else:
                validation_results['warnings'].append("Project may not support object detection predictions")
            
            self.logger.info(f"Project validation: {len(validation_results['errors'])} errors, "
                           f"{len(validation_results['warnings'])} warnings")
            
        except Exception as e:
            validation_results['errors'].append(f"Validation failed: {e}")
        
        return validation_results


def create_labelstudio_project(
    client: LabelStudioClient,
    project_config: Dict[str, Any]
) -> Optional[int]:
    """
    Create a new Label Studio project
    
    Args:
        client: Label Studio client instance
        project_config: Project configuration
        
    Returns:
        Project ID if successful, None otherwise
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Create project using SDK
        new_project = client.client.start_project(
            title=project_config['title'],
            label_config=project_config['labeling_config'],
            description=project_config.get('description', '')
        )
        
        project_id = new_project.get_params()['id']
        logger.info(f"Created new project with ID: {project_id}")
        
        return project_id
        
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        return None 