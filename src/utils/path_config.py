import os
from pathlib import Path
import logging

class PathConfig:
    """Configure all paths used in the project"""
    def __init__(self, project_root: str = None):
        """Initialize path configuration
        
        Args:
            project_root: Optional path to project root. If None, will be automatically
                         determined based on file structure.
        """
        if project_root is None:
            # Get project root by going up two levels from this file
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent
            
        self.project_root = Path(project_root)
        
        # Define standard directory paths
        self.src_dir = self.project_root / 'src'
        self.data_dir = self.project_root / 'data'
        self.results_dir = self.data_dir / 'results'
        self.logs_dir = self.data_dir / 'logs'
        self.test_dataset_dir = self.data_dir / 'test_dataset'
        self.configs_dir = self.project_root / 'configs'
        
        # Create directories if they don't exist
        self._create_directories()
        
    def _create_directories(self):
        """Create all required directories if they don't exist"""
        directories = [
            self.data_dir,
            self.results_dir,
            self.logs_dir,
            self.test_dataset_dir,
            self.configs_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
                
    def get_results_path(self, filename: str) -> Path:
        """Get full path for a results file"""
        return self.results_dir / filename
        
    def get_log_path(self, filename: str) -> Path:
        """Get full path for a log file"""
        return self.logs_dir / filename
        
    def get_config_path(self, filename: str) -> Path:
        """Get full path for a configuration file"""
        return self.configs_dir / filename
        
    def get_test_data_path(self, filename: str = None) -> Path:
        """Get path for test dataset directory or specific test file"""
        if filename:
            return self.test_dataset_dir / filename
        return self.test_dataset_dir