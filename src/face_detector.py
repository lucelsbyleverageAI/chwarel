import os
import pandas as pd
import numpy as np
import time
import torch
import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import logging
from datetime import datetime
import yaml
from pathlib import Path
import seaborn as sns
from utils.path_config import PathConfig


@dataclass
class TestConfig:
    """Configuration settings for face detection testing
    
    Attributes:
        name: Identifier for this configuration
        resize_dimensions: Target image size (width, height)
        batch_size: Number of images to process in one batch
        num_workers: Number of parallel processes
        confidence_threshold: Minimum confidence score for valid detections
        align: Whether to align detected faces
    """
    name: str
    resize_dimensions: tuple[int, int]
    batch_size: int
    num_workers: int
    confidence_threshold: float
    align: bool

class PracticalFaceDetectionTest:
    def __init__(self, test_folder: str, config_path: str, use_gpu: bool = False):
        """Initialize the face detection testing framework
        
        Args:
            test_folder: Path to directory containing test images
            config_path: Path to configuration file
            use_gpu: Whether to use GPU acceleration if available
        """
        self.test_folder = test_folder
        self.device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get ground truth from config
        self.ground_truth = self.config.get('ground_truth', {})
        
        # Initialize path configuration
        self.paths = PathConfig()
        
        # Set up logging
        self._setup_logging()

    def _setup_logging(self):
        """Configure logging based on config settings"""
        log_level = self.config.get('logging_settings', {}).get('log_level', 'INFO')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.paths.get_log_path(f'face_detection_{timestamp}.log')
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def load_and_preprocess(self, image_path: str, resize_dimensions: tuple[int, int]) -> np.ndarray:
        """Load and optionally resize an image
        
        Args:
            image_path: Path to image file
            resize_dimensions: Target dimensions (width, height)
            
        Returns:
            Preprocessed image as numpy array
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # Resize if dimensions are specified
        if resize_dimensions != (0, 0):
            img = cv2.resize(img, resize_dimensions)
            
        return img

    def detect_faces(self, image_path: str, config: TestConfig) -> Dict:
        """Detect faces in a single image
        
        Args:
            image_path: Path to image file
            config: TestConfig object with processing parameters
            
        Returns:
            Dictionary containing detection results and metrics
        """
        start_time = time.time()
        try:
            # Detect faces using DeepFace
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend='retinaface',
                align=config.align,
                enforce_detection=False
            )
            
            # Filter faces based on confidence threshold
            valid_faces = [
                face for face in faces 
                if face.get("confidence", 0) > config.confidence_threshold
            ]
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Get expected number of faces (default to 1 if not specified)
            expected_faces = self.ground_truth.get(os.path.basename(image_path), 1)
            
            # Get image dimensions
            img = cv2.imread(image_path)
            img_dimensions = img.shape[:2]
            
            return {
                "image_name": os.path.basename(image_path),
                "config_name": config.name,
                "detected_faces": len(valid_faces),
                "expected_faces": expected_faces,
                "accurate_detection": len(valid_faces) == expected_faces,
                "processing_time": processing_time,
                "image_dimensions": img_dimensions,
                "confidences": [face.get("confidence", 0) for face in valid_faces],
                "align": config.align
            }
            
        except Exception as e:
            logging.error(f"Error processing {image_path}: {str(e)}")
            return {
                "image_name": os.path.basename(image_path),
                "config_name": config.name,
                "detected_faces": 0,
                "expected_faces": self.ground_truth.get(os.path.basename(image_path), 1),
                "accurate_detection": False,
                "processing_time": time.time() - start_time,
                "image_dimensions": (0, 0),
                "confidences": [],
                "align": config.align
            }

    def process_batch(self, batch_paths: List[str], config: TestConfig) -> List[Dict]:
        """Process a batch of images together
        
        Args:
            batch_paths: List of paths to images in this batch
            config: TestConfig object with processing parameters
            
        Returns:
            List of dictionaries containing results for each image
        """
        results = []
        for path in batch_paths:
            try:
                result = self.detect_faces(path, config)
                results.append(result)
            except Exception as e:
                logging.error(f"Error in batch processing {path}: {str(e)}")
        return results

    def run_tests(self, configs: List[TestConfig]) -> pd.DataFrame:
        """Run face detection tests with all configurations
        
        Args:
            configs: List of TestConfig objects to test
            
        Returns:
            DataFrame containing all test results
        """
        all_results = []
        image_paths = [
            os.path.join(self.test_folder, f) 
            for f in os.listdir(self.test_folder)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ]

        for config in configs:
            logging.info(f"Testing configuration: {config.name}")
            
            # Create batches of images
            batches = [
                image_paths[i:i + config.batch_size]
                for i in range(0, len(image_paths), config.batch_size)
            ]

            if config.num_workers > 1:
                # Parallel processing using ProcessPoolExecutor
                with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
                    process_fn = partial(self.process_batch, config=config)
                    batch_results = executor.map(process_fn, batches)
                    
                    for batch_result in batch_results:
                        all_results.extend(batch_result)
            else:
                # Sequential processing
                for batch in batches:
                    results = self.process_batch(batch, config)
                    all_results.extend(results)

        return pd.DataFrame(all_results)

    def analyze_results(self, results_df: pd.DataFrame) -> None:
        """Analyze and visualize test results with proper alignment grouping
        
        Args:
            results_df: DataFrame containing test results
        """
        if results_df.empty:
            logging.warning("No results to analyze!")
            return
                
        # Get paths
        paths = PathConfig()
        
        # Calculate metrics per configuration and alignment setting
        metrics = results_df.groupby(['config_name', 'align']).agg({
            'processing_time': ['mean', 'std', 'max'],
            'accurate_detection': 'mean',
        }).round(3)
        
        # Print results
        print("\nTest Results Summary:")
        print("=====================")
        print("\nConfiguration Metrics:")
        print(metrics)
        
        # Create figure with increased size for better readability
        plt.figure(figsize=(15, 8))
        
        # Processing Time Boxplot
        plt.subplot(1, 2, 1)
        
        # Create boxplot with alignment distinction
        sns.boxplot(
            data=results_df,
            x='config_name',
            y='processing_time',
            hue='align',
            palette=['lightblue', 'lightgreen'],
        )
        
        plt.title('Processing Time by Configuration and Alignment', pad=20)
        plt.xlabel('Configuration')
        plt.ylabel('Processing Time (seconds)')
        plt.xticks(rotation=45)
        plt.legend(title='Aligned', labels=['False', 'True'])
        
        # Accuracy Bar Plot
        plt.subplot(1, 2, 2)
        accuracy_data = results_df.groupby(['config_name', 'align'])['accurate_detection'].mean()
        accuracy_df = accuracy_data.unstack()
        
        # Plot grouped bar chart for accuracy
        accuracy_df.plot(
            kind='bar',
            width=0.8,
            color=['lightblue', 'lightgreen'],
        )
        
        plt.title('Detection Accuracy by Configuration and Alignment', pad=20)
        plt.xlabel('Configuration')
        plt.ylabel('Accuracy Rate')
        plt.xticks(rotation=45)
        plt.legend(title='Aligned', labels=['False', 'True'])
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = paths.get_results_path(f'detailed_results_{timestamp}.csv')
        results_df.to_csv(results_path, index=False)
        logging.info(f"Detailed results saved to {results_path}")
        
        # Save plot
        plot_path = paths.get_results_path(f'results_plot_{timestamp}.png')
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        logging.info(f"Plot saved to {plot_path}")
        
        # Show plot
        plt.show()
        
        # Print detailed analysis
        print("\nDetailed Analysis:")
        print("=================")
        
        # Processing Time Analysis
        print("\nProcessing Time Analysis:")
        for (config, align), group in results_df.groupby(['config_name', 'align']):
            print(f"\n{config} (Aligned: {align}):")
            print(f"  Mean processing time: {group['processing_time'].mean():.3f}s")
            print(f"  Min processing time:  {group['processing_time'].min():.3f}s")
            print(f"  Max processing time:  {group['processing_time'].max():.3f}s")
            print(f"  Std Dev:             {group['processing_time'].std():.3f}s")
        
        # Accuracy Analysis
        print("\nAccuracy Analysis:")
        for (config, align), group in results_df.groupby(['config_name', 'align']):
            print(f"\n{config} (Aligned: {align}):")
            print(f"  Accuracy rate: {group['accurate_detection'].mean()*100:.1f}%")
            print(f"  Total images: {len(group)}")
            print(f"  Correct detections: {group['accurate_detection'].sum()}")



def load_configurations(config_path: str) -> list:
    """Load and parse test configurations from YAML file, creating aligned variants
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        list: List of TestConfig objects containing test parameters
        
    Note:
        Expected YAML structure:
        test_configurations:
          config_name:
            name: str
            resize_dimensions: [width, height]
            batch_size: int
            num_workers: int
            confidence_threshold: float
        
        For each configuration in YAML, creates two TestConfig objects:
        1. Original configuration with align=False
        2. Aligned variant with align=True and "_Aligned" suffix
    """
    # Open and parse YAML file
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    # Initialize empty list to store configurations
    configs = []
    
    # Convert each YAML configuration into two TestConfig objects (aligned and non-aligned)
    for config_name, config_dict in config_data['test_configurations'].items():
        # Create non-aligned version
        configs.append(TestConfig(
            name=config_dict['name'],
            resize_dimensions=tuple(config_dict['resize_dimensions']),
            batch_size=config_dict['batch_size'],
            num_workers=config_dict['num_workers'],
            confidence_threshold=config_dict['confidence_threshold'],
            align=False  # Non-aligned version
        ))
        
        # Create aligned version with "_Aligned" suffix
        configs.append(TestConfig(
            name=f"{config_dict['name']}_Aligned",
            resize_dimensions=tuple(config_dict['resize_dimensions']),
            batch_size=config_dict['batch_size'],
            num_workers=config_dict['num_workers'],
            confidence_threshold=config_dict['confidence_threshold'],
            align=True  # Aligned version
        ))
        
        # Log the configuration creation
        logging.debug(f"Created configurations for {config_dict['name']}")
        
    logging.info(f"Loaded {len(configs)} configurations (including aligned variants)")
    return configs


def main():
    # Get project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define paths
    config_path = os.path.join(project_root, 'configs', 'test_configs.yaml')
    test_folder = os.path.join(project_root, 'data', 'test_dataset')
    output_folder = os.path.join(project_root, 'data', 'sample_outputs')
    
    # Load configurations from YAML
    configs = load_configurations(config_path)
    
    # Initialize tester
    tester = PracticalFaceDetectionTest(test_folder, use_gpu=True)
    
    # Run tests and analyze results
    results = tester.run_tests(configs)
    tester.analyze_results(results)

if __name__ == "__main__":
    main()