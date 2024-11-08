# Standard library imports for file/path operations, logging, and time
import os
import yaml  # For reading configuration files
from face_detector import PracticalFaceDetectionTest, TestConfig  # Our custom face detection classes
import logging  # For tracking test execution and errors
from datetime import datetime  # For timestamps in logs and filenames
from utils.path_config import PathConfig  # Import from utils package
from face_detector import PracticalFaceDetectionTest, TestConfig
from utils.system_resources import SystemResourceChecker

def load_configurations(config_path: str) -> list:
    """Load and parse test configurations from YAML file, creating aligned variants
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        list: List of TestConfig objects containing test parameters
    """
    # Open and parse YAML file
    with open(config_path, 'r') as f:
        config_data = yaml.safe_load(f)
    
    configs = []
    
    # Process each configuration and create aligned variant
    for config_name, config_dict in config_data['test_configurations'].items():
        # Create base (non-aligned) configuration
        configs.append(TestConfig(
            name=config_dict['name'],
            resize_dimensions=tuple(config_dict['resize_dimensions']),
            batch_size=config_dict['batch_size'],
            num_workers=config_dict['num_workers'],
            confidence_threshold=config_dict['confidence_threshold'],
            align=False  # Base configuration is non-aligned
        ))
        
        # Create aligned variant
        configs.append(TestConfig(
            name=f"{config_dict['name']}_Aligned",  # Add _Aligned suffix
            resize_dimensions=tuple(config_dict['resize_dimensions']),
            batch_size=config_dict['batch_size'],
            num_workers=config_dict['num_workers'],
            confidence_threshold=config_dict['confidence_threshold'],
            align=True  # Aligned variant
        ))
    
    logging.info(f"Created {len(configs)} configurations including aligned variants")
    return configs

def setup_logging():
    """Configure logging settings for test execution
    
    Creates:
        - Log directory if it doesn't exist
        - Log file with timestamp
        - Console output for immediate feedback
        
    Format:
        timestamp - log_level - message
    """
    # Ensure logs directory exists
    if not os.path.exists('data/logs'):
        os.makedirs('data/logs')
    
    # Create unique timestamp for this test run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Configure logging to both file and console
    logging.basicConfig(
        level=logging.INFO,  # Set minimum log level
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            # File handler - writes to log file
            logging.FileHandler(f'data/logs/test_run_{timestamp}.log'),
            # Stream handler - writes to console
            logging.StreamHandler()
        ]
    )

def main():
    """Main function to run face detection tests
    
    Workflow:
    1. Set up logging
    2. Configure file paths
    3. Load test configurations
    4. Initialize face detection tester
    5. Run tests and save results
    6. Analyze and visualize results
    
    Handles errors gracefully and logs all steps
    """
    # Initialize logging for this test run
    setup_logging()
    
    # Check system resources before running tests
    logging.info("Checking system resources...")
    checker = SystemResourceChecker()
    checker.print_summary()

    # Get recommendations for configuration
    recommendations = checker.get_recommendations()
    logging.info("System recommendations:")
    for rec in recommendations:
        logging.info(f"- {rec}")


    # Initialize path configuration
    paths = PathConfig()
    
    
    # Load configurations using proper paths
    config_path = paths.get_config_path('test_configs.yaml')
    test_folder = paths.get_test_data_path()
    
    # Load test configurations from YAML
    logging.info("Loading test configurations...")
    configs = load_configurations(config_path)
    
    # Initialize the face detection tester
    logging.info(f"Initializing tester with test folder: {test_folder}")
    tester = PracticalFaceDetectionTest(test_folder, config_path, use_gpu=True)
    
    # Execute tests with error handling
    logging.info("Starting tests...")
    try:
        # Run all configured tests
        results = tester.run_tests(configs)
        
        # Generate unique timestamp for output files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save raw results to CSV file using path config
        results_path = paths.get_results_path(f'test_results_{timestamp}.csv')
        results.to_csv(results_path, index=False)
        logging.info(f"Results saved to {results_path}")
        
        # Generate analysis and visualizations
        logging.info("Analyzing results...")
        tester.analyze_results(results)
        
    except Exception as e:
        # Log any errors that occur during execution
        logging.error(f"Error during test execution: {str(e)}")
        # Re-raise the exception after logging
        raise

if __name__ == "__main__":
    main()