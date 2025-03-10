import ee
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

def initializeEarthEngine(verbose=True):
    """
    Initialize Google Earth Engine using service account credentials.
    
    This function loads credentials from environment variables defined in a .env file
    and uses them to initialize the Earth Engine API.
    
    Args:
        verbose (bool): Whether to print status messages. Default is True.
        
    Returns:
        bool: True if initialization was successful
        
    Raises:
        ValueError: If required environment variables are not set
        FileNotFoundError: If the key file does not exist
    """
    # Set up logging
    logger = logging.getLogger(__name__)
    
    # Check if Earth Engine is already initialized
    try:
        ee.Image(0)
        if verbose:
            logger.info("Google Earth Engine is already initialized.")
        return True
    except ee.EEException:
        pass  # Not initialized, continue with initialization
    
    # Define path to .env file in the outils directory
    current_dir = Path(__file__).parent
    dotenv_path = current_dir / '.env'
    
    # Check if .env file exists
    if not dotenv_path.exists():
        raise FileNotFoundError(f"No .env file found at {dotenv_path}. Please create one with SERVICE_ACCOUNT and KEY_PATH variables.")
    
    # Load environment variables from .env file
    load_dotenv(dotenv_path)
    
    # Get service account email and key file path from environment variables
    service_account = os.getenv('SERVICE_ACCOUNT')
    key_path = os.getenv('KEY_PATH')
    
    # Validate environment variables
    if not service_account or not key_path:
        raise ValueError("Environment variables SERVICE_ACCOUNT or KEY_PATH are not set.")
    
    # Check if key file exists - handle relative paths from outils directory
    key_path_obj = Path(key_path)
    if not key_path_obj.is_absolute():
        key_path_obj = current_dir / key_path_obj
        
    if not key_path_obj.exists():
        raise FileNotFoundError(f"Key file not found at path: {key_path_obj}")
    
    try:
        # Initialize Earth Engine with credentials
        credentials = ee.ServiceAccountCredentials(service_account, str(key_path_obj))
        ee.Initialize(credentials)
        
        if verbose:
            logger.info("Google Earth Engine initialized successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize Google Earth Engine: {str(e)}")
        raise