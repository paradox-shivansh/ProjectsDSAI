import logging
import os
from datetime import datetime

# Create log file name
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Get project root directory (two levels up from src folder)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Create logs folder path
logs_dir = os.path.join(project_root, "logs")

# Create logs folder if not exists
os.makedirs(logs_dir, exist_ok=True)

# Full log file path
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Create a logger object
logger = logging.getLogger('ETE_DS')
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.INFO)

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)