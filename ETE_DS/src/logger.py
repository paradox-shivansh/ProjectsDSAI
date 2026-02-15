import logging
import os
from datetime import datetime

# Create log file name
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

# Create logs folder path
logs_dir = os.path.join(os.getcwd(), "logs")

# Create logs folder if not exists
os.makedirs(logs_dir, exist_ok=True)

# Full log file path
LOG_FILE_PATH = os.path.join(logs_dir, LOG_FILE)

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Test logging
# logging.info("Logging setup successful")

# if __name__=='__main__':
#     logging.info("Logging has started")