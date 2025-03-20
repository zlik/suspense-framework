import os
from dotenv import load_dotenv

# Get the absolute path of the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Load API keys from .env file
ENV_PATH = os.path.join(PROJECT_DIR, ".env")
load_dotenv(ENV_PATH)

def getenv(key: str, default: str = None):
    return os.getenv(key, default)
