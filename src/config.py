import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / os.getenv("RAW_DATA_DIR", "data/raw")
SUBSET_DATA_DIR = PROJECT_ROOT / os.getenv("SUBSET_DATA_DIR", "data/subset")
SUBSET_VALIDATION_DIR = PROJECT_ROOT / os.getenv("SUBSET_VALIDATION_DIR", "data/validation")
MODEL_DIR = PROJECT_ROOT / os.getenv("MODEL_DIR", "models/final_model")
CHECKPOINT_DIR = PROJECT_ROOT / os.getenv("CHECKPOINT_DIR", "models/checkpoints")