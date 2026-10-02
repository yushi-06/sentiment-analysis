#!/usr/bin/env python3
"""
Configuration settings for sentiment analysis project
"""

import os
from pathlib import Path

# Project Paths
PROJECT_ROOT = Path(__file__).parent
MODEL_DIR = PROJECT_ROOT / "sentiment_model"
KNOWLEDGE_FILE = PROJECT_ROOT / "sentiment_knowledge.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# Model Settings
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_LABELS = 3
LABEL_MAPPING = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Training Settings
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500

# Adaptive Learning Settings
MIN_OCCURRENCES = 3
SENTIMENT_THRESHOLD = 0.7
CONFIDENCE_THRESHOLD = 0.1

# File Format Settings
SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls']
DEFAULT_TEXT_COLUMN = 'text'
DEFAULT_SENTIMENT_COLUMN = 'sentiment'

# Logging Settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Performance Settings
MAX_WORKERS = 4
CACHE_SIZE = 1000

# Validation Settings
MIN_TEXT_LENGTH = 3
MAX_TEXT_LENGTH = 512

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = [MODEL_DIR, RESULTS_DIR, DATA_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_device():
    """Get the best available device for computation"""
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

# Environment-specific overrides
if os.getenv('SENTIMENT_ENV') == 'production':
    LOG_LEVEL = 'WARNING'
    BATCH_SIZE = 32
elif os.getenv('SENTIMENT_ENV') == 'development':
    LOG_LEVEL = 'DEBUG'
    BATCH_SIZE = 8
