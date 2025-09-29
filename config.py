"""
Configuration settings for the NLP Text Search Engine.

This module contains all configurable parameters for the search engine.
Modify these values to customize the behavior of the system.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
CORPUS_DIR = PROJECT_ROOT / "Input_Books"
MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Ensure directories exist
CORPUS_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Text processing settings
LOWERCASE = True  # Use lowercase for training and tokenization
MIN_PARAGRAPH_CHARS = 30  # Minimum characters for a valid paragraph

# BPE (Byte Pair Encoding) settings
BPE_VOCAB_SIZE = 10000  # Target vocabulary size for BPE
BPE_MODEL_PREFIX = "bpe_model_hw1"  # Prefix for BPE model files

# Search settings
MAX_EDIT_DISTANCE = 2  # Maximum edit distance for auto-correction
TOP_K_RESULTS = 10  # Number of results to return
SNIPPET_CHARS = 200  # Maximum characters in result snippets
RANKING_METHOD = "cosine"  # Ranking method: "cosine" or "bm25"

# Corpus settings
NUM_BOOKS = 100  # Number of books to process (None for all)
RANDOM_SEED = 42  # Random seed for reproducible results

# Index settings
WITH_POSITIONS = False  # Whether to store token positions in index
CACHE_INDEX = True  # Whether to cache the built index

# Auto-correction settings
AUTO_CORRECT_ENABLED = True  # Enable/disable auto-correction
MIN_WORD_LENGTH = 2  # Minimum word length for auto-correction

# Output settings
VERBOSE = True  # Enable verbose output during processing
SAVE_INTERMEDIATE = True  # Save intermediate results (vocab, models)

# File extensions
TEXT_EXTENSIONS = ['.txt']  # Supported text file extensions
MODEL_EXTENSION = '.model'  # BPE model file extension
VOCAB_EXTENSION = '.pkl'  # Vocabulary file extension

# Performance settings
BATCH_SIZE = 1000  # Batch size for processing documents
MAX_MEMORY_GB = 4  # Maximum memory usage in GB
PARALLEL_WORKERS = 4  # Number of parallel workers for processing

# Debug settings
DEBUG = False  # Enable debug mode
PROFILE = False  # Enable profiling
LOG_LEVEL = "INFO"  # Logging level: DEBUG, INFO, WARNING, ERROR

# Default search parameters
DEFAULT_QUERY_TIMEOUT = 30  # Query timeout in seconds
DEFAULT_MAX_RESULTS = 100  # Maximum results to consider before ranking

# Highlighting settings
HIGHLIGHT_START = "[["  # Start marker for highlighting
HIGHLIGHT_END = "]]"  # End marker for highlighting
HIGHLIGHT_CASE_SENSITIVE = False  # Case sensitivity for highlighting

# Result formatting
RESULT_FORMAT = "table"  # Result format: "table", "list", "json"
SHOW_SCORES = True  # Show relevance scores in results
SHOW_METADATA = True  # Show document metadata in results

# Export settings
EXPORT_FORMATS = ["txt", "json", "csv"]  # Supported export formats
DEFAULT_EXPORT_FORMAT = "txt"  # Default export format
