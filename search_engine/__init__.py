"""
NLP Text Search Engine

A sophisticated text search engine with BPE tokenization, TF-IDF ranking,
and auto-correction capabilities.

Main components:
- TextSearchEngine: Main search engine class
- Tokenizer: BPE tokenization and text processing
- Indexer: Inverted index construction and management
- Ranker: Document ranking using TF-IDF cosine similarity
- AutoCorrect: Query auto-correction using Levenshtein distance
- Utils: Utility functions for text processing and formatting
"""

from .search_engine import TextSearchEngine
from .tokenizer import Tokenizer
from .indexer import Indexer
from .ranker import Ranker
from .autocorrect import AutoCorrect
from .utils import TextProcessor, ResultFormatter

__version__ = "1.0.0"
__author__ = "Rohan Jain"

__all__ = [
    "TextSearchEngine",
    "Tokenizer", 
    "Indexer",
    "Ranker",
    "AutoCorrect",
    "TextProcessor",
    "ResultFormatter"
]
