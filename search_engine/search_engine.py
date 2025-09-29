"""
Main TextSearchEngine class that orchestrates the entire search pipeline.

This module contains the main TextSearchEngine class that coordinates
all components of the search system including document loading, indexing,
and query processing.
"""

import os
import random
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from .tokenizer import Tokenizer
from .indexer import Indexer
from .ranker import Ranker
from .autocorrect import AutoCorrect
from .utils import TextProcessor, ResultFormatter
import config


class TextSearchEngine:
    """
    Main search engine class that provides a unified interface for text search.
    
    This class orchestrates the entire search pipeline from document loading
    through query processing and result ranking.
    """
    
    def __init__(self, corpus_dir: Optional[str] = None, config_dict: Optional[Dict] = None):
        """
        Initialize the TextSearchEngine.
        
        Args:
            corpus_dir: Path to the directory containing text files. If None, uses config default.
            config_dict: Optional configuration dictionary to override defaults.
        """
        # Load configuration
        self.config = self._load_config(config_dict)
        
        # Set corpus directory
        self.corpus_dir = Path(corpus_dir) if corpus_dir else self.config.CORPUS_DIR
        
        # Initialize components
        self.text_processor = TextProcessor(self.config)
        self.tokenizer = Tokenizer(self.config)
        self.indexer = Indexer(self.config)
        self.ranker = Ranker(self.config)
        self.auto_correct = AutoCorrect(self.config)
        self.result_formatter = ResultFormatter(self.config)
        
        # State variables
        self.books: Dict[str, str] = {}
        self.paragraphs: Dict[int, Tuple[str, int, str]] = {}
        self.para_tokens: Dict[int, List[str]] = {}
        self.vocab_set: set = set()
        self.token_freq: Dict[str, int] = {}
        self.inverted_index: Dict[str, List[Tuple]] = {}
        self.idf: Dict[str, float] = {}
        self.doc_norm: Dict[int, float] = {}
        self.word_vocab: set = set()
        self.word_freq: Dict[str, int] = {}
        self.by_len_index: Dict[int, List[str]] = {}
        
        # Status flags
        self._index_built = False
        self._models_trained = False
    
    def _load_config(self, config_dict: Optional[Dict]) -> Any:
        """Load configuration from config module or provided dictionary."""
        if config_dict:
            # Create a simple config object from dictionary
            class Config:
                def __init__(self, config_dict):
                    for key, value in config_dict.items():
                        setattr(self, key, value)
            return Config(config_dict)
        return config
    
    def load_books(self, num_books: Optional[int] = None, seed: int = None) -> Dict[str, str]:
        """
        Load text files from the corpus directory.
        
        Args:
            num_books: Number of books to load. If None, loads all books.
            seed: Random seed for reproducible sampling.
            
        Returns:
            Dictionary mapping filename to text content.
        """
        if seed is None:
            seed = self.config.RANDOM_SEED
            
        files = [f for f in os.listdir(self.corpus_dir) if f.endswith('.txt')]
        random.seed(seed)
        random.shuffle(files)
        
        if num_books is not None:
            files = files[:num_books]
        
        books = {}
        for filename in files:
            path = self.corpus_dir / filename
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    books[filename] = f.read()
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
                
        self.books = books
        return books
    
    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Build the complete search index.
        
        This method performs the following steps:
        1. Load books from corpus directory
        2. Split text into paragraphs
        3. Build word vocabulary for auto-correction
        4. Train BPE tokenizer
        5. Tokenize paragraphs
        6. Build inverted index
        7. Compute TF-IDF weights
        
        Args:
            force_rebuild: If True, rebuild index even if already built.
        """
        if self._index_built and not force_rebuild:
            print("Index already built. Use force_rebuild=True to rebuild.")
            return
        
        print("Building search index...")
        
        # Step 1: Load books
        num_books = self.config.NUM_BOOKS
        self.load_books(num_books=num_books)
        print(f"Loaded {len(self.books)} books")
        
        # Step 2: Split into paragraphs
        self.paragraphs = self.text_processor.break_into_paragraphs(
            self.books, min_par_chars=self.config.MIN_PARAGRAPH_CHARS
        )
        print(f"Created {len(self.paragraphs)} paragraphs")
        
        # Step 3: Build word vocabulary for auto-correction
        self.word_vocab, self.word_freq = self.text_processor.build_word_vocab(
            self.paragraphs, lowercase=self.config.LOWERCASE
        )
        self.by_len_index = self.auto_correct.build_len_index(self.word_vocab)
        print(f"Built word vocabulary: {len(self.word_vocab)} unique words")
        
        # Step 4: Train BPE tokenizer
        if not self._models_trained or force_rebuild:
            model_path = self.tokenizer.train_bpe_model(
                self.books, 
                model_prefix=self.config.BPE_MODEL_PREFIX,
                vocab_size=self.config.BPE_VOCAB_SIZE,
                lowercase=self.config.LOWERCASE
            )
            print(f"Trained BPE model: {model_path}")
        
        # Load the BPE model
        self.tokenizer.load_bpe(f"{self.config.BPE_MODEL_PREFIX}.model")
        
        # Step 5: Tokenize paragraphs
        self.para_tokens, self.vocab_set, self.token_freq = self.tokenizer.tokenize_paragraphs(
            self.paragraphs, lowercase=self.config.LOWERCASE
        )
        print(f"Tokenized paragraphs: {len(self.vocab_set)} unique tokens")
        
        # Step 6: Build inverted index
        self.inverted_index, para_len_tokens, N = self.indexer.build_inverted_index(
            self.para_tokens, with_positions=self.config.WITH_POSITIONS
        )
        print(f"Built inverted index: {len(self.inverted_index)} tokens across {N} paragraphs")
        
        # Step 7: Compute TF-IDF weights
        self.idf, self.doc_norm = self.ranker.compute_idf_and_docnorms(
            self.inverted_index, N
        )
        print("Computed TF-IDF weights")
        
        self._index_built = True
        self._models_trained = True
        print("Index building complete!")
    
    def search(self, query: str, top_k: Optional[int] = None) -> List[Tuple[int, float]]:
        """
        Search for documents matching the given query.
        
        Args:
            query: Search query string.
            top_k: Number of results to return. If None, uses config default.
            
        Returns:
            List of (paragraph_id, score) tuples sorted by relevance.
        """
        if not self._index_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        if top_k is None:
            top_k = self.config.TOP_K_RESULTS
        
        # Process query
        query_words = self.text_processor.extract_words(query, lowercase=self.config.LOWERCASE)
        
        # Auto-correct query words
        if self.config.AUTO_CORRECT_ENABLED:
            corrected_words, changes, oov_words = self.auto_correct.autocorrect_query_words(
                query_words, self.word_vocab, self.word_freq, self.by_len_index
            )
            if changes:
                print(f"Query corrections: {changes}")
        else:
            corrected_words = query_words
        
        # Tokenize corrected query
        corrected_query = " ".join(corrected_words)
        query_tokens = self.tokenizer.tokenize_query(corrected_query, lowercase=self.config.LOWERCASE)
        
        # Filter tokens that exist in vocabulary
        in_vocab_tokens = [t for t in query_tokens if t in self.vocab_set]
        
        if not in_vocab_tokens:
            return []
        
        # Rank documents
        ranked_results = self.ranker.cosine_rank(
            in_vocab_tokens, self.inverted_index, self.idf, self.doc_norm, topk=top_k
        )
        
        return ranked_results
    
    def get_result_snippet(self, para_id: int, query_words: List[str], max_chars: Optional[int] = None) -> str:
        """
        Get a highlighted snippet for a paragraph.
        
        Args:
            para_id: Paragraph ID.
            query_words: Query words to highlight.
            max_chars: Maximum snippet length. If None, uses config default.
            
        Returns:
            Highlighted snippet string.
        """
        if max_chars is None:
            max_chars = self.config.SNIPPET_CHARS
        
        if para_id not in self.paragraphs:
            return ""
        
        _, _, text = self.paragraphs[para_id]
        return self.result_formatter.make_snippet(text, query_words, max_chars)
    
    def interactive_search(self) -> None:
        """
        Start an interactive search session.
        
        This method provides a command-line interface for searching.
        Type 'exit' or 'quit' to end the session.
        """
        if not self._index_built:
            print("Building index first...")
            self.build_index()
        
        print("\n=== Interactive Search ===")
        print("Type 'exit' or 'quit' to quit.")
        
        while True:
            try:
                query = input("Enter search query: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            
            if not query:
                continue
            if query.lower() in ('exit', 'quit'):
                print("Goodbye!")
                break
            
            # Search
            results = self.search(query)
            
            if results:
                # Extract query words for highlighting
                query_words = self.text_processor.extract_words(query, lowercase=self.config.LOWERCASE)
                
                # Format and display results
                self.result_formatter.print_results_table(
                    results, self.paragraphs, query_words, 
                    max_chars=self.config.SNIPPET_CHARS
                )
            else:
                print("No matching documents found.")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the built index.
        
        Returns:
            Dictionary containing various statistics.
        """
        if not self._index_built:
            return {"error": "Index not built"}
        
        stats = {
            "num_books": len(self.books),
            "num_paragraphs": len(self.paragraphs),
            "num_tokens": len(self.vocab_set),
            "num_words": len(self.word_vocab),
            "index_size": len(self.inverted_index),
            "avg_paragraph_length": sum(len(text) for _, _, text in self.paragraphs.values()) / len(self.paragraphs) if self.paragraphs else 0
        }
        
        return stats
