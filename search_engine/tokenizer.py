"""
BPE (Byte Pair Encoding) tokenization module.

This module handles BPE model training, loading, and text tokenization
using SentencePiece.
"""

import os
import tempfile
import pickle
from typing import Dict, List, Tuple, Set, Counter
import sentencepiece as spm


class Tokenizer:
    """Handles BPE tokenization and model management."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.sp = None
    
    def train_bpe_model(self, books: Dict[str, str], model_prefix: str = None, 
                       vocab_size: int = None, lowercase: bool = None) -> str:
        """
        Train a SentencePiece BPE model on the provided books.
        
        Args:
            books: Dictionary mapping book_id to text content.
            model_prefix: Prefix for model files.
            vocab_size: Target vocabulary size.
            lowercase: Whether to use lowercase.
            
        Returns:
            Path to the trained model file.
        """
        if model_prefix is None:
            model_prefix = self.config.BPE_MODEL_PREFIX
        if vocab_size is None:
            vocab_size = self.config.BPE_VOCAB_SIZE
        if lowercase is None:
            lowercase = self.config.LOWERCASE
        
        # Concatenate all book texts for training
        if lowercase:
            corpus_text = "\n".join((text.lower() for text in books.values()))
        else:
            corpus_text = "\n".join(books.values())
        
        # Write to a temporary file because SentencePieceTrainer expects a file path
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding="utf-8") as tmp:
            tmp.write(corpus_text)
            tmp_path = tmp.name
        
        # Train BPE model
        spm.SentencePieceTrainer.train(
            input=tmp_path,
            model_prefix=model_prefix,
            vocab_size=vocab_size,
            model_type="bpe",
            character_coverage=1.0,  # assume English/ASCII-heavy Gutenberg
            input_sentence_size=0,    # use all provided data
            shuffle_input_sentence=False
        )
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Store model path
        model_path = f"{model_prefix}.model"
        return model_path
    
    def load_bpe(self, model_path: str) -> spm.SentencePieceProcessor:
        """
        Load a trained SentencePiece model from disk.
        
        Args:
            model_path: Path to the model file.
            
        Returns:
            Loaded SentencePiece processor.
        """
        sp = spm.SentencePieceProcessor()
        sp.load(model_path)
        self.sp = sp
        return sp
    
    def tokenize_query(self, text: str, lowercase: bool = None) -> List[str]:
        """
        Tokenize a query string.
        
        Args:
            text: Text to tokenize.
            lowercase: Whether to use lowercase.
            
        Returns:
            List of tokens.
        """
        if lowercase is None:
            lowercase = self.config.LOWERCASE
        
        if self.sp is None:
            raise RuntimeError("BPE model not loaded. Call load_bpe() first.")
        
        txt = text.lower() if lowercase else text
        return self.sp.encode(txt, out_type=str)
    
    def tokenize_paragraphs(self, paragraphs: Dict[int, Tuple[str, int, str]], 
                           sp: spm.SentencePieceProcessor = None, lowercase: bool = None) -> Tuple[Dict[int, List[str]], Set[str], Counter]:
        """
        Tokenize each paragraph using the provided SentencePiece processor.
        
        Args:
            paragraphs: Dictionary mapping para_id to (book_id, para_idx, text).
            sp: SentencePiece processor. If None, uses self.sp.
            lowercase: Whether to use lowercase.
            
        Returns:
            Tuple of (para_tokens, vocab_set, token_freq).
        """
        if sp is None:
            sp = self.sp
        if sp is None:
            raise RuntimeError("BPE model not loaded. Call load_bpe() first.")
        if lowercase is None:
            lowercase = self.config.LOWERCASE
        
        para_tokens = {}
        vocab_set = set()
        token_freq = Counter()
        
        for pid, (_book, _idx, text) in paragraphs.items():
            txt = text.lower() if lowercase else text
            tokens = sp.encode(txt, out_type=str)
            para_tokens[pid] = tokens
            vocab_set.update(tokens)
            token_freq.update(tokens)
        
        return para_tokens, vocab_set, token_freq
    
    def save_vocab(self, vocab_set: Set[str], token_freq: Counter, prefix: str = "vocab") -> None:
        """
        Save vocabulary and frequency data to disk.
        
        Args:
            vocab_set: Set of vocabulary tokens.
            token_freq: Token frequency counter.
            prefix: Prefix for saved files.
        """
        with open(f"{prefix}_set.pkl", "wb") as f:
            pickle.dump(vocab_set, f)
        with open(f"{prefix}_freq.pkl", "wb") as f:
            pickle.dump(token_freq, f)
    
    def load_vocab(self, prefix: str = "vocab") -> Tuple[Set[str], Counter]:
        """
        Load vocabulary and frequency data from disk.
        
        Args:
            prefix: Prefix for saved files.
            
        Returns:
            Tuple of (vocab_set, token_freq) or (None, None) if not found.
        """
        try:
            with open(f"{prefix}_set.pkl", "rb") as f:
                vocab = pickle.load(f)
            with open(f"{prefix}_freq.pkl", "rb") as f:
                freq = pickle.load(f)
            return vocab, freq
        except FileNotFoundError:
            return None, None
    
    def summarize_vocabulary(self, token_freq: Counter, topn: int = 10) -> None:
        """
        Print a summary of vocabulary statistics.
        
        Args:
            token_freq: Token frequency counter.
            topn: Number of top tokens to show.
        """
        total_occurrences = sum(token_freq.values())
        unique_tokens = len(token_freq)
        print("\n=== Vocabulary Summary ===")
        print(f"Unique tokens: {unique_tokens}  |  Total token occurrences: {total_occurrences}")
        if topn > 0 and unique_tokens > 0:
            most_common = token_freq.most_common(topn)
            preview = ", ".join(f"{tok}:{cnt}" for tok, cnt in most_common)
            print(f"Top {topn} tokens: {preview}")
