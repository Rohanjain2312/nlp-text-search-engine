"""
Inverted index construction and management.

This module handles building and managing inverted indexes for efficient
text search and retrieval.
"""

from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter


class Indexer:
    """Handles inverted index construction and management."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
    
    def build_inverted_index(self, para_tokens: Dict[int, List[str]], with_positions: bool = None) -> Tuple[Dict[str, List[Tuple]], Dict[int, int], int]:
        """
        Build an inverted index from paragraph tokens.
        
        Args:
            para_tokens: Dictionary mapping paragraph ID to list of tokens.
            with_positions: Whether to store token positions within paragraphs.
            
        Returns:
            Tuple of (inverted_index, para_len_tokens, N).
            - inverted_index: token -> List[(para_id, freq)] or List[(para_id, freq, positions)]
            - para_len_tokens: para_id -> number of tokens
            - N: total number of paragraphs
        """
        if with_positions is None:
            with_positions = self.config.WITH_POSITIONS
        
        postings = defaultdict(list)      # token -> list of postings
        para_len_tokens = {}              # para_id -> length
        N = len(para_tokens)              # total paragraphs
        
        for pid, tokens in para_tokens.items():
            # Record paragraph token length for later ranking
            para_len_tokens[pid] = len(tokens)
            
            if not tokens:
                continue
            
            if with_positions:
                # Build both counts and positions
                pos_map = defaultdict(list)  # token -> [positions]
                for idx, tok in enumerate(tokens):
                    pos_map[tok].append(idx)
                # For each token in this paragraph, append a single posting with (pid, freq, positions)
                for tok, positions in pos_map.items():
                    freq = len(positions)
                    postings[tok].append((pid, freq, positions))
            else:
                # Only counts (faster and smaller)
                counts = Counter(tokens)  # token -> freq in this paragraph
                for tok, freq in counts.items():
                    postings[tok].append((pid, freq))
        
        # Sort postings by para_id for determinism, and convert to regular dict
        inverted_index = {}
        for tok, plist in postings.items():
            inverted_index[tok] = sorted(plist, key=lambda x: x[0])  # sort by para_id
        
        return inverted_index, para_len_tokens, N
    
    def get_posting_list(self, token: str, inverted_index: Dict[str, List[Tuple]]) -> List[Tuple]:
        """
        Get the posting list for a token.
        
        Args:
            token: Token to look up.
            inverted_index: The inverted index.
            
        Returns:
            List of postings for the token.
        """
        return inverted_index.get(token, [])
    
    def get_document_frequency(self, token: str, inverted_index: Dict[str, List[Tuple]]) -> int:
        """
        Get the document frequency (number of documents containing the token).
        
        Args:
            token: Token to look up.
            inverted_index: The inverted index.
            
        Returns:
            Document frequency of the token.
        """
        postings = self.get_posting_list(token, inverted_index)
        return len(postings)
    
    def get_collection_frequency(self, token: str, inverted_index: Dict[str, List[Tuple]]) -> int:
        """
        Get the collection frequency (total occurrences of the token).
        
        Args:
            token: Token to look up.
            inverted_index: The inverted index.
            
        Returns:
            Collection frequency of the token.
        """
        postings = self.get_posting_list(token, inverted_index)
        return sum(freq for (_, freq, *_) in postings)
    
    def get_documents_containing(self, tokens: List[str], inverted_index: Dict[str, List[Tuple]]) -> Set[int]:
        """
        Get the set of documents containing any of the given tokens.
        
        Args:
            tokens: List of tokens to search for.
            inverted_index: The inverted index.
            
        Returns:
            Set of document IDs containing at least one of the tokens.
        """
        doc_ids = set()
        for token in tokens:
            postings = self.get_posting_list(token, inverted_index)
            doc_ids.update(pid for (pid, *_) in postings)
        return doc_ids
    
    def get_common_documents(self, tokens: List[str], inverted_index: Dict[str, List[Tuple]]) -> Set[int]:
        """
        Get the set of documents containing all of the given tokens.
        
        Args:
            tokens: List of tokens to search for.
            inverted_index: The inverted index.
            
        Returns:
            Set of document IDs containing all tokens.
        """
        if not tokens:
            return set()
        
        # Start with documents containing the first token
        first_token = tokens[0]
        postings = self.get_posting_list(first_token, inverted_index)
        common_docs = set(pid for (pid, *_) in postings)
        
        # Intersect with documents containing each subsequent token
        for token in tokens[1:]:
            postings = self.get_posting_list(token, inverted_index)
            token_docs = set(pid for (pid, *_) in postings)
            common_docs &= token_docs
        
        return common_docs
    
    def summarize_index(self, inverted_index: Dict[str, List[Tuple]], N: int) -> None:
        """
        Print a summary of the inverted index.
        
        Args:
            inverted_index: The inverted index.
            N: Total number of documents.
        """
        num_tokens = len(inverted_index)
        total_postings = sum(len(postings) for postings in inverted_index.values())
        
        print("\n=== Inverted Index Summary ===")
        print(f"Unique tokens: {num_tokens}")
        print(f"Total postings: {total_postings}")
        print(f"Average postings per token: {total_postings / num_tokens:.2f}")
        print(f"Documents indexed: {N}")
        
        # Show some statistics about posting list lengths
        posting_lengths = [len(postings) for postings in inverted_index.values()]
        if posting_lengths:
            print(f"Min posting list length: {min(posting_lengths)}")
            print(f"Max posting list length: {max(posting_lengths)}")
            print(f"Median posting list length: {sorted(posting_lengths)[len(posting_lengths)//2]}")
