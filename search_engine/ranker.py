"""
Document ranking and scoring module.

This module handles TF-IDF computation and cosine similarity ranking
for document retrieval.
"""

import math
from typing import Dict, List, Tuple
from collections import Counter, defaultdict


class Ranker:
    """Handles document ranking using TF-IDF and cosine similarity."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
    
    def compute_idf_and_docnorms(self, inverted_index: Dict[str, List[Tuple]], N: int) -> Tuple[Dict[str, float], Dict[int, float]]:
        """
        Compute IDF per token and TF-IDF L2 norms per document.
        
        IDF formula: idf = log((N + 1) / (df + 1)) + 1  (smooth, positive)
        TF weighting: tfw = 1 + log(tf)
        
        Args:
            inverted_index: The inverted index.
            N: Total number of documents.
            
        Returns:
            Tuple of (idf, doc_norm).
            - idf: Dict[token, float] - IDF scores
            - doc_norm: Dict[para_id, float] - L2 norms of TF-IDF vectors
        """
        idf = {}
        doc_norm2 = defaultdict(float)
        
        # First compute IDF per token
        for tok, postings in inverted_index.items():
            df = len(postings)
            idf[tok] = math.log((N + 1) / (df + 1)) + 1.0
        
        # Then accumulate squared weights per doc across all tokens
        for tok, postings in inverted_index.items():
            tok_idf = idf[tok]
            for (pid, tf, *rest) in postings:
                tfw = 1.0 + math.log(tf)
                w = tfw * tok_idf
                doc_norm2[pid] += w * w
        
        # Take square roots
        doc_norm = {pid: math.sqrt(v) for pid, v in doc_norm2.items()}
        return idf, doc_norm
    
    def cosine_rank(self, query_tokens: List[str], inverted_index: Dict[str, List[Tuple]], 
                   idf: Dict[str, float], doc_norm: Dict[int, float], topk: int = None) -> List[Tuple[int, float]]:
        """
        Rank documents using cosine similarity over TF-IDF vectors.
        
        Args:
            query_tokens: List of query tokens.
            inverted_index: The inverted index.
            idf: IDF scores for tokens.
            doc_norm: L2 norms of document TF-IDF vectors.
            topk: Number of top results to return.
            
        Returns:
            List of (para_id, score) tuples sorted by score descending.
        """
        if topk is None:
            topk = self.config.TOP_K_RESULTS
        
        if not query_tokens:
            return []
        
        # Build query vector weights
        q_counts = Counter(query_tokens)
        q_weights = {}
        for tok, tf in q_counts.items():
            if tok not in idf:
                continue
            tfw = 1.0 + math.log(tf)
            q_weights[tok] = tfw * idf[tok]
        
        if not q_weights:
            return []
        
        # Precompute query norm
        q_norm = math.sqrt(sum(w * w for w in q_weights.values()))
        if q_norm == 0.0:
            return []
        
        # Accumulate dot-products over candidate docs by iterating postings per token
        scores = defaultdict(float)  # pid -> dot-product
        for tok, wq in q_weights.items():
            postings = inverted_index.get(tok)
            if not postings:
                continue
            for (pid, tf, *rest) in postings:
                tfw = 1.0 + math.log(tf)
                wd = tfw * idf[tok]
                scores[pid] += wq * wd
        
        # Convert to cosine by dividing by norms
        ranked = []
        for pid, dot in scores.items():
            dn = doc_norm.get(pid, 0.0)
            if dn > 0.0:
                ranked.append((pid, dot / (q_norm * dn)))
        
        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked[:topk]
    
    def compute_tf_idf_score(self, token: str, doc_id: int, inverted_index: Dict[str, List[Tuple]], 
                            idf: Dict[str, float], N: int) -> float:
        """
        Compute TF-IDF score for a token in a specific document.
        
        Args:
            token: The token.
            doc_id: Document ID.
            inverted_index: The inverted index.
            idf: IDF scores for tokens.
            N: Total number of documents.
            
        Returns:
            TF-IDF score.
        """
        if token not in inverted_index:
            return 0.0
        
        # Find the document in the posting list
        postings = inverted_index[token]
        for (pid, tf, *rest) in postings:
            if pid == doc_id:
                tfw = 1.0 + math.log(tf)
                return tfw * idf[token]
        
        return 0.0
    
    def compute_document_vector_norm(self, doc_id: int, inverted_index: Dict[str, List[Tuple]], 
                                   idf: Dict[str, float]) -> float:
        """
        Compute the L2 norm of a document's TF-IDF vector.
        
        Args:
            doc_id: Document ID.
            inverted_index: The inverted index.
            idf: IDF scores for tokens.
            
        Returns:
            L2 norm of the document's TF-IDF vector.
        """
        norm_squared = 0.0
        
        for token, postings in inverted_index.items():
            for (pid, tf, *rest) in postings:
                if pid == doc_id:
                    tfw = 1.0 + math.log(tf)
                    w = tfw * idf[token]
                    norm_squared += w * w
                    break
        
        return math.sqrt(norm_squared)
    
    def get_top_tokens_for_document(self, doc_id: int, inverted_index: Dict[str, List[Tuple]], 
                                  idf: Dict[str, float], topk: int = 10) -> List[Tuple[str, float]]:
        """
        Get the top-k highest TF-IDF scoring tokens for a document.
        
        Args:
            doc_id: Document ID.
            inverted_index: The inverted index.
            idf: IDF scores for tokens.
            topk: Number of top tokens to return.
            
        Returns:
            List of (token, tf_idf_score) tuples sorted by score descending.
        """
        token_scores = []
        
        for token, postings in inverted_index.items():
            for (pid, tf, *rest) in postings:
                if pid == doc_id:
                    tfw = 1.0 + math.log(tf)
                    score = tfw * idf[token]
                    token_scores.append((token, score))
                    break
        
        token_scores.sort(key=lambda x: x[1], reverse=True)
        return token_scores[:topk]
    
    def summarize_ranking_stats(self, idf: Dict[str, float], doc_norm: Dict[int, float]) -> None:
        """
        Print summary statistics about the ranking system.
        
        Args:
            idf: IDF scores for tokens.
            doc_norm: L2 norms of document TF-IDF vectors.
        """
        if not idf or not doc_norm:
            print("No ranking statistics available.")
            return
        
        print("\n=== Ranking Statistics ===")
        print(f"Number of tokens with IDF scores: {len(idf)}")
        print(f"Number of documents with norms: {len(doc_norm)}")
        
        # IDF statistics
        idf_values = list(idf.values())
        print(f"IDF range: {min(idf_values):.3f} - {max(idf_values):.3f}")
        print(f"Average IDF: {sum(idf_values) / len(idf_values):.3f}")
        
        # Document norm statistics
        norm_values = list(doc_norm.values())
        print(f"Document norm range: {min(norm_values):.3f} - {max(norm_values):.3f}")
        print(f"Average document norm: {sum(norm_values) / len(norm_values):.3f}")
