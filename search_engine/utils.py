"""
Utility functions for text processing and result formatting.

This module contains helper functions for text preprocessing, paragraph splitting,
highlighting, and result formatting.
"""

import re
from typing import Dict, List, Tuple, Set, Counter
from collections import Counter, defaultdict


class TextProcessor:
    """Handles text preprocessing and paragraph splitting."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self.word_regex = re.compile(r"[a-z0-9']+")
    
    def split_into_paragraphs(self, text: str, min_par_chars: int = None) -> List[str]:
        """
        Split raw book text into paragraphs using 2+ newline boundaries.
        
        Args:
            text: Raw text to split.
            min_par_chars: Minimum characters for a valid paragraph.
            
        Returns:
            List of paragraph strings.
        """
        if min_par_chars is None:
            min_par_chars = self.config.MIN_PARAGRAPH_CHARS
        
        # Split on two or more consecutive newlines
        raw_paras = re.split(r'\n{2,}', text)
        paras = []
        for seg in raw_paras:
            p = seg.strip()
            if len(p) >= min_par_chars:
                paras.append(p)
        return paras
    
    def break_into_paragraphs(self, books: Dict[str, str], min_par_chars: int = None) -> Dict[int, Tuple[str, int, str]]:
        """
        Assign global paragraph IDs and keep (book_id, para_idx_in_book, text).
        
        Args:
            books: Dictionary mapping book_id to text content.
            min_par_chars: Minimum characters for a valid paragraph.
            
        Returns:
            Dictionary mapping para_id to (book_id, para_idx_in_book, text).
        """
        if min_par_chars is None:
            min_par_chars = self.config.MIN_PARAGRAPH_CHARS
        
        paragraphs = {}
        para_id = 0
        for book_id, text in books.items():
            book_paras = self.split_into_paragraphs(text, min_par_chars=min_par_chars)
            for idx_in_book, ptext in enumerate(book_paras):
                paragraphs[para_id] = (book_id, idx_in_book, ptext)
                para_id += 1
        return paragraphs
    
    def extract_words(self, text: str, lowercase: bool = None) -> List[str]:
        """
        Extract words from text using regex.
        
        Args:
            text: Input text.
            lowercase: Whether to convert to lowercase.
            
        Returns:
            List of extracted words.
        """
        if lowercase is None:
            lowercase = self.config.LOWERCASE
        
        txt = text.lower() if lowercase else text
        return self.word_regex.findall(txt)
    
    def build_word_vocab(self, paragraphs: Dict[int, Tuple[str, int, str]], lowercase: bool = None) -> Tuple[Set[str], Counter]:
        """
        Build a word-level vocabulary and frequency dictionary from paragraph text.
        
        Args:
            paragraphs: Dictionary mapping para_id to (book_id, para_idx, text).
            lowercase: Whether to use lowercase.
            
        Returns:
            Tuple of (word_vocab, word_freq).
        """
        if lowercase is None:
            lowercase = self.config.LOWERCASE
        
        freq = Counter()
        for (_pid, (_book, _idx, text)) in paragraphs.items():
            txt = text.lower() if lowercase else text
            words = self.word_regex.findall(txt)
            if words:
                freq.update(words)
        
        word_vocab = set(freq.keys())
        return word_vocab, freq


class ResultFormatter:
    """Handles result formatting and display."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
    
    def _format_tokens(self, tokens: List[str], maxn: int = 12) -> str:
        """
        Return tokens as a compact string; truncate long lists with an ellipsis.
        
        Args:
            tokens: List of tokens to format.
            maxn: Maximum number of tokens to show.
            
        Returns:
            Formatted token string.
        """
        if len(tokens) <= maxn:
            return "[" + ", ".join(tokens) + "]"
        head = ", ".join(tokens[:maxn//2])
        tail = ", ".join(tokens[-maxn//2:])
        return "[" + head + ", …, " + tail + "]"
    
    def highlight_words(self, text: str, words: List[str]) -> str:
        """
        Naive console-safe highlighter: wraps whole-word matches with [[ ]].
        
        Args:
            text: Text to highlight.
            words: List of words to highlight.
            
        Returns:
            Highlighted text.
        """
        if not words:
            return text
        
        # Deduplicate and sort longer-first to avoid partial overshadowing
        uniq = sorted({w for w in words if w}, key=len, reverse=True)
        
        def repl(match):
            return f"{self.config.HIGHLIGHT_START}{match.group(0)}{self.config.HIGHLIGHT_END}"
        
        # Build a combined regex of word boundaries for each word
        patterns = [r"\b" + re.escape(w) + r"\b" for w in uniq]
        if not patterns:
            return text
        
        flags = re.IGNORECASE if not self.config.HIGHLIGHT_CASE_SENSITIVE else 0
        regex = re.compile("|".join(patterns), flags=flags)
        return regex.sub(repl, text)
    
    def make_snippet(self, text: str, query_words: List[str], max_chars: int = None) -> str:
        """
        Produce a snippet with highlighted query words and trimmed to max_chars.
        
        Args:
            text: Text to create snippet from.
            query_words: Words to highlight in snippet.
            max_chars: Maximum characters in snippet.
            
        Returns:
            Highlighted snippet string.
        """
        if max_chars is None:
            max_chars = self.config.SNIPPET_CHARS
        
        # Highlight first, then trim so brackets are visible
        highlighted = self.highlight_words(text, query_words)
        if len(highlighted) <= max_chars:
            return highlighted.replace("\n", " ")
        
        # Try to find the first highlight marker to center the snippet
        marker = highlighted.lower().find(self.config.HIGHLIGHT_START.lower())
        if marker == -1:
            # No highlights found; simple head trim
            return highlighted[:max_chars].replace("\n", " ")
        
        # Center window around the first marker
        start = max(0, marker - max_chars // 3)
        end = min(len(highlighted), start + max_chars)
        snippet = highlighted[start:end]
        
        if start > 0:
            snippet = "…" + snippet
        if end < len(highlighted):
            snippet = snippet + "…"
        
        return snippet.replace("\n", " ")
    
    def print_results_table(self, ranked: List[Tuple[int, float]], paragraphs: Dict[int, Tuple[str, int, str]], 
                           query_tokens: List[str], max_chars: int = None, query_words_for_highlight: List[str] = None) -> None:
        """
        Render top results as a clean ASCII table.
        
        Args:
            ranked: List of (para_id, score) tuples.
            paragraphs: Dictionary mapping para_id to paragraph info.
            query_tokens: Query tokens used for search.
            max_chars: Maximum characters in snippet.
            query_words_for_highlight: Words to highlight in snippets.
        """
        if max_chars is None:
            max_chars = self.config.SNIPPET_CHARS
        
        if not ranked:
            print("No matching paragraphs found.")
            return
        
        # Prepare rows
        rows = []
        for rank, (pid, score) in enumerate(ranked, start=1):
            book_id, idx_in_book, text = paragraphs.get(pid, ("?", -1, ""))
            snippet = self.make_snippet(text, query_words_for_highlight or [], max_chars=max_chars)
            rows.append([str(rank), str(pid), f"{score:.4f}", book_id, str(idx_in_book), snippet])
        
        # Column headers
        headers = ["#", "PID", "Score", "Book", "ParaIdx", "Snippet"]
        
        # Compute column widths with caps for Book and Snippet
        max_widths = [3, 8, 8, 40, 7, max_chars]
        col_widths = []
        for j, h in enumerate(headers):
            width = len(h)
            for row in rows:
                width = max(width, len(row[j]))
            width = min(width, max_widths[j])
            col_widths.append(width)
        
        # Helper to clip and pad
        def clip_pad(s, w):
            if len(s) > w:
                return s[: max(0, w - 1)] + "…" if w >= 2 else s[:w]
            return s.ljust(w)
        
        # Print header
        line = " | ".join(clip_pad(h, col_widths[i]) for i, h in enumerate(headers))
        sep = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
        print("\n=== Top Results ===")
        print(line)
        print(sep)
        
        # Print rows
        for row in rows:
            print(" | ".join(clip_pad(row[i], col_widths[i]) for i in range(len(headers))))
        
        # Footer: query tokens used (compact)
        q_preview = self._format_tokens(query_tokens, maxn=12)
        print(f"\n(query tokens used: {q_preview})\n")
    
    def print_results_simple(self, ranked: List[Tuple[int, float]], paragraphs: Dict[int, Tuple[str, int, str]], 
                           query_tokens: List[str], max_chars: int = None, query_words_for_highlight: List[str] = None) -> None:
        """
        Print a simple view of ranked results.
        
        Args:
            ranked: List of (para_id, score) tuples.
            paragraphs: Dictionary mapping para_id to paragraph info.
            query_tokens: Query tokens used for search.
            max_chars: Maximum characters in snippet.
            query_words_for_highlight: Words to highlight in snippets.
        """
        if max_chars is None:
            max_chars = self.config.SNIPPET_CHARS
        
        if not ranked:
            print("No matching paragraphs found.")
            return
        
        print("\n=== Top Results ===")
        
        # Use word-level terms for highlighting when available; else derive coarse words from tokens
        if query_words_for_highlight is None:
            # fallback: strip leading ▁ from BPE tokens to approximate words
            query_words_for_highlight = [t.lstrip('▁') for t in query_tokens if t and t != '▁']
        
        for rank, (pid, score) in enumerate(ranked, start=1):
            book_id, idx_in_book, text = paragraphs.get(pid, ("?", -1, ""))
            snippet = self.make_snippet(text, query_words_for_highlight, max_chars=max_chars)
            print(f"#{rank}  pid={pid}  score={score:.4f}  book={book_id}  para_idx={idx_in_book}")
            print(f"     {snippet}")
        
        q_preview = self._format_tokens(query_tokens, maxn=12)
        print(f"\n(query tokens used: {q_preview})\n")
