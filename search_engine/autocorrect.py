"""
Auto-correction module for query processing.

This module handles query auto-correction using Levenshtein distance
and word frequency information.
"""

from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
from rapidfuzz.distance import Levenshtein


class AutoCorrect:
    """Handles query auto-correction using edit distance and frequency."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
    
    def build_len_index(self, word_vocab: Set[str]) -> Dict[int, List[str]]:
        """
        Build a length-based index for efficient candidate lookup.
        
        Args:
            word_vocab: Set of vocabulary words.
            
        Returns:
            Dictionary mapping word length to list of words of that length.
        """
        index = defaultdict(list)
        for w in word_vocab:
            index[len(w)].append(w)
        return index
    
    def _candidate_words(self, word: str, word_vocab: Set[str], by_len_index: Dict[int, List[str]], 
                        max_len_diff: int = None) -> List[str]:
        """
        Generate candidate words from vocabulary within a length band.
        
        Args:
            word: Input word.
            word_vocab: Set of vocabulary words.
            by_len_index: Length-based index of vocabulary.
            max_len_diff: Maximum length difference to consider.
            
        Returns:
            List of candidate words.
        """
        if max_len_diff is None:
            max_len_diff = self.config.MAX_EDIT_DISTANCE
        
        L = len(word)
        candidates = []
        
        for dL in range(-max_len_diff, max_len_diff + 1):
            bucket = by_len_index.get(L + dL)
            if bucket:
                candidates.extend(bucket)
        
        return candidates
    
    def suggest_correction(self, word: str, word_vocab: Set[str], word_freq: Dict[str, int], 
                          by_len_index: Dict[int, List[str]], max_dist: int = None) -> Tuple[str, int]:
        """
        Suggest a correction for a word using edit distance and frequency.
        
        Args:
            word: Word to correct.
            word_vocab: Set of vocabulary words.
            word_freq: Word frequency dictionary.
            by_len_index: Length-based index of vocabulary.
            max_dist: Maximum edit distance to consider.
            
        Returns:
            Tuple of (best_word, best_distance) or (None, None) if no good match.
        """
        if max_dist is None:
            max_dist = self.config.MAX_EDIT_DISTANCE
        
        best_word, best_dist, best_freq = None, None, -1
        
        candidates = self._candidate_words(word, word_vocab, by_len_index, max_len_diff=max_dist)
        
        for cand in candidates:
            dist = Levenshtein.distance(word, cand)
            if dist <= max_dist:
                freq = word_freq.get(cand, 0)
                # Tie-break: smaller distance first, then higher frequency
                if (best_dist is None) or (dist < best_dist) or (dist == best_dist and freq > best_freq):
                    best_word, best_dist, best_freq = cand, dist, freq
                if best_dist == 0:  # Perfect match found
                    break
        
        return best_word, best_dist
    
    def autocorrect_query_words(self, words: List[str], word_vocab: Set[str], word_freq: Dict[str, int], 
                               by_len_index: Dict[int, List[str]], max_dist: int = None) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
        """
        Auto-correct a list of query words.
        
        Args:
            words: List of words to correct.
            word_vocab: Set of vocabulary words.
            word_freq: Word frequency dictionary.
            by_len_index: Length-based index of vocabulary.
            max_dist: Maximum edit distance to consider.
            
        Returns:
            Tuple of (corrected_words, changes, oov_no_suggest).
            - corrected_words: List of corrected words
            - changes: List of (original, corrected) pairs
            - oov_no_suggest: List of words with no viable suggestions
        """
        if max_dist is None:
            max_dist = self.config.MAX_EDIT_DISTANCE
        
        corrected = []
        changes = []
        oov_no_suggest = []
        
        for w in words:
            if w in word_vocab:
                corrected.append(w)
                continue
            
            suggestion, dist = self.suggest_correction(w, word_vocab, word_freq, by_len_index, max_dist=max_dist)
            
            if suggestion is not None:
                corrected.append(suggestion)
                if suggestion != w:
                    changes.append((w, suggestion))
            else:
                corrected.append(w)
                oov_no_suggest.append(w)
        
        return corrected, changes, oov_no_suggest
    
    def get_similar_words(self, word: str, word_vocab: Set[str], word_freq: Dict[str, int], 
                         by_len_index: Dict[int, List[str]], max_dist: int = None, 
                         top_k: int = 5) -> List[Tuple[str, int, int]]:
        """
        Get similar words to a given word.
        
        Args:
            word: Input word.
            word_vocab: Set of vocabulary words.
            word_freq: Word frequency dictionary.
            by_len_index: Length-based index of vocabulary.
            max_dist: Maximum edit distance to consider.
            top_k: Number of top similar words to return.
            
        Returns:
            List of (word, distance, frequency) tuples sorted by distance then frequency.
        """
        if max_dist is None:
            max_dist = self.config.MAX_EDIT_DISTANCE
        
        similar_words = []
        candidates = self._candidate_words(word, word_vocab, by_len_index, max_len_diff=max_dist)
        
        for cand in candidates:
            dist = Levenshtein.distance(word, cand)
            if dist <= max_dist:
                freq = word_freq.get(cand, 0)
                similar_words.append((cand, dist, freq))
        
        # Sort by distance first, then by frequency (descending)
        similar_words.sort(key=lambda x: (x[1], -x[2]))
        return similar_words[:top_k]
    
    def is_valid_word(self, word: str, word_vocab: Set[str]) -> bool:
        """
        Check if a word is in the vocabulary.
        
        Args:
            word: Word to check.
            word_vocab: Set of vocabulary words.
            
        Returns:
            True if word is in vocabulary, False otherwise.
        """
        return word in word_vocab
    
    def get_word_frequency(self, word: str, word_freq: Dict[str, int]) -> int:
        """
        Get the frequency of a word.
        
        Args:
            word: Word to look up.
            word_freq: Word frequency dictionary.
            
        Returns:
            Frequency of the word, 0 if not found.
        """
        return word_freq.get(word, 0)
    
    def summarize_autocorrect_stats(self, word_vocab: Set[str], word_freq: Dict[str, int]) -> None:
        """
        Print summary statistics about the auto-correction system.
        
        Args:
            word_vocab: Set of vocabulary words.
            word_freq: Word frequency dictionary.
        """
        if not word_vocab or not word_freq:
            print("No auto-correction statistics available.")
            return
        
        print("\n=== Auto-correction Statistics ===")
        print(f"Vocabulary size: {len(word_vocab)}")
        print(f"Total word occurrences: {sum(word_freq.values())}")
        
        # Word length statistics
        word_lengths = [len(word) for word in word_vocab]
        if word_lengths:
            print(f"Word length range: {min(word_lengths)} - {max(word_lengths)}")
            print(f"Average word length: {sum(word_lengths) / len(word_lengths):.2f}")
        
        # Frequency statistics
        freq_values = list(word_freq.values())
        if freq_values:
            print(f"Frequency range: {min(freq_values)} - {max(freq_values)}")
            print(f"Average frequency: {sum(freq_values) / len(freq_values):.2f}")
        
        # Most common words
        most_common = word_freq.most_common(5)
        print(f"Top 5 most frequent words: {[word for word, freq in most_common]}")
