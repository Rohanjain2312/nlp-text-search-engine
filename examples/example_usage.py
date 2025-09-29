#!/usr/bin/env python3
"""
Example usage of the NLP Text Search Engine.

This script demonstrates how to use the search engine programmatically
for various search tasks.
"""

import sys
from pathlib import Path

# Add parent directory to path to import search_engine
sys.path.append(str(Path(__file__).parent.parent))

from search_engine import TextSearchEngine
import config


def basic_search_example():
    """Demonstrate basic search functionality."""
    print("=== Basic Search Example ===")
    
    # Initialize search engine
    engine = TextSearchEngine()
    
    # Build index
    print("Building index...")
    engine.build_index()
    
    # Perform searches
    queries = [
        "machine learning",
        "artificial intelligence",
        "natural language processing",
        "deep learning",
        "neural networks"
    ]
    
    for query in queries:
        print(f"\nSearching for: '{query}'")
        results = engine.search(query, top_k=5)
        
        if results:
            print(f"Found {len(results)} results:")
            for i, (para_id, score) in enumerate(results[:3], 1):
                book_id, para_idx, text = engine.paragraphs[para_id]
                snippet = engine.get_result_snippet(para_id, query.split())
                print(f"  {i}. Score: {score:.4f} | Book: {book_id} | Para: {para_idx}")
                print(f"     {snippet[:100]}...")
        else:
            print("  No results found.")


def autocorrect_example():
    """Demonstrate auto-correction functionality."""
    print("\n=== Auto-correction Example ===")
    
    engine = TextSearchEngine()
    engine.build_index()
    
    # Test queries with typos
    test_queries = [
        "machne learning",  # typo in "machine"
        "artifical inteligence",  # typos in "artificial" and "intelligence"
        "naturl langage processing",  # typos in "natural" and "language"
        "deep lerning",  # typo in "learning"
        "neural netwroks"  # typo in "networks"
    ]
    
    for query in test_queries:
        print(f"\nOriginal query: '{query}'")
        
        # Extract words and show auto-correction
        words = engine.text_processor.extract_words(query)
        corrected_words, changes, oov = engine.auto_correct.autocorrect_query_words(
            words, engine.word_vocab, engine.word_freq, engine.by_len_index
        )
        
        if changes:
            print(f"Corrections: {changes}")
        if oov:
            print(f"Words with no suggestions: {oov}")
        
        corrected_query = " ".join(corrected_words)
        print(f"Corrected query: '{corrected_query}'")
        
        # Search with corrected query
        results = engine.search(corrected_query, top_k=3)
        print(f"Results: {len(results)} documents found")


def advanced_search_example():
    """Demonstrate advanced search features."""
    print("\n=== Advanced Search Example ===")
    
    engine = TextSearchEngine()
    engine.build_index()
    
    # Get index statistics
    stats = engine.get_stats()
    print("Index Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test different query types
    queries = [
        "machine learning algorithms",
        "deep neural networks",
        "natural language understanding",
        "computer vision applications",
        "artificial intelligence research"
    ]
    
    print(f"\nTesting {len(queries)} different queries:")
    
    for query in queries:
        results = engine.search(query, top_k=10)
        print(f"'{query}': {len(results)} results")
        
        if results:
            # Show top result details
            top_result = results[0]
            para_id, score = top_result
            book_id, para_idx, text = engine.paragraphs[para_id]
            
            print(f"  Top result: {book_id} (para {para_idx}) - Score: {score:.4f}")
            
            # Get detailed snippet
            snippet = engine.get_result_snippet(para_id, query.split(), max_chars=300)
            print(f"  Snippet: {snippet}")


def performance_test():
    """Test search performance with multiple queries."""
    print("\n=== Performance Test ===")
    
    import time
    
    engine = TextSearchEngine()
    engine.build_index()
    
    # Test queries
    test_queries = [
        "machine learning",
        "artificial intelligence", 
        "natural language processing",
        "deep learning",
        "neural networks",
        "computer vision",
        "data science",
        "algorithm",
        "programming",
        "software engineering"
    ]
    
    print(f"Testing performance with {len(test_queries)} queries...")
    
    total_time = 0
    total_results = 0
    
    for query in test_queries:
        start_time = time.time()
        results = engine.search(query, top_k=10)
        end_time = time.time()
        
        query_time = end_time - start_time
        total_time += query_time
        total_results += len(results)
        
        print(f"'{query}': {len(results)} results in {query_time:.3f}s")
    
    avg_time = total_time / len(test_queries)
    print(f"\nPerformance Summary:")
    print(f"  Average query time: {avg_time:.3f}s")
    print(f"  Total results: {total_results}")
    print(f"  Queries per second: {1/avg_time:.2f}")


def main():
    """Run all examples."""
    print("NLP Text Search Engine - Example Usage")
    print("=" * 50)
    
    try:
        basic_search_example()
        autocorrect_example()
        advanced_search_example()
        performance_test()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
