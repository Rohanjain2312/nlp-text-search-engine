#!/usr/bin/env python3
"""
Main entry point for the NLP Text Search Engine.

This script provides a command-line interface for the search engine.
"""

import argparse
import sys
from pathlib import Path

from search_engine import TextSearchEngine
import config


def main():
    """Main entry point for the search engine."""
    parser = argparse.ArgumentParser(
        description="NLP Text Search Engine with BPE tokenization and TF-IDF ranking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Start interactive search
  python main.py --corpus-dir ./books              # Use custom corpus directory
  python main.py --query "machine learning"        # Single query mode
  python main.py --build-only                      # Just build index, don't search
        """
    )
    
    parser.add_argument(
        "--corpus-dir", 
        type=str, 
        default=None,
        help="Directory containing text files (default: Input_Books/)"
    )
    
    parser.add_argument(
        "--query", 
        type=str, 
        default=None,
        help="Single query to process (non-interactive mode)"
    )
    
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=None,
        help="Number of results to return (default: 10)"
    )
    
    parser.add_argument(
        "--build-only", 
        action="store_true",
        help="Only build the index, don't start interactive search"
    )
    
    parser.add_argument(
        "--force-rebuild", 
        action="store_true",
        help="Force rebuild of the index even if it exists"
    )
    
    parser.add_argument(
        "--stats", 
        action="store_true",
        help="Show index statistics after building"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to custom configuration file"
    )
    
    args = parser.parse_args()
    
    # Initialize search engine
    try:
        engine = TextSearchEngine(corpus_dir=args.corpus_dir)
        print("Initialized Text Search Engine")
    except Exception as e:
        print(f"Error initializing search engine: {e}")
        sys.exit(1)
    
    # Build index
    try:
        print("Building search index...")
        engine.build_index(force_rebuild=args.force_rebuild)
        print("Index built successfully!")
    except Exception as e:
        print(f"Error building index: {e}")
        sys.exit(1)
    
    # Show statistics if requested
    if args.stats:
        stats = engine.get_stats()
        print("\n=== Index Statistics ===")
        for key, value in stats.items():
            print(f"{key}: {value}")
    
    # Handle different modes
    if args.build_only:
        print("Index building complete. Exiting.")
        return
    
    if args.query:
        # Single query mode
        try:
            results = engine.search(args.query, top_k=args.top_k)
            
            if results:
                query_words = engine.text_processor.extract_words(args.query)
                engine.result_formatter.print_results_table(
                    results, engine.paragraphs, query_words
                )
            else:
                print("No matching documents found.")
        except Exception as e:
            print(f"Error processing query: {e}")
            sys.exit(1)
    else:
        # Interactive mode
        try:
            engine.interactive_search()
        except KeyboardInterrupt:
            print("\nExiting.")
        except Exception as e:
            print(f"Error in interactive mode: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
