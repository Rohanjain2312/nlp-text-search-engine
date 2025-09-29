# NLP Text Search Engine

A sophisticated text search engine built with BPE (Byte Pair Encoding) tokenization, TF-IDF ranking, and auto-correction capabilities. This project implements a complete information retrieval system that can search through large collections of text documents.

## Features

- **BPE Tokenization**: Uses SentencePiece for subword tokenization
- **TF-IDF Ranking**: Cosine similarity-based document ranking
- **Auto-correction**: Intelligent query correction using Levenshtein distance
- **Interactive Search**: Command-line interface for real-time searching
- **Snippet Generation**: Context-aware text snippets with query highlighting
- **Scalable Architecture**: Handles large document collections efficiently

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nlp-text-search-engine.git
cd nlp-text-search-engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your text corpus:
   - Create a directory called `Input_Books/` in the project root
   - Add your `.txt` files to this directory
   - The search engine will automatically process all text files in this directory
   - Example structure:
   ```
   Input_Books/
   ├── book1.txt
   ├── book2.txt
   └── ...
   ```

## Usage

### Basic Usage

```python
from search_engine import TextSearchEngine

# Initialize the search engine
engine = TextSearchEngine(corpus_dir="Input_Books/")

# Build the index (this may take some time for large corpora)
engine.build_index()

# Start interactive search
engine.interactive_search()
```

### Configuration

You can customize the search engine behavior by modifying `config.py`:

```python
# Search parameters
MAX_EDIT_DIST = 2          # Auto-correction threshold
TOPK_RESULTS = 10          # Number of results to return
SNIPPET_CHARS = 200        # Snippet length
BPE_VOCAB_SIZE = 10000     # BPE vocabulary size
NUM_BOOKS = 100           # Number of books to process (None for all)
```

### Command Line Interface

Run the main script directly:

```bash
python main.py
```

This will start an interactive search session where you can enter queries and get ranked results.

## Architecture

The search engine consists of several key components:

1. **Document Processing**: Splits text into paragraphs and preprocesses content
2. **Tokenization**: Uses BPE for subword tokenization
3. **Indexing**: Builds inverted index with TF-IDF weights
4. **Query Processing**: Handles auto-correction and query expansion
5. **Ranking**: Uses cosine similarity for document ranking
6. **Result Presentation**: Generates highlighted snippets and formatted results

## File Structure

```
nlp-text-search-engine/
├── README.md
├── requirements.txt
├── config.py
├── main.py
├── search_engine/
│   ├── __init__.py
│   ├── tokenizer.py
│   ├── indexer.py
│   ├── ranker.py
│   ├── autocorrect.py
│   └── utils.py
├── Input_Books/
│   └── *.txt files
└── examples/
    └── example_usage.py
```

## Dependencies

- Python 3.7+
- sentencepiece
- rapidfuzz
- numpy
- scipy

## Performance

The search engine is optimized for:
- Fast query processing (< 100ms for most queries)
- Memory-efficient indexing
- Scalable to large document collections
- Real-time auto-correction

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Academic Use

This project was developed as part of coursework in Natural Language Processing. If you use this code for academic purposes, please cite appropriately.

## Future Enhancements

- [ ] BM25 ranking implementation
- [ ] Phrase search capabilities
- [ ] Query expansion
- [ ] Web interface
- [ ] Distributed indexing
- [ ] Advanced auto-correction algorithms
