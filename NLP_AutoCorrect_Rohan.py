import os
import random
import re
import sentencepiece as spm
from collections import Counter, defaultdict
import tempfile
import math
import pickle
from rapidfuzz.distance import Levenshtein

# Global configuration flags
LOWERCASE = True  # use lowercase for training and tokenization
MAX_EDIT_DIST = 2  # threshold for auto-correct suggestions (tune as needed)
TOPK_RESULTS = 10      # number of results to display
SNIPPET_CHARS = 200    # snippet length in characters
RANKER = "cosine"      # ranking function: "cosine" (we're removing BM25)
NUM_BOOKS = 100       # number of books to sample from corpus (None = all)
corpus_dir = '/Users/rohanjain/Desktop/UMD - MSML/Sem 3/NLP/HW_1/Input_Books'

def load_books(corpus_dir, num_books=None, seed=42):
    files = [f for f in os.listdir(corpus_dir) if f.endswith('.txt')]
    random.seed(seed)
    random.shuffle(files)
    if num_books is not None:
        files = files[:num_books]
    books = {}
    for filename in files:
        path = os.path.join(corpus_dir, filename)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            books[filename] = f.read()
    return books


# --- Paragraph splitting utilities ---
def split_into_paragraphs(text, min_par_chars=30):
    """
    Split raw book text into paragraphs using 2+ newline boundaries,
    trim whitespace, and drop paragraphs shorter than min_par_chars.
    Returns a list of paragraph strings (order preserved).
    """
    # Split on two or more consecutive newlines
    raw_paras = re.split(r'\n{2,}', text)
    paras = []
    for seg in raw_paras:
        p = seg.strip()
        if len(p) >= min_par_chars:
            paras.append(p)
    return paras


def break_into_paragraphs(books, min_par_chars=30):
    """
    Assign global paragraph IDs and keep (book_id, para_idx_in_book, text).
    Returns a dict: para_id -> (book_id, para_idx_in_book, text)
    """
    paragraphs = {}
    para_id = 0
    for book_id, text in books.items():
        book_paras = split_into_paragraphs(text, min_par_chars=min_par_chars)
        for idx_in_book, ptext in enumerate(book_paras):
            paragraphs[para_id] = (book_id, idx_in_book, ptext)
            para_id += 1
    return paragraphs


# --- BPE utilities ---
def train_bpe_model(books, model_prefix="bpe", vocab_size=10000, lowercase=True):
    """
    Train a SentencePiece BPE model on the sampled books.
    Returns the path to the trained model file (e.g., 'bpe.model').
    """
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
    # Returns 'model' path
    return f"{model_prefix}.model"


def load_bpe(model_path):
    """
    Load a trained SentencePiece model from disk and return the processor.
    """
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    return sp


def tokenize_paragraphs(paragraphs, sp, lowercase=True):
    """
    Tokenize each paragraph using the provided SentencePiece processor.
    Returns:
      - para_tokens: dict[para_id] -> List[str] of tokens
      - vocab_set: set of all unique tokens
      - token_freq: Counter mapping token -> global frequency
    """
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

def summarize_build(books, paragraphs, para_tokens, vocab_set, min_par_chars, bpe_vocab_size, model_path):
    """
    Print a concise summary of important build metrics.
    """
    num_books = len(books)
    num_unique_books = len({b for (b, _, _) in paragraphs.values()})
    num_paragraphs = len(paragraphs)
    if num_paragraphs > 0:
        avg_par_chars = sum(len(text) for (_b, _i, text) in paragraphs.values()) / num_paragraphs
        avg_tokens = sum(len(para_tokens.get(pid, ())) for pid in paragraphs.keys()) / num_paragraphs
    else:
        avg_par_chars = 0.0
        avg_tokens = 0.0

    print("\n=== Build Summary ===")
    print(f"Books sampled: {num_books} (unique files: {num_unique_books})")
    print(f"Paragraphs: {num_paragraphs}  |  min_par_chars={min_par_chars}")
    print(f"BPE vocab size: {len(vocab_set)} (target={bpe_vocab_size})")
    print(f"Avg paragraph: {avg_par_chars:.1f} chars  |  Avg tokens: {avg_tokens:.1f}")
    print(f"BPE model: {model_path}")

# --- Step 3: Inverted Index ---
def build_inverted_index(para_tokens, with_positions=False):
    """
    Build an inverted index from paragraph tokens.

    Parameters
    ----------
    para_tokens : Dict[int, List[str]]
        Mapping from paragraph ID -> list of BPE token strings for that paragraph.
    with_positions : bool
        If True, also store token positions (offsets) within each paragraph.

    Returns
    -------
    inverted_index : Dict[str, List[tuple]]
        If with_positions=False: token -> List[(para_id, freq)]
        If with_positions=True:  token -> List[(para_id, freq, positions_list)]
        Postings lists are sorted by para_id for reproducibility.
    para_len_tokens : Dict[int, int]
        Paragraph token lengths: para_id -> number of tokens.
    N : int
        Total number of paragraphs.

    Notes
    -----
    - We aggregate per-paragraph counts with a Counter to avoid duplicate entries.
    - Positions are helpful for phrase/proximity search and highlighting later.
    """
    postings = defaultdict(list)      # token -> list of postings
    para_len_tokens = {}              # para_id -> length
    N = len(para_tokens)              # total paragraphs

    for pid, tokens in para_tokens.items():
        # Record paragraph token length for later ranking (e.g., BM25 uses doc length)
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

# --- TF-IDF / Cosine Retrieval Utilities ---
def compute_idf_and_docnorms(inverted_index, N):
    """
    Compute IDF per token and TF-IDF L2 norms per document (paragraph).
    IDF formula: idf = log((N + 1) / (df + 1)) + 1  (smooth, positive)
    TF weighting: tfw = 1 + log(tf)
    Returns:
      - idf: Dict[token, float]
      - doc_norm: Dict[para_id, float]  (sqrt of sum of squared tf-idf weights)
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

def cosine_rank(query_tokens, inverted_index, idf, doc_norm, topk=10):
    """
    Rank paragraphs using cosine similarity over TF-IDF vectors.
    Only tokens that appear in the query and the index contribute.
    Returns: List[(para_id, score)] sorted by score desc (length <= topk)
    """
    # Build query vector weights
    if not query_tokens:
        return []
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

# --- Output Tidy/Highlight Utilities ---
def _format_tokens(tokens, maxn=12):
    """
    Return tokens as a compact string; truncate long lists with an ellipsis.
    """
    if len(tokens) <= maxn:
        return "[" + ", ".join(tokens) + "]"
    head = ", ".join(tokens[:maxn//2])
    tail = ", ".join(tokens[-maxn//2:])
    return "[" + head + ", …, " + tail + "]"

def highlight_words(text, words):
    """
    Naive console-safe highlighter: wraps whole-word matches with [[ ]].
    Case-insensitive; ignores empty words.
    """
    if not words:
        return text
    # Deduplicate and sort longer-first to avoid partial overshadowing
    uniq = sorted({w for w in words if w}, key=len, reverse=True)
    def repl(match):
        return f"[[{match.group(0)}]]"
    # Build a combined regex of word boundaries for each word
    patterns = [r"\b" + re.escape(w) + r"\b" for w in uniq]
    if not patterns:
        return text
    regex = re.compile("|".join(patterns), flags=re.IGNORECASE)
    return regex.sub(repl, text)


def make_snippet(text, query_words, max_chars=200):
    """
    Produce a snippet with highlighted query words and trimmed to max_chars.
    Attempts to center around the first match; falls back to start of paragraph.
    """
    # Highlight first, then trim so brackets are visible
    highlighted = highlight_words(text, query_words)
    if len(highlighted) <= max_chars:
        return highlighted.replace("\n", " ")
    # Try to find the first highlight marker to center the snippet
    marker = highlighted.lower().find("[[")
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

# --- ASCII Table Results Printer ---
def print_results_table(ranked, paragraphs, query_tokens, max_chars=SNIPPET_CHARS, query_words_for_highlight=None):
    """
    Render top results as a clean ASCII table: Rank | PID | Score | Book | ParaIdx | Snippet
    """
    if not ranked:
        print("No matching paragraphs found.")
        return
    # Prepare rows
    rows = []
    for rank, (pid, score) in enumerate(ranked, start=1):
        book_id, idx_in_book, text = paragraphs.get(pid, ("?", -1, ""))
        snippet = make_snippet(text, query_words_for_highlight or [], max_chars=max_chars)
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
    q_preview = _format_tokens(query_tokens, maxn=12)
    print(f"\n(query tokens used: {q_preview})\n")

def pretty_print_results(ranked, paragraphs, query_tokens, max_chars=SNIPPET_CHARS, query_words_for_highlight=None):
    """
    Print a concise view of ranked results with book id, paragraph index, score, and a snippet.
    """
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
        snippet = make_snippet(text, query_words_for_highlight, max_chars=max_chars)
        print(f"#{rank}  pid={pid}  score={score:.4f}  book={book_id}  para_idx={idx_in_book}")
        print(f"     {snippet}")
    q_preview = _format_tokens(query_tokens, maxn=12)
    print(f"\n(query tokens used: {q_preview})\n")

# --- Phase 3 (Step 1): Word-level Vocabulary Build & Persist ---
WORD_REGEX = re.compile(r"[a-z0-9']+")

def build_word_vocab(paragraphs, lowercase=LOWERCASE):
    """
    Build a word-level vocabulary and frequency dictionary from raw paragraph text.
    Returns:
      - word_vocab: set of unique words
      - word_freq: Counter of word -> global count
    Notes:
      - This is separate from BPE; used for auto-correct decisions.
    """
    freq = Counter()
    for (_pid, (_book, _idx, text)) in paragraphs.items():
        txt = text.lower() if lowercase else text
        # extract simple word tokens (alnum + apostrophes)
        words = WORD_REGEX.findall(txt)
        if words:
            freq.update(words)
    word_vocab = set(freq.keys())
    return word_vocab, freq

def save_word_vocab(word_vocab, word_freq, prefix="word_vocab"):
    """
    Persist word-level vocab and frequencies to disk using pickle.
    Creates <prefix>_set.pkl and <prefix>_freq.pkl in the current directory.
    """
    with open(f"{prefix}_set.pkl", "wb") as f:
        pickle.dump(word_vocab, f)
    with open(f"{prefix}_freq.pkl", "wb") as f:
        pickle.dump(word_freq, f)

def load_word_vocab(prefix="word_vocab"):
    """
    Load persisted word-level vocab and frequencies if available; else return (None, None).
    """
    try:
        with open(f"{prefix}_set.pkl", "rb") as f:
            vocab = pickle.load(f)
        with open(f"{prefix}_freq.pkl", "rb") as f:
            freq = pickle.load(f)
        return vocab, freq
    except FileNotFoundError:
        return None, None


def summarize_word_vocab(word_freq, topn=10):
    """
    Print a concise summary of word-level vocabulary.
    """
    total = sum(word_freq.values()) if word_freq else 0
    uniq = len(word_freq) if word_freq else 0
    print("\n=== Word-level Vocabulary Summary ===")
    print(f"Unique words: {uniq}  |  Total word occurrences: {total}")
    if word_freq and topn > 0:
        most_common = word_freq.most_common(topn)
        preview = ", ".join(f"{w}:{c}" for w, c in most_common)
        print(f"Top {topn} words: {preview}")

# --- Phase 3 (Step 2): Auto-correct helpers ---
def _candidate_words(word, word_vocab, by_len_index, max_len_diff=None):
    """Yield candidate words from vocab within a length band to reduce comparisons."""
    if max_len_diff is None:
        max_len_diff = MAX_EDIT_DIST
    L = len(word)
    for dL in range(-max_len_diff, max_len_diff + 1):
        bucket = by_len_index.get(L + dL)
        if not bucket:
            continue
        for w in bucket:
            yield w

def build_len_index(word_vocab):
    """Build a simple length→list index to prune candidates quickly."""
    index = defaultdict(list)
    for w in word_vocab:
        index[len(w)].append(w)
    return index

def suggest_correction(word, word_vocab, word_freq, by_len_index, max_dist=None):
    """
    Return (best_word, best_dist) if a candidate within max_dist exists; else (None, None).
    Tie-breaks by: smaller distance first, then higher frequency.
    """
    if max_dist is None:
        max_dist = MAX_EDIT_DIST
    best_word, best_dist, best_freq = None, None, -1
    for cand in _candidate_words(word, word_vocab, by_len_index, max_len_diff=max_dist):
        dist = Levenshtein.distance(word, cand)
        if dist <= max_dist:
            freq = word_freq.get(cand, 0)
            if (best_dist is None) or (dist < best_dist) or (dist == best_dist and freq > best_freq):
                best_word, best_dist, best_freq = cand, dist, freq
            if best_dist == 0:
                break
    return best_word, best_dist

def autocorrect_query_words(words, word_vocab, word_freq, by_len_index, max_dist=None):
    """
    Given a list of lowercased words, return (corrected_words, changes, oov_no_suggest).
    - corrected_words: list[str] after applying corrections when within max_dist
    - changes: list[(original, corrected)] for reporting to user
    - oov_no_suggest: list[str] of words not in vocab and with no viable suggestion (to BPE-segment)
    """
    if max_dist is None:
        max_dist = MAX_EDIT_DIST
    corrected = []
    changes = []
    oov_no_suggest = []
    for w in words:
        if w in word_vocab:
            corrected.append(w)
            continue
        suggestion, dist = suggest_correction(w, word_vocab, word_freq, by_len_index, max_dist=max_dist)
        if suggestion is not None:
            corrected.append(suggestion)
            if suggestion != w:
                changes.append((w, suggestion))
        else:
            corrected.append(w)
            oov_no_suggest.append(w)
    return corrected, changes, oov_no_suggest

def summarize_vocabulary(token_freq, topn=10):
    """
    Print a concise summary of vocabulary statistics and the most frequent tokens.
    """
    total_occurrences = sum(token_freq.values())
    unique_tokens = len(token_freq)
    print("\n=== Vocabulary Summary ===")
    print(f"Unique tokens: {unique_tokens}  |  Total token occurrences: {total_occurrences}")
    if topn > 0 and unique_tokens > 0:
        most_common = token_freq.most_common(topn)
        preview = ", ".join(f"{tok}:{cnt}" for tok, cnt in most_common)
        print(f"Top {topn} tokens: {preview}")

# --- Phase 2: User Interaction & Search (Part 1: Query Input + Token Check) ---
def interactive_query_loop(sp, vocab_set, inverted_index, idf, doc_norm, paragraphs, word_vocab, word_freq, by_len_index, lowercase=LOWERCASE):
    """
    Start an interactive loop that:
      - Prompts the user for a query
      - Tokenizes the query with the trained BPE tokenizer
      - Reports which tokens are in-vocab vs missing
      - Performs BM25 retrieval using in-vocab tokens
    Type 'exit' to quit.
    """
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

        qraw = query.lower() if lowercase else query
        # Extract word tokens for auto-correct step
        q_words = WORD_REGEX.findall(qraw)
        corrected_words, changes, oov_no_suggest = autocorrect_query_words(q_words, word_vocab, word_freq, by_len_index, max_dist=MAX_EDIT_DIST)
        corrected_query = " ".join(corrected_words)
        if changes:
            # Show a compact correction summary and proceed using corrected terms
            print(f"Did you mean: \"{corrected_query}\"? (using corrected terms)")
        # For words with no viable suggestion, show BPE fallback segmentation
        for w in oov_no_suggest:
            sub_tokens = sp.encode(w, out_type=str)
            print(f"No close match for \"{w}\". Using subwords: {sub_tokens}")

        # Now tokenize with BPE for retrieval using the corrected text
        q_tokens = sp.encode(corrected_query, out_type=str)

        # Partition tokens into present vs missing
        in_vocab = [t for t in q_tokens if t in vocab_set]
        missing = [t for t in q_tokens if t not in vocab_set]

        print("\n--- Query ---")
        print("Original :", query)
        if changes:
            print("Corrected:", corrected_query)
        if oov_no_suggest:
            print("BPE fallback terms:", ", ".join(oov_no_suggest))
        print("Tokens   :", _format_tokens(q_tokens, maxn=12))
        print()

        # Retrieve and rank using cosine TF-IDF over BPE tokens
        if in_vocab:
            ranked = cosine_rank(in_vocab, inverted_index, idf, doc_norm, topk=TOPK_RESULTS)
            if ranked:
                print_results_table(ranked, paragraphs, in_vocab, max_chars=SNIPPET_CHARS, query_words_for_highlight=corrected_words)
            else:
                print("No paragraph found for this query.")
        else:
            print("No paragraph found for this query.")


books = load_books(corpus_dir, num_books=NUM_BOOKS)

# Build paragraph mapping with a configurable minimum paragraph character threshold
min_par_chars = 30  # change this as needed
paragraphs = break_into_paragraphs(books, min_par_chars=min_par_chars)

# --- Phase 3 Step 1: Build & Persist Word-level Vocabulary ---
# Try to load existing word-level vocab; if not present, build and save
word_vocab, word_freq = load_word_vocab(prefix="word_vocab_hw1")
if word_vocab is None or word_freq is None:
    word_vocab, word_freq = build_word_vocab(paragraphs, lowercase=LOWERCASE)
    save_word_vocab(word_vocab, word_freq, prefix="word_vocab_hw1")
summarize_word_vocab(word_freq, topn=10)
# --- Build length index for fast candidate lookup (auto-correct) ---
by_len_index = build_len_index(word_vocab)

# --- Step 2: Train BPE & Tokenize ---
bpe_vocab_size = 10000   # change as needed
bpe_model_prefix = "bpe_model_hw1"
model_path = train_bpe_model(books, model_prefix=bpe_model_prefix, vocab_size=bpe_vocab_size, lowercase=LOWERCASE)
sp = load_bpe(model_path)

para_tokens, vocab_set, token_freq = tokenize_paragraphs(paragraphs, sp, lowercase=LOWERCASE)
summarize_build(books, paragraphs, para_tokens, vocab_set, min_par_chars, bpe_vocab_size, model_path)

 # --- Step 3: Build Inverted Index ---
with_positions = False  # set True if you want positions for phrase/proximity search
inverted_index, para_len_tokens, N = build_inverted_index(para_tokens, with_positions=with_positions)
print(f"Inverted index built for {len(inverted_index)} tokens across {N} paragraphs. with_positions={with_positions}")

# Precompute IDF and document norms for cosine similarity
idf, doc_norm = compute_idf_and_docnorms(inverted_index, N)

# --- Step 4: Vocabulary (from Step 2) ---
# Use vocab_set and token_freq produced during tokenization; do not recompute.
summarize_vocabulary(token_freq, topn=10)

# --- Start Interactive Loop Immediately ---
interactive_query_loop(sp, vocab_set, inverted_index, idf, doc_norm, paragraphs, word_vocab, word_freq, by_len_index, lowercase=LOWERCASE)
