"""Train the Byte pair encoding with Inverted Index Optimization"""
import re
import time
import regex
import multiprocessing as mp
from collections import defaultdict
from typing import List, Dict, Set, Tuple

def _load_data(input_path: str):
    """Loads the text data from the input path."""
    with open(input_path, 'r', encoding='utf-8') as file:
        data = file.read()
    return data


def _pretokenize(chunk: str, pattern: str, special_tokens: List[str]) -> Dict[Tuple[bytes, ...], int]:
    """
    Pretokenize the text and create word counts dictionary, excluding special tokens for a single worker.
    """
    if special_tokens:
        split_pattern = "|".join(re.escape(t) for t in special_tokens)
        parts = re.split(f"({split_pattern})", chunk)
    else:
        parts = [chunk]
    
    local_counts = defaultdict(int)
    for part in parts:
        if part in special_tokens:
            continue
        
        matches = [m.group(0) for m in regex.finditer(pattern, part)]
        for m in matches:
            word_bytes = tuple(bytes([w]) for w in m.encode('utf-8'))
            local_counts[word_bytes] += 1
            
    return local_counts


def _parallel_pretokenize(text: str, pattern: str, special_tokens: List[str], n_jobs: int = -1) -> Dict[Tuple[bytes, ...], int]:
    """
    Pretokenize the chunks accross different workers and then merge
    """
    if n_jobs == -1:
        num_processes = mp.cpu_count()
    else:
        num_processes = n_jobs
    
    chunk_size = len(text) // num_processes

    chunks = []
    start = 0
    for _ in range(num_processes - 1):
        end = start + chunk_size
        while end < len(text) and not text[end].isspace():
            end += 1
        chunks.append(text[start:end])
        start = end
    chunks.append(text[start:])

    with mp.Pool(processes=num_processes) as pool:
        results = pool.starmap(_pretokenize, [(c, pattern, special_tokens) for c in chunks])

    final_counts = defaultdict(int)
    for result in results:
        for word, count in result.items():
            final_counts[word] += count

    return final_counts


def _init_vocab_and_index(word_counts: Dict[Tuple[bytes, ...], int]) -> Tuple[Dict[Tuple[bytes, bytes], int], Dict[Tuple[bytes, bytes], Set[Tuple[bytes, ...]]]]:
    """
    Initializes word_counts, pair_counts, and inverted_index from the list of words.
    
    Returns:
        pair_counts: Dict[pair -> total_count]
        inverted_index: Dict[pair -> Set[word_tuples]]
    """
    pair_counts = defaultdict(int)
    inverted_index = defaultdict(set)

    for word, freq in word_counts.items():
        # Iterate over all adjacent pairs in the word
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] += freq
            inverted_index[pair].add(word)
            
    return pair_counts, inverted_index


def update_counts_and_index(pair_counts: Dict[Tuple[bytes, bytes], int], inverted_index: Dict[Tuple[bytes, bytes], Set[Tuple[bytes, ...]]],
                            word_counts: Dict[Tuple[bytes, ...], int], best_pair: Tuple[bytes, bytes]) -> None:
    """
    Updates pair_counts, inverted_index, and word_counts after merging best_pair.
    Optimized to only process words containing the best_pair.
    """
    if best_pair not in inverted_index:
        return
        
    words_to_update = list(inverted_index[best_pair])
    
    del inverted_index[best_pair]
    
    for word in words_to_update:
        freq = word_counts[word]
        
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pair_counts[pair] -= freq
            if pair_counts[pair] == 0:
                del pair_counts[pair]
            
            if pair in inverted_index and word in inverted_index[pair]:
                inverted_index[pair].remove(word)
                if not inverted_index[pair]:
                    del inverted_index[pair]

        new_word_list = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == best_pair[0] and word[i+1] == best_pair[1]:
                new_word_list.append(best_pair[0] + best_pair[1])
                i += 2
            else:
                new_word_list.append(word[i])
                i += 1
        new_word = tuple(new_word_list)
        
        del word_counts[word]
        word_counts[new_word] = freq

        for i in range(len(new_word) - 1):
            pair = (new_word[i], new_word[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            
            if pair not in inverted_index:
                inverted_index[pair] = set()
            inverted_index[pair].add(new_word)


def train(input_path: str, vocab_size: int, special_tokens: List[str], n_jobs=1):
    """Trains a Byte Pair Encoding tokenizer using Inverted Index Optimization"""

    print(f"Loading data from {input_path}...")
    data = _load_data(input_path)
    original_length = len(data.encode("utf-8"))

    print("Pretokenizing...")
    PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    word_counts = _parallel_pretokenize(data, PAT, special_tokens, n_jobs=n_jobs)

    print("Initializing indices...")
    pair_counts, inverted_index = _init_vocab_and_index(word_counts)

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode('utf-8')

    num_merges = vocab_size - len(vocab)
    merges = []

    print(f"Starting training for {num_merges} merges...")
    start_time = time.time()
    
    new_starting_idx = len(vocab)
    
    for i in range(num_merges):
        if not pair_counts:
            break
            
        top_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        
        vocab[new_starting_idx + i] = top_pair[0] + top_pair[1]
        merges.append(top_pair)
        
        update_counts_and_index(pair_counts, inverted_index, word_counts, top_pair)
        
        if (i + 1) % 100 == 0:
            print(f"Merge {i+1}/{num_merges}: {top_pair} (Count: {pair_counts.get(top_pair, 'Merged')})")

    final_token_count = sum(word_counts.values())
    compression_ratio = original_length / final_token_count

    print(f"Training complete in {time.time() - start_time:.2f} seconds.")
    print(f"Original length: {original_length:,}")
    print(f"Final length: {final_token_count:,}")
    print(f"Compression ratio achieved: {compression_ratio:.2f}X")
    return vocab, merges
