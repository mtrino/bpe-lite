import re
import regex
import pickle
from typing import List, Dict, Tuple, Iterable, Iterator

class Tokenizer:

    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        # Create inverse vocab for O(1) encoding
        self.inverse_vocab = {v: k for k, v in vocab.items()}
        # Create merge ranks for O(1) priority lookup
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        # Loading the vocabulary from vocab json
        with open(vocab_filepath, 'rb') as file:
            vocab = pickle.load(file)

        # Loading the merges
        with open(merges_filepath, 'rb') as file:
            merges = pickle.load(file)

        return cls(vocab, merges, special_tokens)
    
    def _pretokenize(self, text: str) -> List[str]:
        """Pretokenize a text into a list of strings with unbroken special tokens and according to the pattern.
        
        Returns.
            result: A list of strings
        """
        PAT = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

        if len(text) < 2:
            return [text]
        
        if self.special_tokens:
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            split_pattern = "|".join(re.escape(t) for t in sorted_special)
            parts = re.split(f"({split_pattern})", text)
        else:
            parts = [text]

        result = []
        for part in parts:
            if part in self.special_tokens:
                result.append(part)
            else:
                matches = [m.group(0) for m in regex.finditer(PAT, part)]
                result.extend(matches)
        return result

    def _convert_tok_to_bytes(self, tok_list: List[str]) -> List[Tuple[bytes, ...]]:
        """Given a list of string, convert it to a list of single-byte bytes object tuples
        
        Returns.
            result: List containing tuples of single-byte bytes object for each token of the input string
        """
        result = []
        for tok in tok_list:
            if tok in self.special_tokens:
                result.append((tok.encode('utf-8'), ))
            else:
                tok_bytes = [bytes([b]) for b in tok.encode('utf-8')]
                result.append(tuple(tok_bytes))
        return result
    
    def _get_byte_pairs(self, tok_bytes: Tuple[bytes, ...]) -> List[Tuple[bytes, bytes]]:
        """Given a tuple of bytes, this returns the list of consecutive bytes
        
        Returns.
            result: A list of consecutive pairs of the input bytes
        """
        result = []
        for pair in zip(tok_bytes[:-1], tok_bytes[1:]):
            result.append(pair)
        return result
    
    def _get_merge_order(self, pairs: List[Tuple[bytes, bytes]]) -> List[Tuple[bytes, bytes]]:
        """Given a list of consecutive bytes, this returns the order of eligible merges.
        
        Returns:
            order: Order of merges to be done from the pair input
        """
        common_pairs = [pair for pair in pairs if pair in self.merges]
        if common_pairs:
            order = sorted(common_pairs, key=lambda x: self.merge_ranks[x])
        else:
            order = []
        return order
    
    def _merge(self, tok_bytes: Tuple[bytes, ...], pair: Tuple[bytes, bytes]) -> List[Tuple[bytes, ...]]:
        """Merges the pair in the token bytes.
        
        Returns.
            new_tokens: List of tokens resulting from merging the pair in the original token bytes
        """
        new_tokens = []
        i = 0
        while i < len(tok_bytes):
            if i < len(tok_bytes) - 1 and pair[0] == tok_bytes[i] and pair[1] == tok_bytes[i+1]:
                new_tokens.append(pair[0] + pair[1])
                i += 2
            else:
                new_tokens.append(tok_bytes[i])
                i += 1
        return new_tokens
    
    def encode(self, text: str) -> List[int]:
        """Given a string, this encodes the string into integer ids according to the vocab and merges.
        
        Returns.
            ids: List on integers resulting from encoding the string to integers
        """
        result = []
        pretokens = self._pretokenize(text)
        bytes_list = self._convert_tok_to_bytes(pretokens)
        
        for byte_list in bytes_list:
            merged_bytes = byte_list
            while len(merged_bytes) >= 2:
                pairs = self._get_byte_pairs(merged_bytes)
                ordered_merge_pairs = self._get_merge_order(pairs)
                if not ordered_merge_pairs:
                    break
                
                best_pair = ordered_merge_pairs[0]
                merged_bytes = self._merge(merged_bytes, best_pair)
                
            result.extend(merged_bytes)

        ids = []
        for tok in result:
            if tok in self.inverse_vocab:
                ids.append(self.inverse_vocab[tok])
            else:
                print(f"Warning: Unknown token {tok}")
        return ids
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Lazily encodes an iterable of strings.

        Returns.
            tokens: Iterator of tens lazily
        """
        for chunk in iterable:
            chunk_ids = self.encode(chunk)
            for token in chunk_ids:
                yield token
    
    def decode(self, token_ids: List[int]) -> str:
        """Given a list of integer ids, this returns a string
        
        Return.
            string: String representation of the integer ids after decoding
        """
        byte_parts = [self.vocab[token] for token in token_ids]
        all_bytes = b''.join(byte_parts)
        return all_bytes.decode('utf-8', errors='replace')

