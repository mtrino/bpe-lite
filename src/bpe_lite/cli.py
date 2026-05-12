import argparse
import time
import os
import pickle
from .train import train 

def main():
    parser = argparse.ArgumentParser(description="Train a BPE Tokenizer")
    parser.add_argument("--input", type=str, required=True, help="Path to input text file")
    parser.add_argument("--vocab-size", type=int, default=10000, help="Target vocabulary size")
    parser.add_argument("--special-tokens", nargs='+', default=['<|endoftext|>'], help="List of special tokens")
    parser.add_argument("--out-dir", type=str, default=".", help="Directory to save vocab and merges")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    start = time.time()
    vocab, merges = train(args.input, args.vocab_size, args.special_tokens)
    
    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "vocab.pkl"), 'wb') as f:
        pickle.dump(vocab, f)
    with open(os.path.join(args.out_dir, "merges.pkl"), 'wb') as f:
        pickle.dump(merges, f)
        
    print(f"Time taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    main()