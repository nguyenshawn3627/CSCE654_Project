import argparse
import os
import json
from collections import Counter

# ----------------------------------------------------------
# HOW TO RUN: python3 training.py     --data training_corpus.txt     --vocab_size 5000     --out training_data     --dump_readable
# OUTPUT: ./training_data
#            L> merges_readable.txt
#            L> merges.txt
#            L> vocab.json
# ----------------------------------------------------------

# ----------------------------------------------------------
# Helper: read corpus as raw bytes
# ----------------------------------------------------------
def load_corpus_as_byte_sequences(filenames):
    sequences = []
    for fname in filenames:
        with open(fname, "rb") as f:
            data = f.read()
            sequences.append(list(data))  # list of ints 0–255
    return sequences


# ----------------------------------------------------------
# Utilities for printing tokens as hex (for merges/vocab)
# ----------------------------------------------------------
def token_to_hex(t):
    """Convert a tuple of bytes into a hex string like '74 68' for 'th'."""
    return " ".join(f"{b:02x}" for b in t)


def save_vocab_json(vocab, out_dir):
    vocab_path = os.path.join(out_dir, "vocab.json")
    # Convert tuple tokens to hex strings
    hex_vocab = {token_to_hex(tok): idx for tok, idx in vocab.items()}
    with open(vocab_path, "w") as f:
        json.dump(hex_vocab, f, indent=2)
    print(f"Saved vocab.json → {vocab_path}")

# ----------------------------------------------------------
# Human-readable dump of hex merges (for debugging only)
# ----------------------------------------------------------
def decode_token(hex_bytes):
    """Convert '74 68' into readable ASCII, escaping non-printables."""
    bs = bytes.fromhex(hex_bytes)

    readable = ""
    for b in bs:
        if 32 <= b <= 126:  # printable ASCII range
            readable += chr(b)
        elif b == 0x0A:
            readable += "\\n"
        elif b == 0x09:
            readable += "\\t"
        else:
            readable += f"\\x{b:02x}"
    return readable


def write_readable_merges(merges_txt_path, out_path):
    """Create a human-readable version of merges.txt."""
    with open(merges_txt_path, "r") as f:
        lines = f.readlines()

    with open(out_path, "w") as f:
        f.write("# Human-readable merge rules\n")
        f.write("# Format: HEX → \"A\" + \"B\"\n\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()

            mid = len(parts) // 2
            A_hex = " ".join(parts[:mid])
            B_hex = " ".join(parts[mid:])

            A_read = decode_token(A_hex)
            B_read = decode_token(B_hex)

            f.write(f"{A_hex}   {B_hex}    →    \"{A_read}\" + \"{B_read}\"\n")

# ----------------------------------------------------------
# Train BPE merges on raw byte sequences
# ----------------------------------------------------------
def train_byte_bpe(byte_sequences, vocab_size, out_dir,
                   min_pair_count=2, max_merges=None):
    """
    byte_sequences : list[list[int]]
        Each inner list is a sequence of bytes (0–255).
    vocab_size : int
        Target *maximum* vocab size (including 256 byte tokens).
        Actual vocab may be smaller if no frequent pairs remain.
    min_pair_count : int
        Stop if the most frequent pair appears fewer than this many times.
    max_merges : int or None
        Hard cap on number of merges. If None, defaults to vocab_size - 256.
    """
    os.makedirs(out_dir, exist_ok=True)

    if max_merges is None:
        max_merges = max(0, vocab_size - 256)

    print("Preparing sequences...")

    # Start with each byte as its own token: (b,)
    sequences = []
    for seq in byte_sequences:
        sequences.append([(b,) for b in seq])

    # Initial vocab: 256 byte tokens
    vocab = {(i,): i for i in range(256)}
    merges = []
    current_vocab_size = 256

    print("Training BPE...")
    print(f"Target vocab size ≤ {vocab_size}, max merges = {max_merges}")

    # BPE loop
    while current_vocab_size < vocab_size and len(merges) < max_merges:
        pair_counts = Counter()

        # Count adjacent pairs of tokens
        for seq in sequences:
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i+1])
                pair_counts[pair] += 1

        if not pair_counts:
            print("No more pairs to merge. Stopping.")
            break

        # Most frequent pair and its frequency
        (A, B), freq = pair_counts.most_common(1)[0]

        # If the best pair is too rare, further merges are not helpful
        if freq < min_pair_count:
            print(f"Stopping: best pair frequency = {freq} < min_pair_count = {min_pair_count}")
            break

        merged = A + B  # tuple concatenation

        # Safety: avoid accidental duplicates (very unlikely, but just in case)
        if merged in vocab:
            print("Merged token already in vocab; stopping to avoid collisions.")
            break

        merges.append((A, B))
        vocab[merged] = current_vocab_size
        current_vocab_size += 1

        # Replace occurrences of A B with merged token
        for si, seq in enumerate(sequences):
            i = 0
            new_seq = []
            L = len(seq)

            while i < L:
                if i < L - 1 and seq[i] == A and seq[i+1] == B:
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1

            sequences[si] = new_seq

        if len(merges) % 500 == 0:
            print(f"Merges learned: {len(merges)}, current vocab size: {current_vocab_size}")

    print(f"\nFinished training.")
    print(f"Total merges learned: {len(merges)}")
    print(f"Final vocab size: {current_vocab_size}")

    # ---------------------------------------------------------
    # Save merges.txt (tokens written in hex)
    # ---------------------------------------------------------
    merges_path = os.path.join(out_dir, "merges.txt")
    with open(merges_path, "w") as f:
        f.write("#version: raw-byte-bpe\n")
        for A, B in merges:
            f.write(f"{token_to_hex(A)} {token_to_hex(B)}\n")
    print(f"Saved merges.txt → {merges_path}")

    # ---------------------------------------------------------
    # Save vocab.json
    # ---------------------------------------------------------
    save_vocab_json(vocab, out_dir)


def byte_to_printable(b: int) -> str:
    """Convert a single byte to a readable symbol."""
    if 32 <= b <= 126:           # printable ASCII
        if chr(b) == " ":
            return "▁"           # HF uses special whitespace markers (optional)
        return chr(b)
    elif b == 0x0A:
        return "\\n"
    elif b == 0x09:
        return "\\t"
    else:
        return f"\\x{b:02x}"


def token_bytes_to_string(token_bytes: bytes) -> str:
    """Convert a multi-byte token into readable string."""
    out = ""
    for b in token_bytes:
        out += byte_to_printable(b)
    return out


def write_hf_style_merges(hex_merges_path, out_path):
    """Write merges_readable.txt in HuggingFace GPT-2 style format."""
    with open(hex_merges_path, "r") as f:
        lines = f.readlines()

    with open(out_path, "w") as f:
        f.write("#version: hf-readable\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            mid = len(parts) // 2
            A_hex = " ".join(parts[:mid])
            B_hex = " ".join(parts[mid:])

            A = bytes.fromhex(A_hex)
            B = bytes.fromhex(B_hex)

            A_print = token_bytes_to_string(A)
            B_print = token_bytes_to_string(B)

            f.write(f"{A_print} {B_print}\n")

def bytes_to_unicode():
    """
    Returns the byte-to-unicode mapping used by GPT-2.
    This is 100% identical to HuggingFace + OpenAI GPT-2 tokenizer.
    """
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = bs[:]

    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1

    cs = [chr(c) for c in cs]
    return {b: c for b, c in zip(bs, cs)}

BYTE_TO_UNICODE = bytes_to_unicode()

def token_bytes_to_gpt2_unicode(token_bytes: bytes) -> str:
    return "".join(BYTE_TO_UNICODE[b] for b in token_bytes)

def write_hf_style_merges(hex_merges_path, out_path):
    """Produce merges in actual GPT-2 format using GPT-2 unicode byte mapping."""
    with open(hex_merges_path, "r") as f:
        lines = f.readlines()

    with open(out_path, "w") as f:
        f.write("#version: gpt2-readable\n")

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            mid = len(parts) // 2
            A_hex = " ".join(parts[:mid])
            B_hex = " ".join(parts[mid:])

            A = bytes.fromhex(A_hex)
            B = bytes.fromhex(B_hex)

            A_print = token_bytes_to_gpt2_unicode(A)
            B_print = token_bytes_to_gpt2_unicode(B)

            # EXACT GPT-2 pairs format:
            f.write(f"{A_print} {B_print}\n")

# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a raw-byte-level BPE tokenizer (safe version)")
    parser.add_argument("--data", nargs="+", required=True, help="Corpus text files")
    parser.add_argument("--vocab_size", type=int, default=50000, help="Target *max* vocab size")
    parser.add_argument("--out", type=str, default="byte_bpe", help="Output directory")
    parser.add_argument("--min_pair_count", type=int, default=2,
                        help="Minimum frequency for a pair to be merged")
    parser.add_argument("--max_merges", type=int, default=None,
                        help="Hard cap on number of merges (optional)")
    parser.add_argument("--dump_readable", action="store_true",
                    help="Write merges_readable.txt for human inspection")
    args = parser.parse_args()

    byte_sequences = load_corpus_as_byte_sequences(args.data)
    train_byte_bpe(byte_sequences,
                   vocab_size=args.vocab_size,
                   out_dir=args.out,
                   min_pair_count=args.min_pair_count,
                   max_merges=args.max_merges)

if args.dump_readable:
    merges_path = os.path.join(args.out, "merges.txt")
    readable_out = os.path.join(args.out, "merges_readable.txt")
    write_hf_style_merges(merges_path, readable_out)
    print(f"GPT-2-style merges written to {readable_out}")
