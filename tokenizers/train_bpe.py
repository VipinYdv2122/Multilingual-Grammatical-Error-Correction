#!/usr/bin/env python3
"""
Train BPE tokenizers at multiple merge levels for each language.

Usage:
    python tokenizers/train_bpe.py --lang english --data data/processed/english
    python tokenizers/train_bpe.py --lang english --data data/processed/english --test-mode

This script:
  1. Reads source and target plain-text files (trains on both jointly).
  2. Trains BPE models at 8 merge levels:
       500, 1000, 2000, 4000, 8000, 16000, 25000, 32000
     producing files: bpe_0.5K.codes, bpe_1K.codes, ..., bpe_32K.codes
  3. No <unk> token — BPE covers all characters by construction.
  4. Saves codes to tokenizers/models/<lang>/

The subword-nmt backend (default) produces .codes files applied by apply_bpe.sh.
The hf backend produces .json tokenizers (alternative, less common).
"""

import argparse
import os
import sys
from pathlib import Path

try:
    from subword_nmt.learn_bpe import learn_bpe
    from subword_nmt.apply_bpe import BPE
    HAS_SUBWORD_NMT = True
except ImportError:
    HAS_SUBWORD_NMT = False

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    HAS_HF_TOKENIZERS = True
except ImportError:
    HAS_HF_TOKENIZERS = False


# ─── Config ───────────────────────────────────────────────────────────────────
# 8 merge levels for the full tokenization grid
DEFAULT_MERGE_LEVELS = [500, 1000, 2000, 4000, 8000, 16000, 25000, 32000]

# Mapping from merge count → K-notation label
MERGE_LABELS = {
    500:   "0.5K",
    1000:  "1K",
    2000:  "2K",
    4000:  "4K",
    8000:  "8K",
    16000: "16K",
    25000: "25K",
    32000: "32K",
}

# BUG FIX: was ["<pad>", "<s>", "</s>"] which does NOT match the Vocabulary
# class in models/dataset.py (which uses "<sos>" and "<eos>").
# Using the wrong names caused the HF-backend tokenizer to reserve indices for
# tokens that Vocabulary never produces, shifting all real vocab indices by
# however many mismatched specials were inserted.
SPECIAL_TOKENS = ["<pad>", "<sos>", "<eos>"]


def merges_to_label(num_merges: int) -> str:
    """Convert merge count to K-notation label (e.g., 500 → '0.5K')."""
    if num_merges in MERGE_LABELS:
        return MERGE_LABELS[num_merges]
    # Fallback for custom merge values
    if num_merges >= 1000 and num_merges % 1000 == 0:
        return f"{num_merges // 1000}K"
    return f"{num_merges / 1000:.1f}K".replace(".0K", "K")


# ─── subword-nmt backend ──────────────────────────────────────────────────────
def train_bpe_subword_nmt(input_files, output_codes, num_merges):
    """Train BPE codes using subword-nmt (fairseq-compatible)."""
    import codecs

    # Concatenate all input files into one stream
    combined_text = []
    for fpath in input_files:
        with open(fpath, "r", encoding="utf-8") as f:
            combined_text.extend(f.readlines())

    combined_path = output_codes + ".tmp_corpus"
    with open(combined_path, "w", encoding="utf-8") as f:
        f.writelines(combined_text)

    print(f"  Training {num_merges} merges ...", end=" ", flush=True)
    with codecs.open(combined_path, encoding="utf-8") as infile:
        with codecs.open(output_codes, "w", encoding="utf-8") as outfile:
            learn_bpe(infile, outfile,
                      num_symbols=num_merges,
                      min_frequency=2,
                      verbose=False)
    print("done.")

    os.remove(combined_path)
    print(f"  ✓ BPE codes saved: {output_codes}")


# ─── HuggingFace backend ──────────────────────────────────────────────────────
def train_bpe_hf(input_files, output_dir, name, num_merges):
    """Train BPE tokenizer using HuggingFace tokenizers library."""
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder       = decoders.BPEDecoder()

    trainer = trainers.BpeTrainer(
        # vocab_size = merges + base characters + special tokens
        vocab_size=num_merges + 256 + len(SPECIAL_TOKENS),
        min_frequency=2,
        special_tokens=SPECIAL_TOKENS,   # BUG FIX: now matches Vocabulary
        show_progress=True,
    )

    tokenizer.train(files=[str(f) for f in input_files], trainer=trainer)

    out_path = os.path.join(output_dir, f"{name}.json")
    tokenizer.save(out_path)
    print(f"  ✓ HF tokenizer saved: {out_path}")
    return tokenizer


# ─── Sanity check ────────────────────────────────────────────────────────────
def test_tokenizer(codes_path, sample_text):
    """Verify that BPE tokenisation produces no <unk> tokens."""
    if not HAS_SUBWORD_NMT:
        print("  ⚠ subword-nmt not installed, skipping test")
        return

    import codecs
    with codecs.open(codes_path, encoding="utf-8") as codes_file:
        bpe = BPE(codes_file)

    for line in sample_text:
        encoded = bpe.process_line(line.strip())
        tokens  = encoded.split()
        assert "<unk>" not in tokens, f"Found <unk> in: {encoded}"
        print(f"  Input:  {line.strip()}")
        print(f"  BPE:    {encoded}")
        print(f"  Tokens: {len(tokens)}")
        print()

    print("  ✓ No <unk> tokens — BPE covers all inputs")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Train BPE tokenizers at multiple merge levels for one language")
    parser.add_argument("--lang",         required=True,
                        help="Language name (e.g. english, russian, hindi)")
    parser.add_argument("--data",         required=True,
                        help="Path to processed data dir with train.src / train.tgt")
    parser.add_argument("--merges",       type=str, default=None,
                        help="Comma-separated merge counts (default: 500,1000,2000,4000,8000,16000,25000,32000)")
    parser.add_argument("--backend",      choices=["subword_nmt", "hf"],
                        default="subword_nmt",
                        help="BPE library (default: subword_nmt)")
    parser.add_argument("--test-mode",    action="store_true",
                        help="Run sanity check on 5 sample lines after training")
    args = parser.parse_args()

    # Parse merge levels
    if args.merges:
        merge_levels = [int(x.strip()) for x in args.merges.split(",")]
    else:
        merge_levels = DEFAULT_MERGE_LEVELS

    data_dir  = Path(args.data)
    src_train = data_dir / "train.src"
    tgt_train = data_dir / "train.tgt"

    # Collect existing input files
    input_files = []
    for f in (src_train, tgt_train):
        if f.exists():
            input_files.append(f)
        else:
            print(f"  WARNING: {f} not found, skipping")
    if not input_files:
        print("ERROR: No input files found.  Run preprocess.py first.")
        sys.exit(1)

    out_dir = Path("tokenizers/models") / args.lang
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Training BPE for {args.lang} ===")
    print(f"  Input files  : {[str(f) for f in input_files]}")
    print(f"  Output dir   : {out_dir}")
    print(f"  Merge levels : {merge_levels}")
    print(f"  Labels       : {[merges_to_label(m) for m in merge_levels]}")
    print()

    if args.backend == "subword_nmt":
        if not HAS_SUBWORD_NMT:
            print("ERROR: subword-nmt not installed.  "
                  "Run: pip install subword-nmt")
            sys.exit(1)

        for num_merges in merge_levels:
            label = merges_to_label(num_merges)
            print(f"--- BPE {label} ({num_merges} merges) ---")
            codes_path = str(out_dir / f"bpe_{label}.codes")
            train_bpe_subword_nmt(input_files, codes_path, num_merges)
            print()

        if args.test_mode:
            sample = []
            with open(input_files[0], encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    sample.append(line)

            print("=== Sanity Check ===")
            for num_merges in merge_levels:
                label = merges_to_label(num_merges)
                codes_path = str(out_dir / f"bpe_{label}.codes")
                print(f"--- {label} BPE ---")
                test_tokenizer(codes_path, sample)

    elif args.backend == "hf":
        if not HAS_HF_TOKENIZERS:
            print("ERROR: tokenizers not installed.  "
                  "Run: pip install tokenizers")
            sys.exit(1)

        for num_merges in merge_levels:
            label = merges_to_label(num_merges)
            print(f"--- BPE {label} ({num_merges} merges) ---")
            train_bpe_hf(input_files, str(out_dir), f"bpe_{label}", num_merges)
            print()

    print("Done!")


if __name__ == "__main__":
    main()
