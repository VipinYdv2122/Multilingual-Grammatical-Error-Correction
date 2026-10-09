#!/usr/bin/env python3
"""
Train a Kneser-Ney LM-based recaser (truecasing model).

This script trains a truecaser using a Kneser-Ney smoothed n-gram language
model. The recommended setup is to train on the source column so the recaser
matches the input distribution without leaking gold target text.

The approach:
1. Build a mapping from lowercased words to all their casing variants seen in training
2. Train an n-gram language model on properly cased text
3. At inference: for each word, score all casing variants in context, pick the best

Usage:
    python scripts/train_recaser.py --lang english --input data/raw/english/train.csv
    python scripts/train_recaser.py --lang hindi --input data/raw/hindi/train.csv

Output:
    tokenizers/recaser/<lang>/
        lm.pkl              - Kneser-Ney language model
        casing_variants.json - word casing variants
        config.json         - model configuration
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.kn_recaser import train_kn_recaser


def _resolve_text_column(df, requested: str | None) -> str:
    """Resolve a requested column name, accepting common source/target aliases."""
    columns_by_lower = {col.lower().strip(): col for col in df.columns}

    alias_groups = {
        "source": ("source", "src"),
        "src": ("src", "source"),
        "target": ("target", "tgt", "trg"),
        "tgt": ("tgt", "target", "trg"),
        "trg": ("trg", "target", "tgt"),
    }

    if requested:
        requested_lower = requested.lower().strip()
        for candidate in alias_groups.get(requested_lower, (requested_lower,)):
            resolved = columns_by_lower.get(candidate)
            if resolved is not None:
                return resolved
        available = ", ".join(df.columns)
        raise ValueError(
            f"Could not find column {requested!r} in CSV. "
            f"Available columns: {available}"
        )

    # Prefer source-side text to avoid leakage from gold targets.
    for col_name in ("source", "src", "text", "sentence", "target", "tgt", "trg"):
        resolved = columns_by_lower.get(col_name)
        if resolved is not None:
            return resolved

    return df.columns[0]


def load_sentences_from_csv(csv_path: str, text_column: str = None) -> list:
    """Load sentences from CSV file."""
    import pandas as pd
    
    df = pd.read_csv(csv_path)
    
    # Find text column, accepting source/src and target/tgt/trg aliases.
    col = _resolve_text_column(df, text_column)
    
    print(f"  Using column: {col}")
    if col.lower() not in ("source", "src"):
        print("  WARNING: using a non-source column for recaser training")
        print("           this may not match the noisy input distribution")
    sentences = df[col].dropna().astype(str).tolist()
    return sentences


def load_sentences_from_txt(txt_path: str) -> list:
    """Load sentences from text file (one per line)."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Train Kneser-Ney LM Recaser")
    parser.add_argument("--lang", required=True, help="Language name")
    parser.add_argument("--input", required=True, help="Input CSV or TXT file with cased text")
    parser.add_argument("--output", default="tokenizers/recaser", help="Output directory")
    parser.add_argument(
        "--text-column",
        default=None,
        help="Column name for text (CSV only). Aliases like source/src and target/tgt/trg are accepted.",
    )
    
    # LM hyperparameters
    parser.add_argument("--order", type=int, default=3, help="N-gram order (default: 3 for trigram)")
    parser.add_argument("--discount", type=float, default=0.75, help="Kneser-Ney discount (default: 0.75)")
    parser.add_argument("--decode-beam-size", type=int, default=6,
                        help="Beam size for sentence-level recasing decode")
    parser.add_argument("--max-candidates", type=int, default=8,
                        help="Maximum casing variants to score per token")
    parser.add_argument("--lexical-weight", type=float, default=0.5,
                        help="Weight for casing-variant lexical probabilities")
    
    args = parser.parse_args()
    
    # Output directory
    output_dir = os.path.join(args.output, args.lang)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # Load data
    print(f"\nLoading data from {args.input}...")
    if args.input.endswith(".csv"):
        sentences = load_sentences_from_csv(args.input, args.text_column)
    else:
        sentences = load_sentences_from_txt(args.input)
    
    print(f"  Loaded {len(sentences):,} sentences")
    
    # Filter empty sentences
    sentences = [s for s in sentences if s.strip()]
    print(f"  After filtering: {len(sentences):,} sentences")
    
    # Train recaser
    print(f"\nTraining {args.order}-gram Kneser-Ney recaser...")
    recaser = train_kn_recaser(
        sentences,
        order=args.order,
        discount=args.discount,
        decode_beam_size=args.decode_beam_size,
        max_candidates_per_token=args.max_candidates,
        lexical_weight=args.lexical_weight,
    )
    
    # Save model
    print(f"\nSaving model to {output_dir}...")
    recaser.save(output_dir)
    
    # Test on a few examples
    print("\n" + "=" * 60)
    print("Testing recaser:")
    print("=" * 60)
    
    test_sentences = [
        "i went to new york yesterday .",
        "the president of the united states spoke today .",
        "my name is john and i live in london .",
        "apple released a new iphone this year .",
    ]
    
    for sent in test_sentences:
        recased = recaser.recase(sent)
        print(f"  Input:  {sent}")
        print(f"  Output: {recased}")
        print()
    
    print("Training complete!")
    print(f"Model saved to: {output_dir}")


if __name__ == "__main__":
    main()
