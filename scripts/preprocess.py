#!/usr/bin/env python3
"""
Preprocess CSV data into plain text files for GEC training.

Usage:
    python scripts/preprocess.py \\
        --lang english \\
        --input data/raw/english/train.csv \\
        --output data/processed/english/

Reads a CSV with 'source' and 'target' columns, performs:
  1. Unicode NFC normalisation
  2. Basic text cleaning (whitespace collapse)
  3. 90/5/5 train/valid/test split
     OR uses a separate test CSV if --test-input is given
  4. Writes: train.src, train.tgt, valid.src, valid.tgt, test.src, test.tgt
  5. Writes LCS/CRPF/error-type reports:
     train.quality.csv, valid.quality.csv, test.quality.csv,
     quality_summary.json
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher

from sacrebleu.metrics import CHRF


CHRFPP = CHRF(word_order=2)


def normalize_text(text: str) -> str:
    """Unicode NFC normalisation + lowercase + punctuation detachment.

    Lowercasing: ensures 'The' and 'the' map to the same BPE token.
    Punct detach: 'sure.' → 'sure .' so punctuation is its own token,
                  allowing the model to learn insertion/deletion of
                  punctuation independently of the surrounding word.
    Casing is restored at inference time by the LM recaser
    (scripts/train_recaser.py + scripts/generate.py).
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"([.,!?;:'\"\-])", r" \1 ", text)
    return " ".join(text.split())


def read_csv(filepath: str, preserve_case: bool = False) -> list:
    """Read CSV with 'source'/'src' and 'target'/'tgt'/'trg' columns.
    
    Args:
        filepath: Path to CSV file
        preserve_case: If True, return original cased text alongside normalized
                      Returns: [(norm_src, norm_tgt, orig_src, orig_tgt), ...]
                      If False, return only normalized: [(norm_src, norm_tgt), ...]
    """
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        src_col = tgt_col = None
        for fn in fieldnames:
            fn_l = fn.lower().strip()
            if fn_l in ("source", "src"):
                src_col = fn
            elif fn_l in ("target", "tgt", "trg"):
                tgt_col = fn

        if src_col is None or tgt_col is None:
            print(f"ERROR: Could not find source/target columns in {filepath}")
            print(f"  Found   : {fieldnames}")
            print(f"  Expected: 'source'/'src'  and  'target'/'tgt'/'trg'")
            sys.exit(1)

        print(f"  Using columns: src='{src_col}', tgt='{tgt_col}'")
        for row in reader:
            orig_src = unicodedata.normalize("NFC", row[src_col])
            orig_tgt = unicodedata.normalize("NFC", row[tgt_col])
            norm_src = normalize_text(row[src_col])
            norm_tgt = normalize_text(row[tgt_col])
            if norm_src and norm_tgt:
                if preserve_case:
                    pairs.append((norm_src, norm_tgt, orig_src, orig_tgt))
                else:
                    pairs.append((norm_src, norm_tgt))

    return pairs


def write_parallel(pairs: list, output_dir: str, prefix: str, save_original: bool = False) -> None:
    """Write source and target files.
    
    Args:
        pairs: List of (src, tgt) or (src, tgt, orig_src, orig_tgt) tuples
        output_dir: Output directory
        prefix: File prefix (train/valid/test)
        save_original: If True, also write {prefix}_original.src/tgt with cased text
    """
    src_path = os.path.join(output_dir, f"{prefix}.src")
    tgt_path = os.path.join(output_dir, f"{prefix}.tgt")
    
    with open(src_path, "w", encoding="utf-8") as f_src, \
         open(tgt_path, "w", encoding="utf-8") as f_tgt:
        for pair in pairs:
            src, tgt = pair[0], pair[1]
            f_src.write(src + "\n")
            f_tgt.write(tgt + "\n")
    
    print(f"  {prefix}: {len(pairs):,} pairs → {src_path}")
    
    # Save original cased versions if requested
    if save_original and len(pairs) > 0 and len(pairs[0]) == 4:
        orig_src_path = os.path.join(output_dir, f"{prefix}_original.src")
        orig_tgt_path = os.path.join(output_dir, f"{prefix}_original.tgt")
        with open(orig_src_path, "w", encoding="utf-8") as f_src, \
             open(orig_tgt_path, "w", encoding="utf-8") as f_tgt:
            for pair in pairs:
                orig_src, orig_tgt = pair[2], pair[3]
                f_src.write(orig_src + "\n")
                f_tgt.write(orig_tgt + "\n")
        print(f"        + original cased: {orig_tgt_path}")


def _safe_div(num: float, den: float) -> float:
    """Return 0.0 for empty denominators."""
    return num / den if den else 0.0


def _lcs_length(left_tokens: list, right_tokens: list) -> int:
    """Return exact longest common subsequence length using O(min(n, m)) space."""
    if len(left_tokens) < len(right_tokens):
        short_tokens, long_tokens = left_tokens, right_tokens
    else:
        short_tokens, long_tokens = right_tokens, left_tokens

    previous = [0] * (len(short_tokens) + 1)
    for long_token in long_tokens:
        current = [0]
        for idx, short_token in enumerate(short_tokens, start=1):
            if long_token == short_token:
                current.append(previous[idx - 1] + 1)
            else:
                current.append(max(previous[idx], current[-1]))
        previous = current
    return previous[-1]


def sentence_quality(src: str, tgt: str) -> dict:
    """Compute LCS-based quality/error statistics for one source-target pair.

    CRPF columns are:
      C = correction_ratio: share of tokens touched by an edit
      R = recall-like LCS coverage over target tokens
      P = precision-like LCS coverage over source tokens
      F = F0.5 over the LCS-based P/R values
    """
    src_tokens = src.split()
    tgt_tokens = tgt.split()
    matcher = SequenceMatcher(None, src_tokens, tgt_tokens, autojunk=False)

    lcs_len = _lcs_length(src_tokens, tgt_tokens)
    insertions = deletions = substitutions = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        src_span = i2 - i1
        tgt_span = j2 - j1
        if tag == "insert":
            insertions += tgt_span
        elif tag == "delete":
            deletions += src_span
        elif tag == "replace":
            substitutions += min(src_span, tgt_span)
            if src_span > tgt_span:
                deletions += src_span - tgt_span
            elif tgt_span > src_span:
                insertions += tgt_span - src_span

    edit_tokens = insertions + deletions + substitutions
    src_len = len(src_tokens)
    tgt_len = len(tgt_tokens)
    precision = _safe_div(lcs_len, src_len)
    recall = _safe_div(lcs_len, tgt_len)
    beta = 0.5
    f05 = _safe_div((1 + beta**2) * precision * recall,
                    beta**2 * precision + recall)
    chrfpp = CHRFPP.sentence_score(tgt, [src]).score

    if edit_tokens == 0:
        dominant_error_type = "none"
    else:
        error_counts = {
            "substitution": substitutions,
            "insertion": insertions,
            "deletion": deletions,
        }
        dominant_error_type = max(error_counts, key=error_counts.get)

    return {
        "source_tokens": src_len,
        "target_tokens": tgt_len,
        "lcs_tokens": lcs_len,
        "lcs_ratio_source": precision,
        "lcs_ratio_target": recall,
        "correction_ratio": _safe_div(edit_tokens, max(src_len, tgt_len)),
        "precision": precision,
        "recall": recall,
        "f0_5": f05,
        "chrf_pp": chrfpp,
        "edit_tokens": edit_tokens,
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "dominant_error_type": dominant_error_type,
    }


def write_quality_report(pairs: list, output_dir: str, prefix: str) -> dict:
    """Write per-sentence LCS/CRPF/error-type stats and return averages."""
    metrics_path = os.path.join(output_dir, f"{prefix}.quality.csv")
    rows = []
    error_type_counts = Counter()

    for idx, pair in enumerate(pairs, start=1):
        src, tgt = pair[0], pair[1]
        metrics = sentence_quality(src, tgt)
        error_type_counts[metrics["dominant_error_type"]] += 1
        rows.append({
            "sentence_id": idx,
            "source": src,
            "target": tgt,
            **metrics,
        })

    fieldnames = [
        "sentence_id", "source", "target",
        "source_tokens", "target_tokens", "lcs_tokens",
        "lcs_ratio_source", "lcs_ratio_target",
        "correction_ratio", "precision", "recall", "f0_5", "chrf_pp",
        "edit_tokens", "insertions", "deletions", "substitutions",
        "dominant_error_type",
    ]
    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    numeric_fields = [
        "source_tokens", "target_tokens", "lcs_tokens",
        "lcs_ratio_source", "lcs_ratio_target",
        "correction_ratio", "precision", "recall", "f0_5", "chrf_pp",
        "edit_tokens", "insertions", "deletions", "substitutions",
    ]
    averages = {
        field: _safe_div(sum(row[field] for row in rows), len(rows))
        for field in numeric_fields
    }
    summary = {
        "sentences": len(rows),
        "averages": averages,
        "error_type_counts": dict(error_type_counts),
        "report_path": metrics_path,
    }
    print(f"        + quality report: {metrics_path}")
    return summary


def write_quality_summary(summaries: dict, output_dir: str) -> None:
    """Write split-level average CRPF/LCS/error-type stats."""
    summary_path = os.path.join(output_dir, "quality_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"  Quality summary → {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess CSV data for GEC")
    parser.add_argument("--lang",        required=True, help="Language name")
    parser.add_argument("--input",       required=True, help="Training CSV path")
    parser.add_argument("--test-input",  default=None,
                        help="Optional: separate held-out test CSV")
    parser.add_argument("--output",      required=True,
                        help="Output directory for processed files")
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--max-samples", type=int,   default=None,
                        help="Cap total pairs after shuffle (e.g. 300000)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    random.seed(args.seed)

    print(f"=== Preprocessing {args.lang} ===")
    print(f"  Input: {args.input}")

    pairs = read_csv(args.input)
    print(f"  Total pairs read: {len(pairs):,}")

    random.shuffle(pairs)

    if args.max_samples and len(pairs) > args.max_samples:
        pairs = pairs[:args.max_samples]
        print(f"  Capped to {args.max_samples:,} pairs")

    if args.test_input:
        # Separate test file provided — split remaining data into train+valid
        # BUG FIX: was valid_pairs = pairs[n_train:] which took ALL remaining
        # data (100% - train_ratio).  Now we correctly cap valid at valid_ratio.
        
        # Read test data with original casing preserved
        test_pairs = read_csv(args.test_input, preserve_case=True)

        total_ratio = args.train_ratio + args.valid_ratio
        if total_ratio == 0:
            raise ValueError("train_ratio and valid_ratio cannot both be zero")
        
        n_train = int(len(pairs) * (args.train_ratio / total_ratio))
        n_valid = int(len(pairs) * (args.valid_ratio / total_ratio))

        train_pairs = pairs[:n_train]
        valid_pairs = pairs[n_train: n_train + n_valid]
        # Any tiny remainder (due to int rounding) is silently dropped
    else:
        # No separate test file - need to preserve casing for test split
        pairs_with_case = read_csv(args.input, preserve_case=True)
        n_train = int(len(pairs_with_case) * args.train_ratio)
        n_valid = int(len(pairs_with_case) * args.valid_ratio)
        
        # Train and valid don't need original casing (only normalized)
        train_pairs = [(p[0], p[1]) for p in pairs_with_case[:n_train]]
        valid_pairs = [(p[0], p[1]) for p in pairs_with_case[n_train: n_train + n_valid]]
        # Test keeps both normalized and original
        test_pairs  = pairs_with_case[n_train + n_valid:]

    write_parallel(train_pairs, args.output, "train", save_original=False)
    train_quality = write_quality_report(train_pairs, args.output, "train")
    write_parallel(valid_pairs, args.output, "valid", save_original=False)
    valid_quality = write_quality_report(valid_pairs, args.output, "valid")
    write_parallel(test_pairs,  args.output, "test",  save_original=True)
    test_quality = write_quality_report(test_pairs, args.output, "test")
    write_quality_summary({
        "train": train_quality,
        "valid": valid_quality,
        "test": test_quality,
    }, args.output)

    print(f"\n  Summary for {args.lang}:")
    print(f"    Train : {len(train_pairs):,}")
    print(f"    Valid : {len(valid_pairs):,}")
    print(f"    Test  : {len(test_pairs):,}")
    print("    Avg CRPF/LCS saved in quality_summary.json")
    print("Done!")


if __name__ == "__main__":
    main()
