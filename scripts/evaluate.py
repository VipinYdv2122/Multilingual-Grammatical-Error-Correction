#!/usr/bin/env python3
"""
Evaluate GEC model outputs — PolyGEC v2.

Metrics (matching project outline):
  1. GLEU            — Napoles et al. GEC-specific source-aware metric
  2. Corpus BLEU     — overall translation quality                (sacrebleu)
  3. Token Accuracy  — exact position-aligned token match
  4. P / R / F0.5    — ERRANT edit precision/recall
  5. Efficiency      — latency (ms/sentence), model size (MB), param count
                       (requires --checkpoint)

Usage — single experiment:
    python scripts/evaluate.py \\
        --hyp  results/english_lstm_H2H.hyp \\
        --ref  results/english_lstm_H2H.ref \\
        --src  results/english_lstm_H2H.src \\
        --checkpoint checkpoints/english/lstm_H2H/best.pt

Usage — evaluate ALL experiments:
    python scripts/evaluate.py --run-all --lang english

Results are auto-saved to results/<lang>_<timestamp>.{json,txt}

GLEU note:
    This script reports a source-aware GEC GLEU score following Napoles et al.
    It rewards hypothesis/reference n-gram overlap and penalizes hypothesis
    n-grams copied from the source when the reference changed them. Source
    sentences are required for GLEU.
"""

import argparse
import datetime
import csv
import json
import os
import sys
import time
from collections import Counter
from difflib import SequenceMatcher

import sacrebleu
from sacrebleu.metrics import CHRF
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from scripts.recaser_utils import detokenize_text as detokenize_project_text
except ModuleNotFoundError:
    from recaser_utils import detokenize_text as detokenize_project_text


CHRFPP = CHRF(word_order=2)


# ─── GLEU (Napoles et al. GEC metric) ────────────────────────────────────────
def _load_gec_metric(metric_id: str):
    """Load a gec-metrics scorer with a clear install error."""
    try:
        from gec_metrics import get_metric
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'gec-metrics'. Install requirements.txt before "
            "running official GLEU/ERRANT evaluation."
        ) from exc

    metric_cls = get_metric(metric_id)
    return metric_cls(metric_cls.Config())


def _as_percent(score: float) -> float:
    """gec-metrics returns scores in [0, 1]; keep [0, 100] if already scaled."""
    return score * 100 if score is not None and score <= 1.0 else score


def compute_gleu(src_sents: list, hyp_sents: list, ref_sents: list) -> float:
    """
    Official-style Napoles GEC GLEU via gec-metrics GLEUOfficial.

    Args:
        src_sents : list of source sentences (pre-correction)
        hyp_sents : list of model hypothesis sentences
        ref_sents : list of reference / gold-correction sentences

    Returns:
        GLEU score in [0, 100]
    """
    metric = _load_gec_metric("gleuofficial")
    score = metric.score_corpus(
        sources=src_sents,
        hypotheses=hyp_sents,
        references=[ref_sents],
    )
    return _as_percent(score)


# ─── Corpus BLEU ──────────────────────────────────────────────────────────────
def compute_corpus_bleu(hyp_sents: list, ref_sents: list) -> float:
    """Corpus BLEU via sacrebleu (standard, reproducible)."""
    return sacrebleu.corpus_bleu(hyp_sents, [ref_sents]).score


# ─── Token Accuracy ───────────────────────────────────────────────────────────
def compute_token_accuracy(hyp_sents: list, ref_sents: list) -> float:
    """
    Position-aligned token accuracy.
    Denominator includes extra/missing tokens in either sequence.
    """
    total = correct = 0
    for hyp, ref in tqdm(
        zip(hyp_sents, ref_sents),
        total=len(hyp_sents), desc="TokenAcc", leave=False
    ):
        h = hyp.split()
        r = ref.split()
        total   += max(len(h), len(r))
        correct += sum(a == b for a, b in zip(h, r))
    return correct / total * 100 if total else 0.0


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


def _sequence_edit_stats(left_text: str, right_text: str) -> dict:
    """Compute LCS ratios and edit-type counts for one token sequence pair."""
    left_tokens = left_text.split()
    right_tokens = right_text.split()
    matcher = SequenceMatcher(None, left_tokens, right_tokens, autojunk=False)

    lcs_len = _lcs_length(left_tokens, right_tokens)
    insertions = deletions = substitutions = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        left_span = i2 - i1
        right_span = j2 - j1
        if tag == "insert":
            insertions += right_span
        elif tag == "delete":
            deletions += left_span
        elif tag == "replace":
            substitutions += min(left_span, right_span)
            if left_span > right_span:
                deletions += left_span - right_span
            elif right_span > left_span:
                insertions += right_span - left_span

    edit_tokens = insertions + deletions + substitutions
    left_len = len(left_tokens)
    right_len = len(right_tokens)
    lcs_precision = _safe_div(lcs_len, left_len)
    lcs_recall = _safe_div(lcs_len, right_len)
    beta = 0.5
    lcs_f05 = _safe_div((1 + beta**2) * lcs_precision * lcs_recall,
                        beta**2 * lcs_precision + lcs_recall)
    chrfpp = CHRFPP.sentence_score(right_text, [left_text]).score

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
        "left_tokens": left_len,
        "right_tokens": right_len,
        "lcs_tokens": lcs_len,
        "lcs_ratio_left": lcs_precision,
        "lcs_ratio_right": lcs_recall,
        "correction_ratio": _safe_div(edit_tokens, max(left_len, right_len)),
        "lcs_precision": lcs_precision,
        "lcs_recall": lcs_recall,
        "lcs_f0_5": lcs_f05,
        "chrf_pp": chrfpp,
        "edit_tokens": edit_tokens,
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
        "dominant_error_type": dominant_error_type,
    }


def _write_sentence_report(
    hyp_path: str,
    hyp_sents: list,
    ref_sents: list,
    src_sents: list = None,
) -> dict:
    """Write per-sentence analysis for model outputs and return summary stats."""
    stem = os.path.splitext(hyp_path)[0]
    report_path = f"{stem}.sentence_metrics.csv"

    rows = []
    behavior_counts = Counter()
    pred_error_counts = Counter()

    for idx, (hyp, ref) in enumerate(zip(hyp_sents, ref_sents), start=1):
        pred_stats = _sequence_edit_stats(hyp, ref)
        pred_error_counts[pred_stats["dominant_error_type"]] += 1

        row = {
            "sentence_id": idx,
            "hypothesis": hyp,
            "reference": ref,
            "prediction_left_tokens": pred_stats["left_tokens"],
            "prediction_right_tokens": pred_stats["right_tokens"],
            "prediction_lcs_tokens": pred_stats["lcs_tokens"],
            "prediction_lcs_ratio_hyp": pred_stats["lcs_ratio_left"],
            "prediction_lcs_ratio_ref": pred_stats["lcs_ratio_right"],
            "prediction_correction_ratio": pred_stats["correction_ratio"],
            "prediction_lcs_precision": pred_stats["lcs_precision"],
            "prediction_lcs_recall": pred_stats["lcs_recall"],
            "prediction_lcs_f0_5": pred_stats["lcs_f0_5"],
            "prediction_chrf_pp": pred_stats["chrf_pp"],
            "prediction_edit_tokens": pred_stats["edit_tokens"],
            "prediction_insertions": pred_stats["insertions"],
            "prediction_deletions": pred_stats["deletions"],
            "prediction_substitutions": pred_stats["substitutions"],
            "prediction_dominant_error_type": pred_stats["dominant_error_type"],
        }

        if src_sents is not None:
            src = src_sents[idx - 1]
            src_ref_stats = _sequence_edit_stats(src, ref)
            src_hyp_stats = _sequence_edit_stats(src, hyp)
            src_equals_hyp = src == hyp
            src_equals_ref = src == ref
            hyp_equals_ref = hyp == ref

            if hyp_equals_ref and src_equals_hyp:
                behavior = "already_correct"
            elif hyp_equals_ref:
                behavior = "correct_edit"
            elif src_equals_hyp and not src_equals_ref:
                behavior = "missed_correction"
            elif not src_equals_hyp and src_equals_ref:
                behavior = "unnecessary_change"
            elif src_hyp_stats["edit_tokens"] > src_ref_stats["edit_tokens"]:
                behavior = "overcorrection"
            else:
                behavior = "partial_correction"

            behavior_counts[behavior] += 1
            row.update({
                "source": src,
                "gold_edit_tokens": src_ref_stats["edit_tokens"],
                "gold_insertions": src_ref_stats["insertions"],
                "gold_deletions": src_ref_stats["deletions"],
                "gold_substitutions": src_ref_stats["substitutions"],
                "predicted_edit_tokens_from_source": src_hyp_stats["edit_tokens"],
                "predicted_insertions_from_source": src_hyp_stats["insertions"],
                "predicted_deletions_from_source": src_hyp_stats["deletions"],
                "predicted_substitutions_from_source": src_hyp_stats["substitutions"],
                "source_reference_lcs_tokens": src_ref_stats["lcs_tokens"],
                "source_hypothesis_lcs_tokens": src_hyp_stats["lcs_tokens"],
                "model_behavior": behavior,
            })

        rows.append(row)

    fieldnames = [
        "sentence_id",
        "source",
        "hypothesis",
        "reference",
        "gold_edit_tokens",
        "gold_insertions",
        "gold_deletions",
        "gold_substitutions",
        "predicted_edit_tokens_from_source",
        "predicted_insertions_from_source",
        "predicted_deletions_from_source",
        "predicted_substitutions_from_source",
        "source_reference_lcs_tokens",
        "source_hypothesis_lcs_tokens",
        "prediction_left_tokens",
        "prediction_right_tokens",
        "prediction_lcs_tokens",
        "prediction_lcs_ratio_hyp",
        "prediction_lcs_ratio_ref",
        "prediction_correction_ratio",
        "prediction_lcs_precision",
        "prediction_lcs_recall",
        "prediction_lcs_f0_5",
        "prediction_chrf_pp",
        "prediction_edit_tokens",
        "prediction_insertions",
        "prediction_deletions",
        "prediction_substitutions",
        "prediction_dominant_error_type",
        "model_behavior",
    ]
    if rows:
        fieldnames = [name for name in fieldnames if any(name in row for row in rows)]

    with open(report_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    numeric_fields = [
        "prediction_lcs_tokens",
        "prediction_lcs_ratio_hyp",
        "prediction_lcs_ratio_ref",
        "prediction_correction_ratio",
        "prediction_lcs_precision",
        "prediction_lcs_recall",
        "prediction_lcs_f0_5",
        "prediction_chrf_pp",
        "prediction_edit_tokens",
        "prediction_insertions",
        "prediction_deletions",
        "prediction_substitutions",
    ]
    if src_sents is not None:
        numeric_fields.extend([
            "gold_edit_tokens",
            "gold_insertions",
            "gold_deletions",
            "gold_substitutions",
            "predicted_edit_tokens_from_source",
            "predicted_insertions_from_source",
            "predicted_deletions_from_source",
            "predicted_substitutions_from_source",
            "source_reference_lcs_tokens",
            "source_hypothesis_lcs_tokens",
        ])

    averages = {
        field: _safe_div(sum(row.get(field, 0) for row in rows), len(rows))
        for field in numeric_fields
    }
    summary = {
        "sentence_report": report_path,
        "sentences": len(rows),
        "averages": averages,
        "prediction_error_type_counts": dict(pred_error_counts),
    }
    if src_sents is not None:
        summary["model_behavior_counts"] = dict(behavior_counts)

    print(f"  Sentence report        : {report_path}")
    return summary


def compute_prf05(hyp_sents: list, ref_sents: list, src_sents: list = None):
    """
    Corpus-level ERRANT Precision, Recall, F0.5.
    F0.5 weights precision twice as heavily as recall — standard for GEC.

    Uses gec-metrics ERRANT, which wraps the ERRANT aligner/scorer instead of
    the earlier SequenceMatcher/LCS approximation.

    Returns (precision, recall, f05, mode), with scores in [0, 100].
    """
    if src_sents is None:
        return None, None, None, "errant_unavailable"

    metric = _load_gec_metric("errant")
    verbose = metric.score_corpus_verbose(
        sources=src_sents,
        hypotheses=hyp_sents,
        references=[ref_sents],
    )

    values = {}
    for line in str(verbose).splitlines():
        for part in line.replace(",", " ").split():
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            try:
                values[key.lower()] = float(value)
            except ValueError:
                pass

    precision = values.get("prec")
    recall = values.get("rec")
    f05 = values.get("f-0.5", values.get("f0.5"))

    if precision is None or recall is None or f05 is None:
        raise RuntimeError(f"Could not parse ERRANT verbose output: {verbose}")

    return _as_percent(precision), _as_percent(recall), _as_percent(f05), "errant"


# ─── Efficiency metrics ───────────────────────────────────────────────────────
def compute_efficiency(ckpt_path: str, max_len: int = 100, n_sentences: int = 200):
    """
    Measure inference latency (ms/sentence), model size (MB), parameter count.
    Loads checkpoint fresh so evaluate.py stays independent of training state.
    """
    try:
        import torch
        from models.dataset  import Vocabulary, SOS_IDX, EOS_IDX
        # Handle both direct execution and module import
        try:
            from scripts.generate import load_model
        except ModuleNotFoundError:
            from generate import load_model

        # Infer arch from checkpoint directory name: <lang>_<arch>_<config>/best.pt
        dir_name = os.path.basename(os.path.dirname(ckpt_path))
        parts    = dir_name.split("_")
        if len(parts) > 0 and parts[0] in ("rnn", "lstm", "transformer"):
            arch = parts[0]
        else:
            arch = "lstm"  # default fallback

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, src_vocab, _ = load_model(arch, ckpt_path, device)
        model.eval()

        dummy_len = max_len // 2
        src     = torch.randint(3, max(len(src_vocab) - 1, 4),
                                (n_sentences, dummy_len), dtype=torch.long).to(device)
        src_len = torch.full((n_sentences,), dummy_len, dtype=torch.long).to(device)

        # Warm-up pass
        with torch.no_grad():
            model.generate(src[:4], src_len[:4], SOS_IDX, EOS_IDX, max_len)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(src, src_len, SOS_IDX, EOS_IDX, max_len)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        param_count = sum(p.numel() for p in model.parameters())
        size_mb     = param_count * 4 / (1024 ** 2)   # float32 = 4 bytes
        latency_ms  = elapsed_ms / n_sentences
        return latency_ms, size_mb, param_count

    except Exception as e:
        print(f"  ⚠  Efficiency measurement failed: {e}")
        return None, None, None


# ─── Core evaluation function ─────────────────────────────────────────────────
def evaluate_files(
    hyp_path: str,
    ref_path: str,
    src_path: str = None,
    ckpt_path: str = None,
    quiet: bool = False,
) -> dict:
    """Compute all metrics for one (hyp, ref, src) triple."""
    with open(hyp_path, encoding="utf-8") as f:
        hyp_sents = [detokenize_project_text(ln.strip()) for ln in f]
    with open(ref_path, encoding="utf-8") as f:
        ref_sents = [detokenize_project_text(ln.strip()) for ln in f]

    if len(hyp_sents) != len(ref_sents):
        n = min(len(hyp_sents), len(ref_sents))
        print(f"  WARNING: length mismatch ({len(hyp_sents)} vs {len(ref_sents)})"
              f" — truncating to {n}", file=sys.stderr)
        hyp_sents, ref_sents = hyp_sents[:n], ref_sents[:n]

    src_sents = None
    if src_path and os.path.exists(src_path):
        with open(src_path, encoding="utf-8") as f:
            src_sents = [detokenize_project_text(ln.strip()) for ln in f]
        if len(src_sents) != len(hyp_sents):
            n = min(len(src_sents), len(hyp_sents))
            print(f"  WARNING: source length mismatch ({len(src_sents)} vs {len(hyp_sents)})"
                  f" — truncating all files to {n}", file=sys.stderr)
            hyp_sents, ref_sents, src_sents = hyp_sents[:n], ref_sents[:n], src_sents[:n]

    sentence_report = _write_sentence_report(
        hyp_path=hyp_path,
        hyp_sents=hyp_sents,
        ref_sents=ref_sents,
        src_sents=src_sents,
    )

    if not quiet:
        print(f"\n  Sentences : {len(hyp_sents):,}")

    bleu    = compute_corpus_bleu(hyp_sents, ref_sents)
    tok_acc = compute_token_accuracy(hyp_sents, ref_sents)
    gleu    = compute_gleu(src_sents, hyp_sents, ref_sents) if src_sents is not None else None
    prec, rec, f05, prf05_mode = compute_prf05(hyp_sents, ref_sents, src_sents)

    latency_ms = size_mb = param_count = None
    if ckpt_path and os.path.exists(ckpt_path):
        latency_ms, size_mb, param_count = compute_efficiency(ckpt_path)

    if not quiet:
        W = 55
        print(f"\n  {'='*W}")
        if gleu is not None:
            print(f"  GLEU (Napoles GEC)   : {gleu:7.2f}")
        else:
            print("  GLEU (Napoles GEC)   :     n/a  (--src required)")
        print(f"  Corpus BLEU          : {bleu:7.2f}")
        print(f"  Token Accuracy       : {tok_acc:7.2f} %")
        print(f"  {'─'*W}")
        if prf05_mode == "errant":
            print(f"  Precision (ERRANT)   : {prec:7.2f} %")
            print(f"  Recall    (ERRANT)   : {rec:7.2f} %")
            print(f"  F0.5      (ERRANT)   : {f05:7.2f} %")
        else:
            print("  Precision (ERRANT)   :     n/a  (--src required)")
            print("  Recall    (ERRANT)   :     n/a  (--src required)")
            print("  F0.5      (ERRANT)   :     n/a  (--src required)")
        if latency_ms is not None:
            print(f"  {'─'*W}")
            print(f"  Latency              : {latency_ms:7.2f} ms/sentence")
            print(f"  Model size           : {size_mb:7.1f} MB")
            print(f"  Parameters           : {param_count:>15,}")
        print(f"  {'='*W}")

        # Sample predictions
        if src_sents:
            print("\n  Sample predictions:")
            print(f"  {'─'*70}")
            for s, h, r in zip(src_sents[:3], hyp_sents[:3], ref_sents[:3]):
                print(f"  SRC : {s}")
                print(f"  HYP : {h}")
                print(f"  REF : {r}")
                print(f"  {'─'*70}")

    return {
        "gleu":       gleu,
        "bleu":       bleu,
        "token_acc":  tok_acc,
        "precision":  prec,
        "recall":     rec,
        "f0.5":       f05,
        "prf0.5_mode": prf05_mode,
        "latency_ms": latency_ms,
        "size_mb":    size_mb,
        "params":     param_count,
        "sentence_report": sentence_report["sentence_report"],
        "sentence_report_summary": sentence_report,
    }


# ─── Auto-save results ────────────────────────────────────────────────────────
def _save_results(all_results: dict, lang_tag: str, test_info: str):
    """Save results to results/<tag>_<timestamp>.{json,txt}."""
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{lang_tag}_{ts}"

    json_path = os.path.join(results_dir, f"{stem}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    txt_path = os.path.join(results_dir, f"{stem}.txt")
    W   = 112
    hdr = (f"  {'Experiment':<38} {'GLEU':>6}  {'BLEU':>6}  {'TokAcc':>7}"
           f"  {'Prec':>7}  {'Rec':>7}  {'F0.5':>7}"
           f"  {'Lat(ms)':>8}  {'MB':>6}  {'Params':>12}")
    sep = (f"  {'─'*38}  {'─'*6}  {'─'*6}  {'─'*7}"
           f"  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*8}  {'─'*6}  {'─'*12}")
    lines = [
        "=" * W,
        f"  PolyGEC Evaluation — {datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
        f"  Source: {test_info}",
        "=" * W, hdr, sep,
    ]

    def _fmt(v, fmt):
        return format(v, fmt) if v is not None else "  n/a"

    for name, m in sorted(all_results.items()):
        if "error" in m:
            lines.append(f"  {name:<38}  ERROR: {m['error']}")
        else:
            lines.append(
                f"  {name:<38}"
                f"  {_fmt(m.get('gleu'),       '6.2f')}"
                f"  {_fmt(m.get('bleu'),       '6.2f')}"
                f"  {_fmt(m.get('token_acc'),  '6.2f')}%"
                f"  {_fmt(m.get('precision'),  '6.2f')}%"
                f"  {_fmt(m.get('recall'),     '6.2f')}%"
                f"  {_fmt(m.get('f0.5'),       '6.2f')}%"
                f"  {_fmt(m.get('latency_ms'), '8.2f')}"
                f"  {_fmt(m.get('size_mb'),    '6.1f')}"
                f"  {str(m.get('params') or 'n/a'):>12}"
            )
    lines.append("=" * W)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n  ✓ Results saved:")
    print(f"      JSON : {json_path}")
    print(f"      TXT  : {txt_path}")
    return json_path, txt_path


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Evaluate GEC model outputs — PolyGEC v2")

    # Single-run mode
    p.add_argument("--hyp",        default=None,
                   help="Hypothesis file (one sentence per line)")
    p.add_argument("--ref",        default=None,
                   help="Reference file (one sentence per line)")
    p.add_argument("--src",        default=None,
                   help="Source file (required for GLEU and ERRANT P/R/F0.5)")
    p.add_argument("--checkpoint", default=None,
                   help="Checkpoint path for efficiency metrics")
    p.add_argument("--output",     default=None,
                   help="Write tab-separated metrics to this file")

    # Run-all mode
    p.add_argument("--run-all",     action="store_true",
                   help="Evaluate all .hyp files in --results-dir")
    p.add_argument("--lang",        default=None,
                   help="Language filter for --run-all (e.g. english)")
    p.add_argument("--arch",        default=None,
                   choices=["rnn", "lstm", "transformer"],
                   help="Architecture filter for --run-all")
    p.add_argument("--config",      default=None,
                   help="Tokenization config filter for --run-all (e.g. 0.5K_1K)")
    p.add_argument("--results-dir", default="results",
                   help="Directory containing .hyp/.ref/.src files")
    p.add_argument("--ckpt-dir",    default="checkpoints",
                   help="Checkpoints directory (for --run-all efficiency metrics)")

    args = p.parse_args()
    all_results = {}

    if args.run_all:
        hyp_files = sorted(
            f for f in os.listdir(args.results_dir) if f.endswith(".hyp"))

        if args.lang:
            hyp_files = [f for f in hyp_files if f.startswith(f"{args.lang}_")]
        if args.arch:
            hyp_files = [f for f in hyp_files if f"_{args.arch}_" in f]
        if args.config:
            hyp_files = [f for f in hyp_files if f.endswith(f"_{args.config}.hyp")]

        if not hyp_files:
            print("No .hyp files matched the given filters.")
            print(f"Available: {os.listdir(args.results_dir)}")
            sys.exit(1)

        print(f"\nFound {len(hyp_files)} experiment(s):")
        for f in hyp_files:
            print(f"  • {f}")

        for fname in hyp_files:
            stem     = fname[:-4]
            hyp_path = os.path.join(args.results_dir, f"{stem}.hyp")
            ref_path = os.path.join(args.results_dir, f"{stem}.ref")
            src_path = os.path.join(args.results_dir, f"{stem}.src")

            # Infer checkpoint path from <lang>_<arch>_<config>.hyp naming
            parts     = stem.split("_")
            ckpt_path = None
            if len(parts) >= 3:
                candidate = os.path.join(
                    args.ckpt_dir, parts[0], f"{parts[1]}_{parts[2]}", "best.pt")
                if os.path.exists(candidate):
                    ckpt_path = candidate

            print(f"\n{'━'*60}\n  Evaluating: {stem}\n{'━'*60}")
            try:
                metrics = evaluate_files(
                    hyp_path=hyp_path,
                    ref_path=ref_path,
                    src_path=src_path if os.path.exists(src_path) else None,
                    ckpt_path=ckpt_path,
                )
                all_results[stem] = metrics
            except Exception as e:
                print(f"  ✗ Failed: {e}")
                all_results[stem] = {"error": str(e)}

        # Summary table
        W   = 112
        hdr = (f"  {'Experiment':<38} {'GLEU':>6}  {'BLEU':>6}  {'TokAcc':>7}"
               f"  {'Prec':>7}  {'Rec':>7}  {'F0.5':>7}")
        sep = (f"  {'─'*38}  {'─'*6}  {'─'*6}  {'─'*7}"
               f"  {'─'*7}  {'─'*7}  {'─'*7}")
        print(f"\n{'='*W}\n  EVALUATION SUMMARY\n{'='*W}")
        print(hdr)
        print(sep)

        def _summary_fmt(value):
            return f"{value:>6.2f}" if value is not None else f"{'n/a':>6}"

        for name, m in sorted(all_results.items()):
            if "error" in m:
                print(f"  {name:<38}  ERROR: {m['error']}")
            else:
                print(
                    f"  {name:<38}"
                    f"  {_summary_fmt(m.get('gleu'))}"
                    f"  {_summary_fmt(m.get('bleu'))}"
                    f"  {_summary_fmt(m.get('token_acc'))}%"
                    f"  {_summary_fmt(m.get('precision'))}%"
                    f"  {_summary_fmt(m.get('recall'))}%"
                    f"  {_summary_fmt(m.get('f0.5'))}%")
        print(f"{'='*W}")

        _save_results(all_results,
                      lang_tag=args.lang or "all",
                      test_info=args.results_dir)

    else:
        if not args.hyp or not args.ref:
            p.error("Provide --hyp and --ref, or use --run-all")

        exp_name = os.path.basename(args.hyp).replace(".hyp", "")
        print(f"\n{'='*55}\n  RESULTS  [{exp_name}]\n{'='*55}")

        metrics = evaluate_files(
            hyp_path=args.hyp,
            ref_path=args.ref,
            src_path=args.src,
            ckpt_path=args.checkpoint,
        )
        all_results[exp_name] = metrics

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                for k, v in metrics.items():
                    f.write(f"{k}\t{v}\n")
            print(f"\n  Metrics saved → {args.output}")

        _save_results(all_results,
                      lang_tag=exp_name.split("_")[0] if "_" in exp_name else "eval",
                      test_info=args.hyp)


if __name__ == "__main__":
    main()
