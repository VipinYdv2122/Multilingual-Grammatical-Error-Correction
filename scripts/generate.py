#!/usr/bin/env python3
"""
Generate GEC corrections using a trained model — PolyGEC v2.

Usage:
    python scripts/generate.py --lang english --arch transformer --config 16K_16K
    python scripts/generate.py --lang english --arch transformer --config 0.5K_1K --beam-size 4

Output:
    results/<lang>_<arch>_<config>.hyp   hypothesis sentences (de-BPE'd)
    results/<lang>_<arch>_<config>.ref   reference sentences  (de-BPE'd)
    results/<lang>_<arch>_<config>.src   source sentences     (de-BPE'd)

Alignment guarantee:
    Every line in test.src produces exactly one output line in .hyp/.ref/.src.
    Sentences longer than --max-len are truncated (not dropped), so line N of
    the hypothesis file always corresponds to line N of the source test file.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm.auto import tqdm
from models.rnn_attention  import RNNSeq2Seq
from models.lstm_attention import LSTMSeq2Seq
from models.transformer    import TransformerModel
from models.dataset        import (Vocabulary, GECDataset, collate_fn,
                                   PAD_IDX, SOS_IDX, EOS_IDX)
from torch.utils.data import DataLoader

# Import recaser utilities
from scripts.recaser_utils import (
    load_recaser,
    apply_recaser_batch,
    recaser_exists,
    detokenize_text,
)


# ─── BPE detokenisation ───────────────────────────────────────────────────────
def debpe(text: str) -> str:
    """
    Remove BPE continuation markers.
      "hel@@ lo"  → "hello"   (mid-word marker: replace "@@ ")
      "test@@"    → "test"    (end-of-string marker: replace "@@")
    Both replacements are required.
    """
    return text.replace("@@ ", "").replace("@@", "")


# ─── Vocab path resolver ─────────────────────────────────────────────────────
def _resolve_vocab_path(saved_path: str, ckpt_path: str) -> str:
    """
    Checkpoints store absolute vocab paths.  Resolve portably:
      1. Try saved absolute path.
      2. Try <ckpt_dir>/<filename>.
      3. Try <cwd>/<filename>.
    """
    if os.path.exists(saved_path):
        return saved_path
    for base in (os.path.dirname(os.path.abspath(ckpt_path)), os.getcwd()):
        candidate = os.path.join(base, os.path.basename(saved_path))
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Cannot locate vocab: {saved_path!r}\n"
        f"  Hint: run from the project root, or pass --checkpoint with full path.")


# ─── Checkpoint loader ───────────────────────────────────────────────────────
def load_model(arch: str, checkpoint_path: str, device):
    """Load a trained model from checkpoint."""
    ckpt  = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = argparse.Namespace(**ckpt["args"])

    src_vocab = Vocabulary().load(
        _resolve_vocab_path(ckpt["src_vocab_path"], checkpoint_path))
    tgt_vocab = Vocabulary().load(
        _resolve_vocab_path(ckpt["tgt_vocab_path"], checkpoint_path))

    common = dict(embed_dim=saved.embed_dim, dropout=0.0,
                  sos_idx=SOS_IDX, eos_idx=EOS_IDX, pad_idx=PAD_IDX)

    if arch == "rnn":
        model = RNNSeq2Seq(
            len(src_vocab), len(tgt_vocab),
            hidden_dim=saved.hidden_dim, enc_layers=saved.n_layers, **common)
    elif arch == "lstm":
        model = LSTMSeq2Seq(
            len(src_vocab), len(tgt_vocab),
            hidden_dim=saved.hidden_dim, enc_layers=saved.n_layers, **common)
    elif arch == "transformer":
        model = TransformerModel(
            len(src_vocab), len(tgt_vocab),
            d_model=saved.embed_dim, nhead=saved.nhead,
            n_enc_layers=saved.n_layers, n_dec_layers=saved.n_layers,
            d_ff=saved.ffn_dim, dropout=0.0, pad_idx=PAD_IDX)
    else:
        raise ValueError(f"Unknown arch: {arch!r}")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, src_vocab, tgt_vocab


# ─── Beam Search (Transformer only) ─────────────────────────────────────────
def beam_search_transformer(
    model: TransformerModel,
    src: torch.Tensor,
    src_lengths,
    sos_idx: int,
    eos_idx: int,
    max_len: int,
    beam_size: int,
) -> torch.Tensor:
    """
    Length-normalised beam search for TransformerModel.
    Processes each item in the batch sequentially.
    Returns (B, out_len) token ids.
    """
    device       = src.device
    B            = src.size(0)
    src_pad_mask = (src == model.pad_idx)
    memory_full  = model.encoder(
        model._embed_src(src),
        src_key_padding_mask=src_pad_mask)

    results = []
    for b in range(B):
        memory   = memory_full[b: b + 1]
        pad_mask = src_pad_mask[b: b + 1]

        beams = [(0.0, [sos_idx], False)]   # (score, token_ids, is_done)

        for _ in range(max_len):
            if all(done for _, _, done in beams):
                break

            new_beams = []
            for score, seq, done in beams:
                if done:
                    new_beams.append((score, seq, True))
                    continue

                inp      = torch.tensor([seq], dtype=torch.long, device=device)
                tgt_mask = model._causal_mask(len(seq), device)
                out = model.decoder(
                    model._embed_tgt(inp), memory,
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=pad_mask)

                log_probs = F.log_softmax(model.fc_out(out[0, -1]), dim=-1)
                k = min(beam_size, log_probs.size(-1))
                top_scores, top_ids = log_probs.topk(k)

                for i in range(k):
                    tok       = top_ids[i].item()
                    new_score = score + top_scores[i].item()
                    new_beams.append((new_score, seq + [tok], tok == eos_idx))

            new_beams.sort(
                key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
            beams = new_beams[:beam_size]

        finished = [(s, seq) for s, seq, done in beams if done]
        best_seq = (
            max(finished, key=lambda x: x[0] / max(len(x[1]), 1))[1]
            if finished
            else max(beams, key=lambda x: x[0] / max(len(x[1]), 1))[1]
        )
        # Skip SOS token (first token), filter out EOS tokens
        if len(best_seq) > 1:
            results.append([t for t in best_seq[1:] if t != eos_idx])
        else:
            results.append([])  # Empty sequence if beam search produces only SOS

    max_out = max((len(r) for r in results), default=1)
    output  = torch.zeros(B, max_out, dtype=torch.long, device=device)
    for b, r in enumerate(results):
        if r:
            output[b, :len(r)] = torch.tensor(r, dtype=torch.long, device=device)
    return output


def build_inference_replicas(
    arch: str,
    checkpoint_path: str,
    primary_model,
    num_devices: int,
):
    """Load one model replica per GPU for sharded greedy inference."""
    devices = [torch.device(f"cuda:{i}") for i in range(num_devices)]
    replicas = [primary_model]
    for device in devices[1:]:
        replica, _, _ = load_model(arch, checkpoint_path, device)
        replicas.append(replica)
    return replicas, devices


def greedy_generate_multi_gpu(
    replicas,
    devices,
    src_cpu: torch.Tensor,
    src_len_cpu: torch.Tensor,
    max_len: int,
):
    """
    Split a batch across multiple GPUs and run greedy generation on each shard.

    This keeps sentence order intact while making use of independent decoder
    workloads on multiple devices.
    """
    batch_size = src_cpu.size(0)
    if batch_size == 0:
        return []

    shard_count = min(len(replicas), batch_size)
    base = batch_size // shard_count
    remainder = batch_size % shard_count

    pending = []
    start = 0
    for shard_idx in range(shard_count):
        shard_size = base + (1 if shard_idx < remainder else 0)
        end = start + shard_size
        if shard_size == 0:
            continue

        device = devices[shard_idx]
        src_chunk = src_cpu[start:end].to(device, non_blocking=True)
        len_chunk = src_len_cpu[start:end].to(device, non_blocking=True)
        out_chunk = replicas[shard_idx].generate(
            src_chunk, len_chunk, SOS_IDX, EOS_IDX, max_len)
        pending.append((start, out_chunk))
        start = end

    ordered = [None] * batch_size
    for start, out_chunk in pending:
        out_cpu = out_chunk.detach().cpu()
        for row_idx in range(out_cpu.size(0)):
            ordered[start + row_idx] = out_cpu[row_idx]
    return ordered


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Generate GEC corrections — PolyGEC v2")
    p.add_argument("--lang",       required=True)
    p.add_argument("--arch",       required=True,
                   choices=["rnn", "lstm", "transformer"])
    p.add_argument("--config",     required=True,
                   help="Tokenization config, e.g. 0.5K_1K, 16K_16K")
    p.add_argument("--max-len",    type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-parallel", action="store_true",
                   help="Use all visible GPUs for greedy inference when possible")
    p.add_argument("--beam-size",  type=int, default=1,
                   help="Beam size for transformer (1=greedy). Ignored for rnn/lstm.")
    p.add_argument("--checkpoint", default=None)
    args = p.parse_args()

    # Validate beam size
    if args.beam_size < 1:
        p.error("--beam-size must be at least 1")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir  = f"data/tokenized/{args.lang}/{args.config}"
    ckpt_path = (args.checkpoint or
                 f"checkpoints/{args.lang}/{args.arch}_{args.config}/best.pt")
    use_beam  = (args.arch == "transformer" and args.beam_size > 1)

    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint at {ckpt_path}")
        sys.exit(1)

    os.makedirs("results", exist_ok=True)
    print(f"Generating: {args.lang} / {args.arch} / {args.config}  ({device})")

    model, src_vocab, tgt_vocab = load_model(args.arch, ckpt_path, device)
    multi_gpu_replicas = None
    multi_gpu_devices = None
    if args.data_parallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if use_beam:
            print("  ⚠ Multi-GPU generation is only enabled for greedy decoding")
        else:
            num_devices = torch.cuda.device_count()
            multi_gpu_replicas, multi_gpu_devices = build_inference_replicas(
                args.arch, ckpt_path, model, num_devices)
            print(f"  ✓ Using {num_devices} GPUs for sharded greedy inference")

    # ── Load test set with truncation (alignment guarantee) ───────────────────
    #
    # BUG FIX: previously used truncate=False (the default), which silently
    # dropped sentences longer than max_len.  This meant the output .hyp file
    # had fewer lines than test.src, breaking line-by-line correspondence with
    # downstream evaluation tools (e.g. ERRANT, M2Scorer).
    #
    # With truncate=True, long sequences are clipped to max_len (EOS preserved
    # at position max_len-1), so every test sentence produces exactly one
    # hypothesis.  The model receives a slightly truncated input for those
    # sentences, which is a minor quality trade-off but is far preferable to
    # silent misalignment.
    #
    # The raw_src / raw_ref collection below uses the same truncation so all
    # three output files (.hyp, .ref, .src) are guaranteed to have the same
    # number of lines as test.src.
    dataset = GECDataset(
        f"{data_dir}/test.src", f"{data_dir}/test.tgt",
        src_vocab, tgt_vocab, args.max_len,
        truncate=True)   # ← keeps every sentence; truncates instead of drops

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0))

    # Collect raw (de-BPE'd) source + reference
    # References: use ORIGINAL cased versions (not recased from model)
    # Sources: use de-BPE'd lowercase (will be recased for display)
    raw_src = []
    raw_ref = []
    
    # Try to load original cased references first (from preprocess.py)
    orig_ref_path = f"data/processed/{args.lang}/test_original.tgt"
    if os.path.exists(orig_ref_path):
        print(f"  ✓ Using original cased references: {orig_ref_path}")
        with open(orig_ref_path, encoding="utf-8") as f:
            raw_ref = [line.strip() for line in f]
    else:
        print(f"  ⚠ No original references found at {orig_ref_path}")
        print(f"      Falling back to recased tokenized references (less accurate)")
        with open(f"{data_dir}/test.tgt", encoding="utf-8") as f:
            raw_ref = [debpe(line.strip()) for line in f]
    
    # Load source sentences (will be recased for display)
    with open(f"{data_dir}/test.src", encoding="utf-8") as f:
        raw_src = [debpe(line.strip()) for line in f]
    
    # Alignment guarantee: truncation logic ensures indices match
    assert len(raw_src) == len(raw_ref), (
        f"Misaligned files: src={len(raw_src)}, ref={len(raw_ref)}")

    print(f"  Test sentences : {len(dataset):,}")

    # ── Generate ─────────────────────────────────────────────────────────────
    # Performance notes:
    #   - Greedy (beam_size=1): ~2-5s for 75k sentences (batched, efficient)
    #   - Beam search (beam_size>1): ~5 hours for 75k (sequential per sample, slow)
    #   - Quality difference: beam search improves ~0.5-1.0 GLEU typically
    hypotheses = []

    t_start = time.perf_counter()
    with torch.no_grad():
        for src, src_len, _ in tqdm(loader, desc="Generating", unit="batch"):
            if multi_gpu_replicas is not None:
                generated_rows = greedy_generate_multi_gpu(
                    multi_gpu_replicas,
                    multi_gpu_devices,
                    src,
                    src_len,
                    args.max_len,
                )
                for row in generated_rows:
                    hypotheses.append(tgt_vocab.decode(row))
                continue

            src = src.to(device, non_blocking=True)
            src_len = src_len.to(device, non_blocking=True)

            if use_beam:
                # WARNING: This is VERY slow (processes each item sequentially)
                # For 75k test set: ~5 hours with beam_size=4
                gen = beam_search_transformer(
                    model, src, src_len, SOS_IDX, EOS_IDX,
                    args.max_len, args.beam_size)
            else:
                # Fast batched greedy decoding (~2-5 seconds for 75k test set)
                gen = model.generate(src, src_len, SOS_IDX, EOS_IDX, args.max_len)

            for i in range(gen.size(0)):
                hypotheses.append(tgt_vocab.decode(gen[i]))

    elapsed = time.perf_counter() - t_start
    n       = len(hypotheses)
    print(f"  ✓ {n:,} sentences  |  {elapsed:.1f}s  |  "
          f"{elapsed / max(n, 1) * 1000:.1f} ms/sentence")

    # Sanity check: all three lists must have the same length
    assert n == len(raw_ref) == len(raw_src), (
        f"Alignment error: hyp={n}, ref={len(raw_ref)}, src={len(raw_src)}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    prefix = f"results/{args.lang}_{args.arch}_{args.config}"

    # Check if references are already cased (from original file)
    has_original_refs = os.path.exists(f"data/processed/{args.lang}/test_original.tgt")

    # Load recaser if available
    recaser = None
    if recaser_exists(args.lang):
        print("  Loading Kneser-Ney recaser...")
        recaser = load_recaser(args.lang)
        print(f"  ✓ Recaser loaded for {args.lang}")
    else:
        print(f"  ⚠ No recaser model found for {args.lang}")
        print(f"    → Run: python scripts/train_recaser.py --lang {args.lang} "
              f"--input data/raw/{args.lang}/train.csv")
        print("    → Using simple first-letter capitalization as fallback")

    # Apply recasing and detokenization
    print("  Applying recasing...")
    hyp_lines = apply_recaser_batch(hypotheses, recaser)
    hyp_lines = [detokenize_text(line) for line in hyp_lines]
    
    if has_original_refs:
        ref_lines = raw_ref  # Already properly cased
    else:
        ref_lines = apply_recaser_batch(raw_ref, recaser)
        ref_lines = [detokenize_text(line) for line in ref_lines]
    
    src_lines = apply_recaser_batch(raw_src, recaser)
    src_lines = [detokenize_text(line) for line in src_lines]

    for suffix, lines in [
        (".hyp", hyp_lines),
        (".ref", ref_lines),
        (".src", src_lines),
    ]:
        with open(prefix + suffix, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    print(f"  Files : {prefix}.{{hyp,ref,src}}")
    print(f"  Next  : python scripts/evaluate.py "
          f"--hyp {prefix}.hyp --ref {prefix}.ref --src {prefix}.src "
          f"--checkpoint {ckpt_path}")


if __name__ == "__main__":
    main()
