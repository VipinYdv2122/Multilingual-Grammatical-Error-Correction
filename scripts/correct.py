#!/usr/bin/env python3
"""
Interactive GEC Checker — type a sentence, get the corrected version.

Usage:
    python scripts/correct.py --lang english --arch transformer --config 16K_16K

Type 'quit' to exit.
"""

import argparse
import os
import re
import sys
import unicodedata

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# BUG FIX: was RNNAttentionModel / LSTMAttentionModel — wrong class names.
# The actual classes are RNNSeq2Seq / LSTMSeq2Seq (consistent with train.py
# and generate.py).
from models.rnn_attention  import RNNSeq2Seq
from models.lstm_attention import LSTMSeq2Seq
from models.transformer    import TransformerModel
from models.dataset        import Vocabulary, PAD_IDX, SOS_IDX, EOS_IDX

# Handle both direct execution and module import
try:
    from scripts.recaser_utils import (
        apply_recaser,
        detokenize_text,
        load_recaser,
        recaser_exists,
    )
except ModuleNotFoundError:
    from recaser_utils import apply_recaser, detokenize_text, load_recaser, recaser_exists


# ─── Normalisation (must match preprocess.py) ─────────────────────────────────
def _normalize(text: str) -> str:
    """Apply the same normalisation as preprocess.py normalize_text().

    BUG FIX: correct.py previously fed raw user input directly to BPE/vocab.
    Since training data is lowercased + punct-detached, raw mixed-case input
    produces OOV tokens that are silently dropped, degrading output quality.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"([.,!?;:'\"\\-])", r" \1 ", text)
    return " ".join(text.split())


# ─── Vocab path resolver (mirrors generate.py) ────────────────────────────────
def _resolve_vocab_path(saved_path: str, ckpt_path: str) -> str:
    if os.path.exists(saved_path):
        return saved_path
    for base in (os.path.dirname(os.path.abspath(ckpt_path)), os.getcwd()):
        candidate = os.path.join(base, os.path.basename(saved_path))
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Cannot locate vocab: {saved_path!r}\n"
        f"  Hint: run from the project root, or pass --checkpoint with full path.")


# ─── Model loader ─────────────────────────────────────────────────────────────
def load_model(arch: str, checkpoint_path: str, device):
    """Load a trained model from checkpoint."""
    ckpt       = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_args = argparse.Namespace(**ckpt["args"])

    src_vocab = Vocabulary().load(
        _resolve_vocab_path(ckpt["src_vocab_path"], checkpoint_path))
    tgt_vocab = Vocabulary().load(
        _resolve_vocab_path(ckpt["tgt_vocab_path"], checkpoint_path))

    # BUG FIX: was emb_dim= / hid_dim= which don't match the constructors.
    # Correct kwargs are embed_dim= / hidden_dim= (consistent with train.py).
    common = dict(embed_dim=saved_args.embed_dim, dropout=0.0,
                  sos_idx=SOS_IDX, eos_idx=EOS_IDX, pad_idx=PAD_IDX)

    if arch == "rnn":
        model = RNNSeq2Seq(
            len(src_vocab), len(tgt_vocab),
            hidden_dim=saved_args.hidden_dim, enc_layers=saved_args.n_layers,
            **common)
    elif arch == "lstm":
        model = LSTMSeq2Seq(
            len(src_vocab), len(tgt_vocab),
            hidden_dim=saved_args.hidden_dim, enc_layers=saved_args.n_layers,
            **common)
    elif arch == "transformer":
        model = TransformerModel(
            len(src_vocab), len(tgt_vocab),
            d_model=saved_args.embed_dim, nhead=saved_args.nhead,
            n_enc_layers=saved_args.n_layers, n_dec_layers=saved_args.n_layers,
            d_ff=saved_args.ffn_dim, dropout=0.0, pad_idx=PAD_IDX)
    else:
        raise ValueError(f"Unknown arch: {arch!r}.  "
                         f"Choose from: rnn, lstm, transformer")

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, src_vocab, tgt_vocab


# ─── Correction ───────────────────────────────────────────────────────────────
def correct_sentence(model, src_vocab: Vocabulary, tgt_vocab: Vocabulary,
                     sentence: str, bpe_codes_path: str | None,
                     device, recaser=None) -> str:
    """
    Correct a single sentence:
      1. Normalise (lowercase + punct detach) — must match preprocess.py
      2. Apply BPE tokenization (if codes file is provided)
      3. Encode with source vocab
      4. Run model.generate()
      5. Decode with target vocab (removes BPE markers)
      6. Apply the LM recaser to restore casing
    """
    # BUG FIX: normalise before BPE so tokens match the training distribution
    sentence = _normalize(sentence.strip())

    # Apply BPE if a codes file is available
    if bpe_codes_path and os.path.exists(bpe_codes_path):
        try:
            import codecs
            from subword_nmt.apply_bpe import BPE
            with codecs.open(bpe_codes_path, encoding="utf-8") as f:
                bpe = BPE(f)
            sentence = bpe.process_line(sentence)
        except ImportError:
            print("  ⚠ subword-nmt not installed — BPE not applied.")

    src_ids    = src_vocab.encode(sentence)
    src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)
    src_length = torch.tensor([len(src_ids)], dtype=torch.long, device=device)

    # Allow slightly longer output than input (corrections can add words)
    out_max = min(len(src_ids) + 20, 150)
    output  = model.generate(src_tensor, src_length, SOS_IDX, EOS_IDX,
                              max_len=out_max)
    corrected = tgt_vocab.decode(output[0])

    corrected = apply_recaser(corrected, recaser)
    return detokenize_text(corrected)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Interactive GEC Checker")
    parser.add_argument("--lang",           required=True)
    parser.add_argument("--arch",           required=True,
                        choices=["rnn", "lstm", "transformer"])
    parser.add_argument("--config",         required=True,
                        help="Tokenization config, e.g. 0.5K_1K, 16K_16K")
    parser.add_argument("--checkpoint",     default=None)
    args = parser.parse_args()

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = (args.checkpoint or
                 f"checkpoints/{args.lang}/{args.arch}_{args.config}/best.pt")

    if not os.path.exists(ckpt_path):
        print(f"ERROR: No checkpoint at {ckpt_path}")
        print(f"  Train first: python scripts/train.py "
              f"--lang {args.lang} --arch {args.arch} --config {args.config}")
        sys.exit(1)

    # Determine which BPE codes to apply to the source side
    # Config format is <src_label>_<tgt_label>, e.g. '0.5K_1K'
    src_bpe_label = args.config.split("_")[0]  # e.g. '0.5K'
    bpe_codes     = f"tokenizers/models/{args.lang}/bpe_{src_bpe_label}.codes"
    if not os.path.exists(bpe_codes):
        print(f"  ⚠ BPE codes not found at {bpe_codes} — raw input will be used.")
        bpe_codes = None

    # Load LM recaser if available
    recaser = None
    if recaser_exists(args.lang):
        try:
            recaser = load_recaser(args.lang)
            print(f"  ✓ LM recaser loaded for {args.lang}")
        except Exception as exc:
            print(f"  ⚠ Failed to load LM recaser for {args.lang}: {exc}")
            print("    → Using first-letter fallback")
    else:
        print("  ⚠ No recaser found — using first-letter fallback")

    print(f"Loading model: {args.lang} / {args.arch} / {args.config}")
    model, src_vocab, tgt_vocab = load_model(args.arch, ckpt_path, device)
    print("Ready!  Type a sentence to correct (or 'quit' to exit).\n")

    while True:
        try:
            sentence = input("Input:  ")
        except (EOFError, KeyboardInterrupt):
            break

        if sentence.strip().lower() in ("quit", "exit", "q"):
            break
        if not sentence.strip():
            continue

        corrected = correct_sentence(
            model, src_vocab, tgt_vocab, sentence, bpe_codes, device,
            recaser=recaser)
        print(f"Output: {corrected}\n")

    print("Bye!")


if __name__ == "__main__":
    main()
