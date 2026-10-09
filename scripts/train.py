#!/usr/bin/env python3
"""
Training script for GEC models — PolyGEC v2.

Supports: TPU (Kaggle XLA), multi-GPU (DataParallel), single GPU, CPU.
AMP (Automatic Mixed Precision) gives ~2× speedup on T4/A100 GPUs.

Usage:
    python scripts/train.py --lang english  --arch transformer --config 0.5K_1K
    python scripts/train.py --lang english  --arch transformer --config 16K_16K

Config names use K-notation: <src_merges>_<tgt_merges>
  e.g. 0.5K_1K = source BPE 500 merges, target BPE 1000 merges

Transformer T4 defaults:
  lr=1e-4  dropout=0.3  weight_decay=1e-4
  embed_dim=512  layers=6  heads=8  ffn=2048
  Noam warmup (2000 steps)  label_smoothing=0.1
"""

import argparse
import csv
import os
import random
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# --- TPU support (Kaggle) ---
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    HAS_TPU = True
except ImportError:
    HAS_TPU = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.rnn_attention  import RNNSeq2Seq
from models.lstm_attention import LSTMSeq2Seq
from models.transformer    import TransformerModel
from models.dataset        import Vocabulary, get_dataloaders, PAD_IDX, SOS_IDX, EOS_IDX


# ─── Noam Warmup Scheduler ────────────────────────────────────────────────────
class NoamScheduler:
    """
    Transformer LR schedule (Vaswani et al. 2017, §5.3):
        lr = d_model^{-0.5} · min(step^{-0.5},  step · warmup^{-1.5})
    Ramps linearly 0 → peak at step == warmup_steps, then decays as 1/√step.
    """

    def __init__(self, optimizer, d_model: int, warmup_steps: int):
        self.optimizer    = optimizer
        self.d_model      = d_model
        self.warmup_steps = max(warmup_steps, 1)
        self._step        = 0
        for pg in self.optimizer.param_groups:
            pg["lr"] = 0.0

    def step(self):
        self._step += 1
        lr = self.d_model ** -0.5 * min(
            self._step ** -0.5,
            self._step * self.warmup_steps ** -1.5)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ─── Device ──────────────────────────────────────────────────────────────────
def get_device():
    if HAS_TPU:
        dev = xm.xla_device()
        print(f"  ✓ TPU: {dev}")
        return dev
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"  ✓ CUDA ({torch.cuda.device_count()} GPU(s))")
        return dev
    print("  CPU")
    return torch.device("cpu")


# ─── Model builder ───────────────────────────────────────────────────────────
def build_model(arch, src_vocab_size, tgt_vocab_size, args, device):
    common = dict(embed_dim=args.embed_dim, dropout=args.dropout,
                  sos_idx=SOS_IDX, eos_idx=EOS_IDX, pad_idx=PAD_IDX)

    if arch == "rnn":
        model = RNNSeq2Seq(
            src_vocab_size, tgt_vocab_size,
            hidden_dim=args.hidden_dim, enc_layers=args.n_layers, **common)
    elif arch == "lstm":
        model = LSTMSeq2Seq(
            src_vocab_size, tgt_vocab_size,
            hidden_dim=args.hidden_dim, enc_layers=args.n_layers, **common)
    elif arch == "transformer":
        model = TransformerModel(
            src_vocab_size, tgt_vocab_size,
            d_model=args.embed_dim, nhead=args.nhead,
            n_enc_layers=args.n_layers, n_dec_layers=args.n_layers,
            d_ff=args.ffn_dim, dropout=args.dropout, pad_idx=PAD_IDX)
    else:
        raise ValueError(f"Unknown arch: {arch!r}")

    model = model.to(device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  {arch} : {n:,} parameters")

    if not HAS_TPU and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"  DataParallel ({torch.cuda.device_count()} GPUs)")

    return model


# ─── Training loop ───────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, clip, device,
                    tf_ratio, use_amp=False, scaler=None, noam=None):
    model.train()
    total_loss, n = 0.0, 0

    bar = tqdm(loader, desc="  Train", unit="batch",
               file=sys.stderr, dynamic_ncols=False, ncols=100,
               mininterval=1.0, leave=True)

    for src, src_len, trg in bar:
        src = src.to(device, non_blocking=True)
        src_len = src_len.to(device, non_blocking=True)
        trg = trg.to(device, non_blocking=True)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            output = model(src, src_len, trg, teacher_forcing_ratio=tf_ratio)
            # output : (B, T-1, V)   trg[:, 1:] : (B, T-1)
            # No [:, 1:] on output — forward() already returns predictions
            # for positions 1..T-1 only.
            loss = criterion(
                output.reshape(-1, output.size(-1)),
                trg[:, 1:].reshape(-1))

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            if HAS_TPU:
                xm.optimizer_step(optimizer)
            else:
                optimizer.step()

        if noam is not None:
            noam.step()

        total_loss += loss.item()
        n += 1
        bar.set_postfix(loss=f"{total_loss / n:.4f}")

    return total_loss / max(n, 1)


def evaluate_loss(model, loader, criterion, device, use_amp=False):
    model.eval()
    total_loss, n = 0.0, 0

    bar = tqdm(loader, desc="  Valid", unit="batch",
               file=sys.stderr, dynamic_ncols=False, ncols=100,
               mininterval=1.0, leave=True)

    with torch.no_grad():
        for src, src_len, trg in bar:
            src = src.to(device, non_blocking=True)
            src_len = src_len.to(device, non_blocking=True)
            trg = trg.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(src, src_len, trg, teacher_forcing_ratio=0.0)
                loss   = criterion(
                    output.reshape(-1, output.size(-1)),
                    trg[:, 1:].reshape(-1))
            total_loss += loss.item()
            n += 1
            bar.set_postfix(loss=f"{total_loss / n:.4f}")

    return total_loss / max(n, 1)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Train a GEC model — PolyGEC v2")
    p.add_argument("--lang",   required=True)
    p.add_argument("--arch",   required=True,
                   choices=["rnn", "lstm", "transformer"])
    p.add_argument("--config", required=True,
                   help="Tokenization config, e.g. 0.5K_1K, 16K_16K")

    # Architecture
    p.add_argument("--embed-dim",  type=int,   default=512)
    p.add_argument("--hidden-dim", type=int,   default=512)
    p.add_argument("--n-layers",   type=int,   default=6,
                   help="Encoder/decoder layers (default 6 for Transformer T4)")
    p.add_argument("--nhead",      type=int,   default=8)
    p.add_argument("--ffn-dim",    type=int,   default=2048)
    p.add_argument("--dropout",    type=float, default=None,
                   help="Default: 0.3 (transformer)")

    # Optimisation
    p.add_argument("--lr",              type=float, default=None,
                   help="Default: 1e-3 (rnn/lstm)  or  1e-4 (transformer)")
    p.add_argument("--weight-decay",    type=float, default=1e-4)
    p.add_argument("--warmup-steps",    type=int,   default=2000,
                   help="Noam warmup steps (transformer only).")
    p.add_argument("--label-smoothing", type=float, default=None,
                   help="Default: 0.1 (transformer)  or  0.0 (rnn/lstm)")
    p.add_argument("--batch-size",      type=int,   default=128)
    p.add_argument("--epochs",          type=int,   default=15)
    p.add_argument("--patience",        type=int,   default=5)
    p.add_argument("--lr-patience",     type=int,   default=2)
    p.add_argument("--min-lr",          type=float, default=1e-5)
    p.add_argument("--clip",            type=float, default=1.0)
    p.add_argument("--teacher-forcing", type=float, default=0.5,
                   help="Initial TF ratio, linearly annealed to 0.")
    p.add_argument("--max-len",         type=int,   default=100)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--num-workers",     type=int,   default=4)

    args = p.parse_args()

    # Arch-aware defaults (Transformer T4)
    is_transformer = (args.arch == "transformer")
    if args.lr              is None: args.lr              = 1e-4 if is_transformer else 1e-3
    if args.label_smoothing is None: args.label_smoothing = 0.1  if is_transformer else 0.0
    if args.dropout         is None: args.dropout         = 0.3  if is_transformer else 0.4

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = get_device()

    data_dir    = f"data/tokenized/{args.lang}/{args.config}"
    save_dir    = f"checkpoints/{args.lang}/{args.arch}_{args.config}"
    vocab_dir   = f"data/vocabs/{args.lang}/{args.config}"
    results_dir = f"results/{args.lang}/{args.config}"
    for d in (save_dir, vocab_dir, results_dir):
        os.makedirs(d, exist_ok=True)

    print(f"\n  Lang={args.lang}  Arch={args.arch}  Config={args.config}")
    print(f"  lr={args.lr:.2e}  dropout={args.dropout}  "
          f"wd={args.weight_decay:.0e}  smooth={args.label_smoothing}")
    if is_transformer:
        print(f"  Noam warmup={args.warmup_steps} steps")
    print()

    # ── Vocabulary ────────────────────────────────────────────────────────────
    src_vocab_path = os.path.join(vocab_dir, "src_vocab.txt")
    tgt_vocab_path = os.path.join(vocab_dir, "tgt_vocab.txt")
    src_vocab, tgt_vocab = Vocabulary(), Vocabulary()

    if os.path.exists(src_vocab_path) and os.path.exists(tgt_vocab_path):
        src_vocab.load(src_vocab_path)
        tgt_vocab.load(tgt_vocab_path)
        print(f"  Vocab: src={len(src_vocab):,}  tgt={len(tgt_vocab):,}")
    else:
        src_files = [os.path.join(data_dir, f"{s}.src")
                     for s in ("train", "valid")
                     if os.path.exists(os.path.join(data_dir, f"{s}.src"))]
        tgt_files = [os.path.join(data_dir, f"{s}.tgt")
                     for s in ("train", "valid")
                     if os.path.exists(os.path.join(data_dir, f"{s}.tgt"))]
        src_vocab.build_from_files(src_files)
        tgt_vocab.build_from_files(tgt_files)
        src_vocab.save(src_vocab_path)
        tgt_vocab.save(tgt_vocab_path)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, valid_loader, _ = get_dataloaders(
        data_dir, src_vocab, tgt_vocab, args.batch_size, args.max_len,
        num_workers=args.num_workers)

    if train_loader is None:
        print("ERROR: Training data not found — check data_dir path.")
        sys.exit(1)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(args.arch, len(src_vocab), len(tgt_vocab), args, device)

    # ── Optimiser & Scheduler ─────────────────────────────────────────────────
    if is_transformer and args.warmup_steps > 0:
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr,
            betas=(0.9, 0.98), eps=1e-9,
            weight_decay=args.weight_decay)
        noam_sched    = NoamScheduler(optimizer, args.embed_dim, args.warmup_steps)
        plateau_sched = None
        print(f"  ✓ Noam ({args.warmup_steps} steps)")
    else:
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay)
        noam_sched    = None
        plateau_sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.5, patience=args.lr_patience,
            min_lr=args.min_lr)

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_IDX, label_smoothing=args.label_smoothing)

    # ── AMP ───────────────────────────────────────────────────────────────────
    use_amp = (getattr(device, "type", "") == "cuda")
    scaler  = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("  ✓ AMP enabled")

    # ── CSV log ───────────────────────────────────────────────────────────────
    log_path   = os.path.join(results_dir,
                              f"{args.lang}_{args.arch}_{args.config}_log.csv")
    log_file   = open(log_path, "w", newline="", encoding="utf-8")
    log_writer = csv.writer(log_file)
    log_writer.writerow(
        ["epoch", "train_loss", "valid_loss", "gap", "lr", "tf_ratio", "time_s"])
    print(f"  Log: {log_path}\n")

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss     = float("inf")
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Teacher-forcing: linearly anneal to 0
        tf_ratio = args.teacher_forcing * max(
            0.0, 1.0 - (epoch - 1) / max(args.epochs - 1, 1))

        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  Epoch {epoch:2d}/{args.epochs}  tf={tf_ratio:.3f}  lr={cur_lr:.2e}")

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            args.clip, device, tf_ratio,
            use_amp=use_amp, scaler=scaler, noam=noam_sched)
        print(f"  train loss: {train_loss:.4f}")

        valid_loss = (evaluate_loss(model, valid_loader, criterion, device, use_amp)
                      if valid_loader else train_loss)
        gap = valid_loss - train_loss
        if valid_loader:
            print(f"  valid loss: {valid_loss:.4f}  (gap={gap:+.4f})")

        if plateau_sched is not None:
            plateau_sched.step(valid_loss)
        cur_lr = optimizer.param_groups[0]["lr"]

        saved = ""
        if valid_loss < best_loss:
            best_loss     = valid_loss
            patience_left = args.patience
            m = model.module if hasattr(model, "module") else model
            torch.save({
                "model_state_dict": m.state_dict(),
                "args":             vars(args),
                "src_vocab_path":   os.path.abspath(src_vocab_path),
                "tgt_vocab_path":   os.path.abspath(tgt_vocab_path),
            }, os.path.join(save_dir, "best.pt"))
            saved = " ✓"
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  Early stopping (epoch {epoch}).")
                break

        elapsed = time.time() - t0
        log_writer.writerow([
            epoch, f"{train_loss:.6f}", f"{valid_loss:.6f}", f"{gap:+.6f}",
            f"{cur_lr:.2e}", f"{tf_ratio:.4f}", f"{elapsed:.1f}"])
        log_file.flush()
        print(f"  ── {elapsed:.0f}s | lr={cur_lr:.2e} | best={best_loss:.4f}{saved}")

        if valid_loader and gap > 1.5:
            print(f"  ⚠  Gap={gap:+.2f} — consider higher dropout or smaller batch")

    log_file.close()
    print(f"\nDone. Best valid loss: {best_loss:.4f}")
    print(f"Checkpoint : {save_dir}/best.pt")
    print(f"CSV log    : {log_path}")


if __name__ == "__main__":
    main()
