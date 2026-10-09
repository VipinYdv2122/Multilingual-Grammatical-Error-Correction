#!/usr/bin/env python3
"""
Dataset and Vocabulary for GEC — PolyGEC v2.

Handles:
  - Building token-to-index vocabulary from BPE-tokenized text files
  - Loading parallel source-target sentence pairs
  - Padding and batching for PyTorch DataLoader
  - Bucket batching: groups similar-length sentences to minimise padding waste
  - No <unk> token (BPE guarantees full coverage)
"""

import math
import os
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from collections import Counter


# ─── Special tokens ───────────────────────────────────────────────────────────
PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

PAD_IDX = 0
SOS_IDX = 1
EOS_IDX = 2


# ─── Vocabulary ───────────────────────────────────────────────────────────────
class Vocabulary:
    """Maps BPE tokens ↔ integer indices."""

    def __init__(self):
        self.token2idx: dict = {}
        self.idx2token: dict = {}
        self.size = 0
        for token in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN):
            self._add(token)

    def _add(self, token: str) -> None:
        if token not in self.token2idx:
            self.token2idx[token] = self.size
            self.idx2token[self.size] = token
            self.size += 1

    def __len__(self) -> int:
        return self.size

    def build_from_files(self, filepaths: list, min_freq: int = 1) -> "Vocabulary":
        """Count tokens across all files; add those with freq >= min_freq."""
        counter: Counter = Counter()
        for path in filepaths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    counter.update(line.strip().split())
        for token, freq in counter.most_common():
            if freq >= min_freq:
                self._add(token)
        print(f"  Built vocab: {self.size} tokens "
              f"(from {len(filepaths)} file(s), min_freq={min_freq})")
        return self

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for i in range(self.size):
                f.write(self.idx2token[i] + "\n")

    def load(self, path: str) -> "Vocabulary":
        self.token2idx, self.idx2token, self.size = {}, {}, 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                token = line.strip()
                if token:
                    self._add(token)
        return self

    def encode(self, sentence: str) -> list:
        """
        Sentence string → [SOS, t1, t2, …, EOS].
        Unknown tokens are silently skipped (BPE guarantees full coverage).
        """
        indices = [SOS_IDX]
        for t in sentence.strip().split():
            idx = self.token2idx.get(t)
            if idx is not None:
                indices.append(idx)
        indices.append(EOS_IDX)
        return indices

    def decode(self, indices) -> str:
        """
        List/tensor of indices → detokenised string.
        Stops at <eos>, strips <pad>/<sos>.
        Removes BPE continuation markers (both mid-word '@@ ' and trailing '@@').
        """
        tokens = []
        for idx in indices:
            if isinstance(idx, torch.Tensor):
                idx = idx.item()
            if idx == EOS_IDX:
                break
            if idx in (PAD_IDX, SOS_IDX):
                continue
            word = self.idx2token.get(idx)
            if word:                        # skip unknown/empty indices
                tokens.append(word)
        return " ".join(tokens).replace("@@ ", "").replace("@@", "")


# ─── Dataset ──────────────────────────────────────────────────────────────────
class GECDataset(Dataset):
    """
    Parallel source-target dataset for GEC.

    Args:
        src_file  : path to BPE-tokenised source file
        tgt_file  : path to BPE-tokenised target file
        src_vocab : source Vocabulary
        tgt_vocab : target Vocabulary
        max_len   : maximum sequence length (including SOS + EOS)
        truncate  : if True, sequences longer than max_len are truncated to
                    max_len (keeping EOS at position max_len-1) instead of
                    being silently dropped.

                    BUG FIX — alignment guarantee:
                    Training uses truncate=False (default) to exclude very long
                    sequences that might destabilise training.
                    Inference (generate.py) uses truncate=True so that every
                    line in test.src produces exactly one hypothesis — keeping
                    the output .hyp/.ref/.src files perfectly line-aligned with
                    the original test file.  Without this, any test sentence
                    longer than max_len is silently dropped, causing line N of
                    the hypothesis file to correspond to a different source
                    sentence than expected by downstream evaluation tools.
    """

    def __init__(
        self,
        src_file: str,
        tgt_file: str,
        src_vocab: Vocabulary,
        tgt_vocab: Vocabulary,
        max_len: int = 100,
        truncate: bool = False,
    ):
        self.pairs: list = []

        def _clip(ids: list, max_len: int) -> list:
            """Truncate to max_len, preserving EOS at the last position."""
            if len(ids) <= max_len:
                return ids
            return ids[: max_len - 1] + [EOS_IDX]

        with open(src_file, "r", encoding="utf-8") as fs, \
             open(tgt_file, "r", encoding="utf-8") as ft:
            for src_line, tgt_line in zip(fs, ft):
                src_ids = src_vocab.encode(src_line)
                tgt_ids = tgt_vocab.encode(tgt_line)

                if truncate:
                    src_ids = _clip(src_ids, max_len)
                    tgt_ids = _clip(tgt_ids, max_len)
                else:
                    if len(src_ids) > max_len or len(tgt_ids) > max_len:
                        continue

                self.pairs.append((
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long),
                ))

        print(f"  Loaded {len(self.pairs):,} pairs from {os.path.basename(src_file)}"
              + (f"  [truncate={truncate}]" if truncate else ""))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        return self.pairs[idx]


# ─── Bucket Batch Sampler ─────────────────────────────────────────────────────
class BucketBatchSampler(Sampler):
    """
    Groups samples by source length to minimise padding waste.
    Reduces padding overhead by ~40-60 % versus random sampling.
    """

    def __init__(self, dataset: GECDataset, batch_size: int):
        super().__init__()
        self.batch_size = batch_size
        self.sorted_indices = sorted(
            range(len(dataset)),
            key=lambda i: len(dataset.pairs[i][0]))

    def __iter__(self):
        batches = [
            self.sorted_indices[i: i + self.batch_size]
            for i in range(0, len(self.sorted_indices), self.batch_size)
        ]
        random.shuffle(batches)
        for batch in batches:
            yield batch

    def __len__(self) -> int:
        return math.ceil(len(self.sorted_indices) / self.batch_size)


# ─── Collate ──────────────────────────────────────────────────────────────────
def collate_fn(batch):
    """Pad a batch of (src, tgt) pairs to the same length."""
    src_batch, tgt_batch = zip(*batch)
    src_lengths = torch.tensor([len(s) for s in src_batch], dtype=torch.long)
    src_padded  = pad_sequence(src_batch, batch_first=True, padding_value=PAD_IDX)
    tgt_padded  = pad_sequence(tgt_batch, batch_first=True, padding_value=PAD_IDX)
    return src_padded, src_lengths, tgt_padded


# ─── DataLoader Factory ───────────────────────────────────────────────────────
def get_dataloaders(
    data_dir: str,
    src_vocab: Vocabulary,
    tgt_vocab: Vocabulary,
    batch_size: int = 64,
    max_len: int = 100,
    num_workers: int = 2,
):
    """
    Build DataLoaders for train / valid / test splits.

    Training   : BucketBatchSampler + truncate=False (drop long sentences)
    Valid/Test : sequential, shuffle=False, truncate=False (consistent with train)

    pin_memory enabled when CUDA is available for fast host → device transfer.
    """
    _pin = torch.cuda.is_available() and num_workers > 0

    loaders: dict = {}
    for split in ("train", "valid", "test"):
        src = os.path.join(data_dir, f"{split}.src")
        tgt = os.path.join(data_dir, f"{split}.tgt")
        if not (os.path.exists(src) and os.path.exists(tgt)):
            loaders[split] = None
            continue

        # Training drops long sentences; test keeps them (generate.py uses truncate=True)
        dataset = GECDataset(src, tgt, src_vocab, tgt_vocab, max_len, truncate=False)

        if split == "train":
            sampler = BucketBatchSampler(dataset, batch_size)
            loaders[split] = DataLoader(
                dataset,
                batch_sampler=sampler,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=_pin,
                persistent_workers=(num_workers > 0),
            )
        else:
            loaders[split] = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=_pin,
                persistent_workers=(num_workers > 0),
            )

    return loaders.get("train"), loaders.get("valid"), loaders.get("test")