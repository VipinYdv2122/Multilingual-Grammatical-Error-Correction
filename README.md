# PolyGEC — Multilingual Grammatical Error Correction

A systematic study of the interaction between **neural architectures** and **BPE tokenization granularity** for Grammatical Error Correction (GEC) across English and Indic languages.

> **Authors:** Ishaan Giri (2025201091), Aditya Narayan Dwivedi (2025201026), Vipin Yadav (2025201013)

---

## Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Workflow](#workflow)
  - [Step 1 — Preprocess Raw CSV Data](#step-1--preprocess-raw-csv-data)
  - [Step 2 — Train BPE Tokenizers](#step-2--train-bpe-tokenizers)
  - [Step 3 — Apply BPE Tokenization](#step-3--apply-bpe-tokenization)
  - [Step 4 — Train Models](#step-4--train-models)
  - [Step 5 — Generate Corrections](#step-5--generate-corrections)
  - [Step 6 — Evaluate](#step-6--evaluate)
  - [Run Everything at Once](#run-everything-at-once)
- [Interactive Correction](#interactive-correction)
- [Experimental Design](#experimental-design)
- [Evaluation Metrics](#evaluation-metrics)
- [Results Summary](#results-summary)
- [Configuration Reference](#configuration-reference)

---

## Overview

PolyGEC implements a **3 × 4 factorial experiment** comparing three neural sequence-to-sequence architectures under four BPE tokenization configurations, evaluated across four languages:

| Dimension       | Options                                                                       |
|-----------------|-------------------------------------------------------------------------------|
| **Architectures** | RNN+Attention (GRU), LSTM+Attention, Small-Scale Transformer                |
| **Tokenization**  | H2H (High→High), H2L (High→Low), L2H (Low→High), L2L (Low→Low)            |
| **Languages**     | English, Hindi, Tamil, Bengali                                               |

This yields up to **48 experiments** (3 architectures × 4 configs × 4 languages), all built from scratch in PyTorch with no external NLP framework dependencies (no fairseq, no HuggingFace models).

---

## Key Findings

1. **LSTM consistently outperforms RNN** by 1–15 pp F₀.₅ across all languages and configurations.
2. **BPE→BPE decoding fails catastrophically for Indic scripts** — Tamil drops to GLEU ≈ 14 and Bengali to GLEU ≈ 15, confirming that sub-word decoding cannot reconstruct valid surface forms in morphologically complex, non-Latin scripts.
3. **Optimal encoding is language-dependent:**
   - English → **BPE→Word** is best (F₀.₅ = 73.16%)
   - Hindi → **Word→Word** is best (F₀.₅ = 96.05%)
   - Tamil → **Word→Word** is best (F₀.₅ = 83.91%)
   - Bengali → **Word→Word** is best (F₀.₅ = 83.82%)

---

## Architecture

### RNN+Attention (GRU)
- **Encoder:** 2-layer bidirectional GRU
- **Decoder:** 1-layer unidirectional GRU with Bahdanau (additive) attention
- Context vector concatenated with decoder input at each step

### LSTM+Attention
- Same structure as RNN, but with LSTM cells (hidden + cell state)
- Better long-range dependency modeling via gating mechanisms

### Small-Scale Transformer
- **Encoder:** 2 layers, 4 attention heads, d_model=256, d_ff=512
- **Decoder:** 2 layers with masked self-attention + cross-attention
- Sinusoidal positional encodings
- Uses PyTorch's `nn.TransformerEncoder` / `nn.TransformerDecoder`

All architectures share a common interface: `forward()` for training (with teacher forcing for seq2seq) and `generate()` for greedy inference.

---

## Project Structure

```
projectv2/
├── configs/
│   └── experiment_configs.yaml    # All hyperparameters and experiment grid
├── data/
│   ├── raw/<lang>/                # Raw CSV files (source, target columns)
│   ├── processed/<lang>/          # Plain-text train/valid/test .src/.tgt files  (Step 1 output)
│   ├── tokenized/<lang>/<config>/ # BPE-tokenized .src/.tgt files               (Step 3 output)
│   └── vocabs/<lang>/<config>/    # Token-to-index vocab files                   (Step 4 output)
├── tokenizers/
│   ├── train_bpe.py               # Train High-merge (16K) & Low-merge (3K) BPE
│   └── models/<lang>/             # Saved BPE codes (.codes files)               (Step 2 output)
├── models/
│   ├── __init__.py
│   ├── rnn_attention.py           # GRU encoder-decoder with Bahdanau attention
│   ├── lstm_attention.py          # LSTM encoder-decoder with Bahdanau attention
│   ├── transformer.py             # Small-scale Transformer encoder-decoder
│   └── dataset.py                 # Vocabulary, GECDataset, DataLoader utilities
├── scripts/
│   ├── preprocess.py              # CSV → plain-text (Unicode NFC, split)
│   ├── apply_bpe.sh               # Apply BPE codes for all 4 configs
│   ├── train.py                   # Training loop (early stopping, LR scheduler)
│   ├── generate.py                # Greedy decoding on test set
│   ├── evaluate.py                # GEC GLEU, BLEU, Token Accuracy, ERRANT F₀.₅
│   ├── correct.py                 # Interactive single-sentence correction
│   └── run_all.sh                 # End-to-end pipeline automation
├── checkpoints/<lang>/<arch>_<config>/  # Saved model checkpoints (best.pt)
├── results/                       # Evaluation output (.hyp, .ref, .src, .eval)
├── requirements.txt
├── report.txt                     # Full research report
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended; CPU training is possible but slow)

### Installation

```bash
# Clone the repository
git clone https://github.com/Ishaan-Giri/PolyGEC.git
cd PolyGEC

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**

| Package       | Purpose                  |
|---------------|--------------------------|
| `torch>=2.0`  | Core deep learning       |
| `subword-nmt` | BPE tokenization         |
| `pandas`      | CSV data handling        |
| `pyyaml`      | Config file parsing      |

> **Note:** Install PyTorch with the appropriate CUDA version for your system from [pytorch.org](https://pytorch.org/get-started/locally/).

### Data Preparation

Place your raw CSV files under `data/raw/<lang>/`. Each CSV must have columns named `source` (or `src`) and `target` (or `tgt`/`trg`):

```
data/raw/english/train.csv
data/raw/hindi/train.csv
data/raw/tamil/train.csv
data/raw/bengali/train.csv
```

---

## Workflow

The pipeline has **6 sequential steps**. Each step can be run individually or orchestrated via `run_all.sh`.

### Step 1 — Preprocess Raw CSV Data

Reads CSV files, applies Unicode NFC normalization, collapses whitespace, and splits into train/valid/test:

```bash
python scripts/preprocess.py \
    --lang english \
    --input data/raw/english/train.csv \
    --output data/processed/english/
```

**Options:**
| Flag              | Default | Description                                 |
|-------------------|---------|---------------------------------------------|
| `--train-ratio`   | 0.90    | Fraction of data for training               |
| `--valid-ratio`   | 0.05    | Fraction for validation                     |
| `--test-input`    | None    | Optional separate test CSV                  |
| `--seed`          | 42      | Random seed for reproducible splits         |

**Output:** `data/processed/<lang>/train.src`, `train.tgt`, `valid.src`, `valid.tgt`, `test.src`, `test.tgt`

---

### Step 2 — Train BPE Tokenizers

Trains two BPE models per language with different merge counts:

```bash
python tokenizers/train_bpe.py --lang english --data data/processed/english
```

| BPE Model   | Merges | Effect                                |
|-------------|--------|---------------------------------------|
| `bpe_high`  | 16,000 | Larger vocab, less aggressive splitting |
| `bpe_low`   | 3,000  | Smaller vocab, more aggressive splitting |

**Output:** `tokenizers/models/<lang>/bpe_high.codes`, `bpe_low.codes`

> **Sanity check:** Add `--test-mode` to verify no `<unk>` tokens are produced.

---

### Step 3 — Apply BPE Tokenization

Applies the trained BPE codes to create 4 tokenization configurations per language:

```bash
bash scripts/apply_bpe.sh english
```

| Config | Source BPE  | Target BPE  | Description                    |
|--------|-------------|-------------|--------------------------------|
| `H2H`  | High (16K)  | High (16K)  | High-merge → High-merge        |
| `H2L`  | High (16K)  | Low (3K)    | High-merge → Low-merge          |
| `L2H`  | Low (3K)    | High (16K)  | Low-merge → High-merge          |
| `L2L`  | Low (3K)    | Low (3K)    | Low-merge → Low-merge           |

**Output:** `data/tokenized/<lang>/<config>/train.src`, `train.tgt`, `valid.src`, `valid.tgt`, `test.src`, `test.tgt`

---

### Step 4 — Train Models

Train a specific architecture + tokenization configuration:

```bash
python scripts/train.py --lang english --arch lstm --config H2H
```

**Architecture choices:** `rnn`, `lstm`, `transformer`

**Key hyperparameters (with defaults):**

| Flag                 | Default | Description                   |
|----------------------|---------|-------------------------------|
| `--embed-dim`        | 256     | Embedding dimension           |
| `--hidden-dim`       | 512     | RNN/LSTM hidden size          |
| `--n-layers`         | 2       | Encoder layers                |
| `--nhead`            | 4       | Transformer attention heads   |
| `--ffn-dim`          | 512     | Transformer FFN dimension     |
| `--dropout`          | 0.3     | Dropout rate                  |
| `--lr`               | 0.001   | Initial learning rate (Adam)  |
| `--batch-size`       | 64      | Batch size                    |
| `--epochs`           | 30      | Max training epochs           |
| `--patience`         | 5       | Early stopping patience       |
| `--teacher-forcing`  | 0.5     | Teacher forcing ratio         |
| `--max-len`          | 100     | Max sequence length           |

**Features:**
- ReduceLROnPlateau scheduler (factor=0.5, patience=2)
- Gradient clipping (max norm = 1.0)
- Early stopping based on validation loss
- Saves best checkpoint to `checkpoints/<lang>/<arch>_<config>/best.pt`

---

### Step 5 — Generate Corrections

Run greedy decoding on the test set using a trained checkpoint:

```bash
python scripts/generate.py --lang english --arch lstm --config H2H
```

**Output:** `results/<lang>_<arch>_<config>.hyp` (hypotheses), `.ref` (references), `.src` (sources)

BPE markers (`@@`) are automatically removed from output files. If an LM
recaser exists under `tokenizers/recaser/<lang>/`, hypotheses and source
sentences are automatically recased before evaluation output is written.

The recaser uses a Kneser-Ney LM plus a casing-variant table and decodes with
a small beam search, which is closer to a classic Moses-style statistical
truecaser than the earlier greedy fallback.

---

### Step 6 — Evaluate

Compute all evaluation metrics:

```bash
python scripts/evaluate.py \
    --hyp results/english_lstm_H2H.hyp \
    --ref results/english_lstm_H2H.ref \
    --src results/english_lstm_H2H.src \
    --output results/english_lstm_H2H.eval
```

**Output:** Prints metrics to console and optionally saves to `.eval` file.

---

### Run Everything at Once

The `run_all.sh` script orchestrates the full pipeline with skip-if-done logic:

```bash
# All languages, all architectures, all configs
bash scripts/run_all.sh

# Single language
bash scripts/run_all.sh english

# Single experiment
bash scripts/run_all.sh english lstm H2H
```

Steps already completed (detected via output file existence) are automatically skipped.
For languages with case (`english`, `russian`, `turkish`), the script also trains
the LM recaser before generation when needed.
On multi-GPU machines such as Kaggle T4x2, training already uses both GPUs and
`run_all.sh` now enables multi-GPU greedy generation as well.

---

## Interactive Correction

After training a model, use it interactively to correct sentences:

```bash
python scripts/correct.py --lang english --arch lstm --config H2H
```

```
Loading model: english / lstm / H2H
Ready! Type a sentence to correct (or 'quit' to exit).

Input:  I goes to school yesterday
Output: I went to school yesterday
```

The script automatically applies the appropriate BPE tokenization to your input based on the config (H → High-merge BPE, L → Low-merge BPE).

---

## Experimental Design

```
                  ┌──────────┐
                  │  Raw CSV │
                  └────┬─────┘
                       │ preprocess.py
                  ┌────▼─────┐
                  │ .src/.tgt│ (plain text)
                  └────┬─────┘
                       │ train_bpe.py
              ┌────────▼────────┐
              │ BPE High + Low  │
              └────────┬────────┘
                       │ apply_bpe.sh
     ┌──────────────────▼──────────────────┐
     │  H2H  │  H2L  │  L2H  │  L2L       │
     └──┬───────┬───────┬───────┬──────────┘
        │ train.py (× 3 architectures each)
        ▼
     ┌──────────────────────────────────────┐
     │  16 models per language              │
     │  (4 archs × 4 configs)              │
     └──┬───────────────────────────────────┘
       │ generate.py → evaluate.py
       ▼
    ┌──────────────────────────────────────┐
    │  Results: GLEU, BLEU, Acc, F₀.₅     │
    └──────────────────────────────────────┘
```

---

## Evaluation Metrics

| Metric             | Description                                                                                          |
|--------------------|------------------------------------------------------------------------------------------------------|
| **GLEU**           | Napoles et al. GEC-specific source-aware n-gram metric                                               |
| **Corpus BLEU**    | Standard corpus-level BLEU with brevity penalty                                                      |
| **Token Accuracy** | Position-wise token match percentage                                                                 |
| **Precision**      | ERRANT edit precision: fraction of system edits that are correct                                    |
| **Recall**         | ERRANT edit recall: fraction of gold edits found by the system                                      |
| **F₀.₅**          | Weighted harmonic mean (precision-heavy, β=0.5)                                                      |

---

## Results Summary

### English (Best: LSTM + BPE→Word)

| Model | Config     | GLEU  | BLEU  | F₀.₅   | Latency |
|-------|------------|-------|-------|---------|---------|
| LSTM  | BPE→Word   | 43.20 | 46.78 | 73.16%  | 8.49 ms |
| RNN   | BPE→Word   | 36.34 | 35.62 | 63.91%  | 8.46 ms |

### Hindi (Best: LSTM + Word→Word)

| Model | Config      | GLEU  | BLEU  | F₀.₅   | Latency |
|-------|-------------|-------|-------|---------|---------|
| LSTM  | Word→Word   | 78.91 | 80.36 | 96.05%  | 1.79 ms |
| RNN   | Word→Word   | 76.03 | 77.89 | 94.86%  | 2.86 ms |

### Tamil (Best: LSTM + Word→Word)

| Model | Config      | GLEU  | BLEU  | F₀.₅   |
|-------|-------------|-------|-------|---------|
| LSTM  | Word→Word   | 41.02 | 41.89 | 83.91%  |
| LSTM  | BPE→BPE     | 14.22 | 10.91 | 25.38%  |

### Bengali (Best: LSTM + Word→Word)

| Model | Config      | GLEU  | BLEU  | F₀.₅   |
|-------|-------------|-------|-------|---------|
| LSTM  | Word→Word   | 42.30 | 44.28 | 83.82%  |
| LSTM  | BPE→BPE     | 15.16 | 12.50 | 27.36%  |

---

## Configuration Reference

All experiment parameters are defined in [`configs/experiment_configs.yaml`](configs/experiment_configs.yaml):

```yaml
# Architecture-specific
arch_params:
  lstm:
    encoder_embed_dim: 256
    encoder_hidden_size: 512
    encoder_layers: 2 (bidirectional)
    decoder_layers: 1
  transformer:
    encoder/decoder_layers: 2
    embed_dim: 256
    ffn_dim: 512
    attention_heads: 4

# Training (shared)
training:
  optimizer: Adam
  learning_rate: 0.001
  dropout: 0.3
  batch_size: 64 (effective via max_tokens: 4096)
  max_epoch: 30
  patience: 5
  seed: 42

# BPE
bpe:
  high_merges: 16000
  low_merges: 3000
```

---

## License

This project was developed as part of the INLP course at IIIT Hyderabad.
