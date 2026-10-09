#!/bin/bash
# ──────────────────────────────────────────────────────
# Run ALL experiments for PolyGEC v2
#
# Usage:
#   bash scripts/run_all.sh                                  # Full pipeline, all languages
#   bash scripts/run_all.sh english                          # Single language
#   bash scripts/run_all.sh english transformer 0.5K_1K      # Single experiment
#
# Pipeline per language:
#   1. Preprocess CSV → plain text
#   1b. Train LM recaser for languages with case
#   2. Train BPE tokenizers (8 merge levels)
#   3. Apply BPE (8×8 = 64 configs)
#   4. Train Transformer models (64 configs)
#   5. Generate & evaluate
# ──────────────────────────────────────────────────────

set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJ_DIR"

GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)" 2>/dev/null || echo 0)
GENERATE_ARGS=""
if [ "${GPU_COUNT}" -gt 1 ]; then
    GENERATE_ARGS="--data-parallel"
fi

LANGUAGES=("english" "hindi" "tamil" "bengali" "russian" "turkish")
ARCHS=("transformer")

# 8 BPE merge levels (K-notation)
BPE_LABELS=("0.5K" "1K" "2K" "4K" "8K" "16K" "25K" "32K")

# Build the 8×8 config grid
CONFIGS=()
for SRC in "${BPE_LABELS[@]}"; do
    for TGT in "${BPE_LABELS[@]}"; do
        CONFIGS+=("${SRC}_${TGT}")
    done
done

# Allow filtering by positional args
if [ $# -ge 1 ]; then LANGUAGES=("$1"); fi
if [ $# -ge 2 ]; then ARCHS=("$2"); fi
if [ $# -ge 3 ]; then CONFIGS=("$3"); fi

echo "══════════════════════════════════════════════════"
echo "  PolyGEC v2 — Full Pipeline"
echo "  Languages    : ${LANGUAGES[*]}"
echo "  Architectures: ${ARCHS[*]}"
echo "  Configs      : ${#CONFIGS[@]} total"
echo "  GPUs         : ${GPU_COUNT}"
echo "══════════════════════════════════════════════════"
echo ""

for LANG in "${LANGUAGES[@]}"; do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Language: ${LANG}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    RAW_DIR="data/raw/${LANG}"
    PROC_DIR="data/processed/${LANG}"

    # ── Step 1: Preprocess CSV → text ────────────────────────────────────────
    if [ ! -f "${PROC_DIR}/train.src" ]; then
        echo ""
        echo "── Step 1: Preprocess CSV → text ──"

        TRAIN_CSV=$(find "$RAW_DIR" -name "*train*.csv" 2>/dev/null | head -1)
        if [ -z "$TRAIN_CSV" ]; then
            TRAIN_CSV=$(find "$RAW_DIR" -name "*.csv" 2>/dev/null | head -1)
        fi
        if [ -z "$TRAIN_CSV" ]; then
            echo "ERROR: No CSV found in ${RAW_DIR}/"
            echo "  Place a CSV with 'source'/'src' and 'target'/'tgt' columns there."
            exit 1
        fi
        echo "  Train CSV: $TRAIN_CSV"

        TEST_ARG=""
        TEST_CSV=$(find "$RAW_DIR" -name "*test*.csv" 2>/dev/null | head -1)
        if [ -n "$TEST_CSV" ]; then
            echo "  Test CSV:  $TEST_CSV"
            TEST_ARG="--test-input $TEST_CSV"
        fi

        python scripts/preprocess.py \
            --lang   "$LANG" \
            --input  "$TRAIN_CSV" \
            $TEST_ARG \
            --output "$PROC_DIR"
    else
        echo "  ✓ Preprocessed data: ${PROC_DIR}/"
    fi

    # ── Step 1b: Train LM recaser for languages with case ────────────────────
    # Hindi, Tamil, and Bengali do not use uppercase/lowercase distinctions.
    # English, Russian, and Turkish do, so we train the recaser there.
    if [[ "$LANG" == "english" || "$LANG" == "russian" || "$LANG" == "turkish" ]]; then
        RECASER_DIR="tokenizers/recaser/${LANG}"
        if [ ! -f "${RECASER_DIR}/lm.pkl" ] || \
           [ ! -f "${RECASER_DIR}/casing_variants.json" ] || \
           [ ! -f "${RECASER_DIR}/config.json" ]; then
            echo ""
            echo "── Step 1b: Train LM recaser for ${LANG} ──"
            TRAIN_CSV=$(find "$RAW_DIR" -name "*train*.csv" 2>/dev/null | head -1)
            if [ -z "$TRAIN_CSV" ]; then
                TRAIN_CSV=$(find "$RAW_DIR" -name "*.csv" 2>/dev/null | head -1)
            fi
            if [ -n "$TRAIN_CSV" ]; then
                python scripts/train_recaser.py \
                    --lang        "$LANG" \
                    --input       "$TRAIN_CSV"
            else
                echo "  WARNING: No CSV found for recaser training — skipping"
            fi
        else
            echo "  ✓ LM recaser: ${RECASER_DIR}/"
        fi
    fi

    # ── Step 2: Train BPE (8 merge levels) ───────────────────────────────────
    # Check if the first BPE codes file exists as a proxy
    if [ ! -f "tokenizers/models/${LANG}/bpe_0.5K.codes" ]; then
        echo ""
        echo "── Step 2: Train BPE tokenizers (8 levels) ──"
        python tokenizers/train_bpe.py --lang "$LANG" --data "$PROC_DIR"
    else
        echo "  ✓ BPE models: tokenizers/models/${LANG}/"
    fi

    # ── Step 3: Apply BPE (8×8 = 64 configs) ────────────────────────────────
    if [ ! -f "data/tokenized/${LANG}/0.5K_0.5K/train.src" ]; then
        echo ""
        echo "── Step 3: Apply BPE (64 configs) ──"
        bash scripts/apply_bpe.sh "$LANG"
    else
        echo "  ✓ Tokenized data: data/tokenized/${LANG}/"
    fi

    # ── Steps 4 & 5: Train → Generate → Evaluate ─────────────────────────────
    for ARCH in "${ARCHS[@]}"; do
        for CONFIG in "${CONFIGS[@]}"; do
            CKPT="checkpoints/${LANG}/${ARCH}_${CONFIG}/best.pt"

            # ── Train ────────────────────────────────────────────────────────
            if [ ! -f "$CKPT" ]; then
                echo ""
                echo "── Step 4: Train ${ARCH} / ${CONFIG} ──"
                python scripts/train.py \
                    --lang   "$LANG" \
                    --arch   "$ARCH" \
                    --config "$CONFIG"
            else
                echo "  ✓ Trained: ${CKPT}"
            fi

            HYP="results/${LANG}_${ARCH}_${CONFIG}.hyp"
            EVAL="results/${LANG}_${ARCH}_${CONFIG}.eval"

            # ── Generate & Evaluate ──────────────────────────────────────────
            if [ ! -f "$HYP" ]; then
                echo ""
                echo "── Step 5: Generate & Evaluate ${ARCH} / ${CONFIG} ──"

                BEAM_ARG=""

                python scripts/generate.py \
                    --lang       "$LANG" \
                    --arch       "$ARCH" \
                    --config     "$CONFIG" \
                    $BEAM_ARG \
                    $GENERATE_ARGS

                python scripts/evaluate.py \
                    --hyp    "${HYP}" \
                    --ref    "results/${LANG}_${ARCH}_${CONFIG}.ref" \
                    --src    "results/${LANG}_${ARCH}_${CONFIG}.src" \
                    --output "${EVAL}"
            else
                echo "  ✓ Results: ${HYP}"
                if [ ! -f "$EVAL" ]; then
                    echo "  Re-evaluating (eval file missing)..."
                    python scripts/evaluate.py \
                        --hyp    "${HYP}" \
                        --ref    "results/${LANG}_${ARCH}_${CONFIG}.ref" \
                        --src    "results/${LANG}_${ARCH}_${CONFIG}.src" \
                        --output "${EVAL}"
                fi
            fi
        done
    done

    echo ""
done

echo "══════════════════════════════════════════════════"
echo "  All experiments complete!"
echo "  Results: results/"
echo "══════════════════════════════════════════════════"
