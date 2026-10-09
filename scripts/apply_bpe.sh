#!/bin/bash
# ──────────────────────────────────────────────────────
# Apply BPE tokenization for all 8×8 source→target combos
#
# Usage:
#   bash scripts/apply_bpe.sh <lang>
#   bash scripts/apply_bpe.sh english
#
# Expects:
#   - Processed text files in data/processed/<lang>/
#   - Trained BPE codes in tokenizers/models/<lang>/
#     Files: bpe_0.5K.codes, bpe_1K.codes, ..., bpe_32K.codes
#
# Produces:
#   - data/tokenized/<lang>/<src_label>_<tgt_label>/train.src, train.tgt, ...
#     for each of the 64 (8×8) config combinations
# ──────────────────────────────────────────────────────

set -euo pipefail

LANG=$1

DATA_DIR="data/processed/${LANG}"
BPE_DIR="tokenizers/models/${LANG}"
OUT_BASE="data/tokenized/${LANG}"

# 8 merge levels with K-notation labels
LABELS=("0.5K" "1K" "2K" "4K" "8K" "16K" "25K" "32K")

# Verify all BPE codes exist
for LABEL in "${LABELS[@]}"; do
    CODES="${BPE_DIR}/bpe_${LABEL}.codes"
    if [ ! -f "$CODES" ]; then
        echo "ERROR: BPE codes not found: $CODES"
        echo "  Run: python tokenizers/train_bpe.py --lang $LANG --data $DATA_DIR"
        exit 1
    fi
done

echo "=== Applying BPE for ${LANG} (8×8 = 64 configs) ==="

COUNT=0
TOTAL=$(( ${#LABELS[@]} * ${#LABELS[@]} ))

for SRC_LABEL in "${LABELS[@]}"; do
    for TGT_LABEL in "${LABELS[@]}"; do
        CONFIG="${SRC_LABEL}_${TGT_LABEL}"
        COUNT=$((COUNT + 1))

        echo "[${COUNT}/${TOTAL}] ${LANG} / ${CONFIG}"
        OUT_DIR="${OUT_BASE}/${CONFIG}"
        mkdir -p "$OUT_DIR"

        src_codes="${BPE_DIR}/bpe_${SRC_LABEL}.codes"
        tgt_codes="${BPE_DIR}/bpe_${TGT_LABEL}.codes"

        for SPLIT in train valid test; do
            # Apply BPE to source
            if [ -f "${DATA_DIR}/${SPLIT}.src" ]; then
                subword-nmt apply-bpe -c "$src_codes" \
                    < "${DATA_DIR}/${SPLIT}.src" \
                    > "${OUT_DIR}/${SPLIT}.src"
            fi

            # Apply BPE to target
            if [ -f "${DATA_DIR}/${SPLIT}.tgt" ]; then
                subword-nmt apply-bpe -c "$tgt_codes" \
                    < "${DATA_DIR}/${SPLIT}.tgt" \
                    > "${OUT_DIR}/${SPLIT}.tgt"
            fi
        done
        echo "  ✓ ${CONFIG}"
    done
done

echo ""
echo "All BPE tokenization done for ${LANG}! (${TOTAL} configs)"
echo "Output: ${OUT_BASE}/<src>_<tgt>/"
