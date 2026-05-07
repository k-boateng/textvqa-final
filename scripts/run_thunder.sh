#!/bin/bash
# Thunder Compute pipeline runner
# Usage: bash scripts/run_thunder.sh [smoke|full]
# - smoke: runs 50-example smoke tests of all 5 jobs (~10 min)
# - full:  runs the full pipeline (~10 hours)

set -e  # exit on error

MODE=${1:-full}

# Disable offloading on Thunder A100 80GB — we have plenty of VRAM
export OFFLOAD_FOLDER=""

# Ensure output dirs exist
mkdir -p outputs/predictions outputs/metrics outputs/offload

echo "=========================================="
echo "Thunder pipeline mode: $MODE"
echo "OFFLOAD_FOLDER='$OFFLOAD_FOLDER' (empty = disabled)"
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=========================================="

if [ "$MODE" = "smoke" ]; then
    LIMIT_FLAG="--limit 50"
    SUFFIX="(SMOKE TEST, 50 examples each)"
else
    LIMIT_FLAG=""
    SUFFIX="(FULL RUN)"
fi

echo ""
echo "Job 1/5: Qwen naive no OCR n=5000 $SUFFIX"
echo "------------------------------------------"
if [ "$MODE" = "smoke" ]; then
    # 13_full_val_eval.py doesn't have --limit; we use a smaller val_size for smoke
    python scripts/13_full_val_eval.py --setting qwen_naive_no_ocr --val_size 50
else
    python scripts/13_full_val_eval.py --setting qwen_naive_no_ocr --val_size 5000
fi

echo ""
echo "Job 2/5: Qwen constrained + OCR n=5000 $SUFFIX"
echo "------------------------------------------"
if [ "$MODE" = "smoke" ]; then
    python scripts/13_full_val_eval.py --setting qwen_constrained_with_ocr --val_size 50
else
    python scripts/13_full_val_eval.py --setting qwen_constrained_with_ocr --val_size 5000
fi

echo ""
echo "Job 3/5: Qwen LoRA-10k n=5000 $SUFFIX"
echo "------------------------------------------"
if [ "$MODE" = "smoke" ]; then
    python scripts/13_full_val_eval.py --setting qwen_lora10k_constrained_with_ocr --val_size 50
else
    python scripts/13_full_val_eval.py --setting qwen_lora10k_constrained_with_ocr --val_size 5000
fi

echo ""
echo "Job 4/5: LLaVA naive no OCR n=2000 $SUFFIX"
echo "------------------------------------------"
python scripts/14_eval_llava.py --val_size 2000 --prompt_type naive $LIMIT_FLAG

echo ""
echo "Job 5/5: LLaVA constrained + OCR n=2000 $SUFFIX"
echo "------------------------------------------"
python scripts/14_eval_llava.py --val_size 2000 --prompt_type constrained --use_ocr $LIMIT_FLAG

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE"
echo "=========================================="
ls -la outputs/metrics/