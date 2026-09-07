#!/bin/bash
cd "$(dirname "$0")/.."
echo "=========================================="
echo "Phase 1: Multimodal Encoder Training"
echo "=========================================="

echo ""
echo "[1/3] Downloading missing images..."
python scripts/download_images.py --dataset all

echo ""
echo "[2/3] Preprocessing datasets..."
python scripts/preprocess.py --skip-missing

echo ""
echo "[3/3] Starting training..."
# RECOMMENDED: Unfrozen top layers, 15 epochs, binary mode
python src/train.py --mode binary --epochs 15 --unfreeze_biobert 4 --unfreeze_vit 2 --fusion concat

echo ""
echo "Training complete! Check checkpoints/ for saved models."