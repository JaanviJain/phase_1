#!/usr/bin/env python3
"""
================================================================================
MASTER EXPERIMENT RUNNER
Place in phase1/scripts/ and run:  python scripts/run_experiments.py
================================================================================
Automatically runs all Phase 1 experiments in sequence:
  1. Frozen baseline (concat, 5 epochs)
  2. Unfrozen concat (15 epochs)
  3. Unfrozen cross_attn (15 epochs)  [skips if OOM]
  4. Unfrozen bilinear (15 epochs)    [skips if OOM]
Each experiment gets its own checkpoint folder.
Results are compared at the end.
================================================================================
"""
import os
import sys
import subprocess
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def banner(text):
    print("\n" + "=" * 75)
    print(text)
    print("=" * 75)


def run_experiment(name, desc, extra_args, ckpt_subdir):
    """
    Run one training experiment.
    Returns (success: bool, metrics: dict or None)
    """
    banner(f"EXPERIMENT: {name}")
    print(f"Description: {desc}")
    
    ckpt_dir = PROJECT_ROOT / "checkpoints" / ckpt_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "src/train.py",
        "--checkpoint_dir", str(ckpt_dir),
    ] + extra_args
    
    print(f"Command: {' '.join(cmd)}")
    print(f"Checkpoints will be saved to: {ckpt_dir}")
    print("Starting training... (this may take 30-90 minutes)\n")
    
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
        if result.returncode != 0:
            print(f"\n⚠️  Experiment {name} exited with code {result.returncode}")
            return False, None
    except Exception as e:
        print(f"\n❌ Experiment {name} failed with exception: {e}")
        return False, None
    
    # Try to load results
    results_file = None
    for f in ckpt_dir.glob("test_results_*.json"):
        results_file = f
        break
    
    if results_file and results_file.exists():
        with open(results_file) as f:
            metrics = json.load(f)
        return True, metrics
    else:
        print(f"⚠️  No results JSON found in {ckpt_dir}")
        return True, None


def print_comparison_table(all_results):
    """Print a nice comparison table of all experiments."""
    banner("EXPERIMENT COMPARISON TABLE")
    
    print(f"\n{'Experiment':<25} {'Test Acc':>10} {'Test F1':>10} {'Test AUC':>10} {'Status':>10}")
    print("-" * 75)
    
    best_f1 = -1.0
    best_name = ""
    
    for name, (success, metrics) in all_results.items():
        if not success:
            print(f"{name:<25} {'--':>10} {'--':>10} {'--':>10} {'FAILED':>10}")
            continue
        
        if metrics is None:
            print(f"{name:<25} {'--':>10} {'--':>10} {'--':>10} {'NO DATA':>10}")
            continue
        
        tm = metrics.get("test_metrics", {})
        acc = tm.get("accuracy", 0.0)
        f1 = tm.get("f1", 0.0)
        auc = tm.get("auc", 0.0)
        
        print(f"{name:<25} {acc:>10.4f} {f1:>10.4f} {auc:>10.4f} {'OK':>10}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_name = name
    
    print("-" * 75)
    print(f"\n🏆 BEST EXPERIMENT: {best_name} (Macro-F1 = {best_f1:.4f})")
    print(f"\n💡 Use the checkpoint from: checkpoints/{best_name.replace(' ', '_').lower()}/")
    
    # Save master comparison
    summary = {
        "timestamp": datetime.now().isoformat(),
        "experiments": {
            name: {
                "success": success,
                "test_metrics": metrics.get("test_metrics", {}) if metrics else {}
            }
            for name, (success, metrics) in all_results.items()
        },
        "best_experiment": best_name,
        "best_f1": best_f1,
    }
    summary_path = PROJECT_ROOT / "checkpoints" / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to: {summary_path}")


def main():
    print("""
     ██████╗ ██╗  ██╗ █████╗ ███████╗███████╗     ██╗
     ██╔══██╗██║  ██║██╔══██╗██╔════╝██╔════╝    ███║
     ██████╔╝███████║███████║█████╗  █████╗      ╚██║
     ██╔═══╝ ██╔══██║██╔══██║██╔══╝  ██╔══╝       ██║
     ██║     ██║  ██║██║  ██║██║     ███████╗     ██║
     ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚══════╝     ╚═╝
    """)
    
    print("This script will run 4 experiments sequentially.")
    print("Estimated total time: 3-6 hours depending on your GPU.")
    print("If cross_attn runs out of VRAM, it will be skipped automatically.")
    print("\nMake sure you have already run: python scripts/preprocess.py --skip-missing")
    
    # Check if preprocessed data exists
    if not (PROJECT_ROOT / "data" / "processed" / "train_multimodal.csv").exists():
        print("\n❌ Preprocessed data not found!")
        print("   Run this first: python scripts/preprocess.py --skip-missing")
        sys.exit(1)
    
    all_results = {}
    
    # =========================================================================
    # EXPERIMENT 1: Frozen Baseline (THE EXACT STEPS - Step 1)
    # =========================================================================
    all_results["frozen_concat_baseline"] = run_experiment(
        name="frozen_concat_baseline",
        desc="STEP 1: Frozen backbones, concat fusion, 5 epochs. Anchor baseline.",
        extra_args=["--mode", "binary", "--epochs", "5", "--freeze_backbones", "--fusion", "concat"],
        ckpt_subdir="exp01_frozen_concat_baseline"
    )
    
    # =========================================================================
    # EXPERIMENT 2: Unfrozen Concat (THE EXACT STEPS - Step 2)
    # =========================================================================
    all_results["unfrozen_concat"] = run_experiment(
        name="unfrozen_concat",
        desc="STEP 2: Unfrozen top 4 BioBERT + top 2 ViT, concat, 15 epochs.",
        extra_args=["--mode", "binary", "--epochs", "15", "--unfreeze_biobert", "4", "--unfreeze_vit", "2", "--fusion", "concat"],
        ckpt_subdir="exp02_unfrozen_concat"
    )
    
    # =========================================================================
    # EXPERIMENT 3: Unfrozen Cross-Attention (THE EXACT STEPS - Step 3)
    # =========================================================================
    print("\n⚠️  WARNING: cross_attn requires ~12-14 GB VRAM.")
    print("   If you have 8 GB, this will likely OOM and be skipped.")
    input("   Press ENTER to continue (or Ctrl+C to skip cross_attn)...")
    
    all_results["unfrozen_cross_attn"] = run_experiment(
        name="unfrozen_cross_attn",
        desc="STEP 3: Unfrozen, cross-attention fusion. VRAM intensive.",
        extra_args=["--mode", "binary", "--epochs", "15", "--unfreeze_biobert", "4", "--unfreeze_vit", "2", "--fusion", "cross_attn"],
        ckpt_subdir="exp03_unfrozen_cross_attn"
    )
    
    # =========================================================================
    # EXPERIMENT 4: Unfrozen Bilinear (THE EXACT STEPS - Step 4 extension)
    # =========================================================================
    all_results["unfrozen_bilinear"] = run_experiment(
        name="unfrozen_bilinear",
        desc="STEP 4: Unfrozen, bilinear fusion. Low-rank alternative.",
        extra_args=["--mode", "binary", "--epochs", "15", "--unfreeze_biobert", "4", "--unfreeze_vit", "2", "--fusion", "bilinear"],
        ckpt_subdir="exp04_unfrozen_bilinear"
    )
    
    # =========================================================================
    # FINAL COMPARISON
    # =========================================================================
    print_comparison_table(all_results)
    
    banner("ALL EXPERIMENTS COMPLETE")
    print("""
NEXT STEPS:
-----------
1. Check checkpoints/expXX_*/ for saved models
2. The best experiment's fusion weights are in:
      checkpoints/<best_experiment>/best_fusion_*.pt
3. Copy that file to your Phase 4 project folder.
4. In your thesis, report the comparison table from above.
    """)


if __name__ == "__main__":
    main()