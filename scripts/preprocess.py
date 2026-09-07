#!/usr/bin/env python3
"""Preprocess datasets for Phase 1 — FIXED VERSION."""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import load_config


def standardize_honest_ooc(df: pd.DataFrame, cap_per_label: int = None) -> pd.DataFrame:
    """Honest OOC: label 1 = true caption (REAL), label 2 = falsified caption (FAKE)."""
    img_col = "image_path" if "image_path" in df.columns else "image"
    cap_col = "caption" if "caption" in df.columns else "claim_text"
    
    df = df.copy()
    df["image_path"] = df[img_col].astype(str)
    df["claim_text"] = df[cap_col].astype(str)
    df["label"] = df["label"].astype(int)
    
    # CRITICAL FIX: Map 1=true/REAL→0, 2=falsified/FAKE→1
    # If your convention is reversed, change this. But BOTH labels MUST be present.
    original_counts = df["label"].value_counts().to_dict()
    df["label"] = df["label"].map({1: 0, 2: 1})
    print(f"  Original labels: {original_counts}")
    print(f"  Mapped to: 0=Real, 1=Fake | {df['label'].value_counts().to_dict()}")
    
    df["source_dataset"] = "honest_ooc"
    
    # CRITICAL: group_id = image_path because both captions share the same image
    df["group_id"] = df["image_path"]
    
    if cap_per_label is not None:
        before = len(df)
        df = df.groupby("label", group_keys=False).apply(
            lambda x: x.sample(n=min(len(x), cap_per_label), random_state=42)
        ).reset_index(drop=True)
        after = len(df)
        print(f"  ⚠️  WARNING: cap_per_label={cap_per_label} active!")
        print(f"      You threw away {before - after} rows. Set to null in config.yaml to keep all.")
    else:
        print(f"  Using ALL {len(df)} rows (no capping).")
        
    return df[["claim_text", "image_path", "label", "source_dataset", "group_id"]]


def standardize_fakehealth(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_col = None
    for c in ["content", "claim_text", "title", "text"]:
        if c in df.columns:
            text_col = c
            break
    img_col = None
    for c in ["image_path", "image_url", "img_path", "image"]:
        if c in df.columns:
            img_col = c
            break
    rating_col = None
    for c in ["rating", "label", "verdict", "is_fake"]:
        if c in df.columns:
            rating_col = c
            break
    if text_col is None or img_col is None or rating_col is None:
        print(f"  Warning: Auto-detect fallback. Columns: {list(df.columns)}")
        text_col = text_col or df.columns[0]
        img_col = img_col or df.columns[1]
        rating_col = rating_col or df.columns[2]
        
    df["claim_text"] = df[text_col].astype(str)
    df["image_path"] = df[img_col].astype(str)
    
    unique_ratings = df[rating_col].unique()
    if set(unique_ratings).issubset({0, 1}):
        df["label"] = df[rating_col].astype(int)
    else:
        print(f"  ⚠️  FakeHealth ratings are not 0/1: {unique_ratings}")
        print(f"      Using median split. VERIFY THIS IS CORRECT.")
        df["label"] = (df[rating_col] > df[rating_col].median()).astype(int)
        
    # Try to find an article ID for grouping, else fallback to image_path
    id_col = None
    for c in ["review_id", "article_id", "news_id", "id", "content_id", "url"]:
        if c in df.columns:
            id_col = c
            break
    df["group_id"] = df[id_col].astype(str) if id_col else df["image_path"]
    
    df["source_dataset"] = "fakehealth"
    return df[["claim_text", "image_path", "label", "source_dataset", "group_id"]]


def standardize_recovery(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_col = None
    for c in ["title", "claim_text", "content", "text", "news_title"]:
        if c in df.columns:
            text_col = c
            break
    img_col = None
    for c in ["image_path", "image_url", "img_path", "image"]:
        if c in df.columns:
            img_col = c
            break
    label_col = None
    for c in ["reliability", "label", "rating", "is_reliable"]:
        if c in df.columns:
            label_col = c
            break
    if text_col is None:
        text_col = df.columns[0]
    if img_col is None:
        img_col = df.columns[1]
    if label_col is None:
        label_col = df.columns[2]
        
    df["claim_text"] = df[text_col].astype(str)
    df["image_path"] = df[img_col].astype(str)
    
    unique_vals = df[label_col].unique()
    if set(unique_vals).issubset({0, 1}):
        df["label"] = df[label_col].astype(int)
    else:
        print(f"  ⚠️  ReCOVery labels are not 0/1: {unique_vals}")
        print(f"      Converting to int. VERIFY THIS IS CORRECT.")
        df["label"] = df[label_col].astype(int)
        
    # Try to find news/article ID for grouping
    id_col = None
    for c in ["news_id", "article_id", "id", "url", "link"]:
        if c in df.columns:
            id_col = c
            break
    df["group_id"] = df[id_col].astype(str) if id_col else df["image_path"]
    
    df["source_dataset"] = "recovery"
    return df[["claim_text", "image_path", "label", "source_dataset", "group_id"]]


def group_split(df: pd.DataFrame, val_ratio: float, test_ratio: float, random_state: int):
    """
    Split at the group level (article/image), not row level.
    All rows with the same group_id go to the same split.
    """
    group_col = "group_id"
    
    # Get unique groups and their majority label for stratification
    group_df = df.groupby(group_col)["label"].agg(lambda x: x.mode()[0]).reset_index()
    group_df.columns = [group_col, "group_label"]
    
    groups = group_df[group_col].values
    labels = group_df["group_label"].values
    
    # First split: trainval vs test
    trainval_idx, test_idx = train_test_split(
        np.arange(len(groups)), test_size=test_ratio, stratify=labels, random_state=random_state
    )
    
    # Second split: train vs val
    val_size = val_ratio / (1 - test_ratio)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_size, stratify=labels[trainval_idx], random_state=random_state
    )
    
    train_groups = groups[train_idx]
    val_groups = groups[val_idx]
    test_groups = groups[test_idx]
    
    train = df[df[group_col].isin(train_groups)].reset_index(drop=True)
    val = df[df[group_col].isin(val_groups)].reset_index(drop=True)
    test = df[df[group_col].isin(test_groups)].reset_index(drop=True)
    
    return train, val, test


def verify_split(train, val, test):
    """Detect leakage and print distributions."""
    print("\n" + "=" * 70)
    print("SPLIT VERIFICATION")
    print("=" * 70)
    
    for name, df in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        print(f"\n{name}: {len(df)} samples")
        print(f"  Source distribution: {df['source_dataset'].value_counts().to_dict()}")
        print(f"  Label distribution:  {df['label'].value_counts().sort_index().to_dict()}")
        for src in df["source_dataset"].unique():
            src_df = df[df["source_dataset"] == src]
            print(f"    {src}: labels = {src_df['label'].value_counts().sort_index().to_dict()}")
    
    # Leakage check
    train_groups = set(train["group_id"])
    val_groups = set(val["group_id"])
    test_groups = set(test["group_id"])
    
    leaks = []
    if train_groups & val_groups:
        leaks.append(f"Train-Val: {len(train_groups & val_groups)} groups")
    if train_groups & test_groups:
        leaks.append(f"Train-Test: {len(train_groups & test_groups)} groups")
    if val_groups & test_groups:
        leaks.append(f"Val-Test: {len(val_groups & test_groups)} groups")
    
    if leaks:
        print(f"\n🚨 GROUP LEAKAGE DETECTED: {', '.join(leaks)}")
        print("   This means the same article/image is in multiple splits.")
        print("   FIX THIS BEFORE TRAINING.")
    else:
        print("\n✅ No group leakage detected.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-missing", action="store_true")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    print("=" * 70)
    print("PHASE 1: DATA PREPROCESSING (FIXED VERSION)")
    print("=" * 70)
    os.makedirs(cfg.data.processed_dir, exist_ok=True)
    
    datasets = []
    
    # --- Honest OOC ---
    if os.path.exists(cfg.data.honest_ooc):
        print(f"\n📂 Loading Honest OOC from {cfg.data.honest_ooc}...")
        df_ho = pd.read_csv(cfg.data.honest_ooc)
        cap = getattr(cfg.data, "honest_ooc_cap_per_label", None)
        df_ho = standardize_honest_ooc(df_ho, cap_per_label=cap)
        datasets.append(df_ho)
    else:
        print(f"\n🚨 WARNING: {cfg.data.honest_ooc} NOT FOUND.")
    
    # --- FakeHealth ---
    if os.path.exists(cfg.data.fakehealth):
        print(f"\n📂 Loading FakeHealth from {cfg.data.fakehealth}...")
        df_fh = pd.read_csv(cfg.data.fakehealth)
        df_fh = standardize_fakehealth(df_fh)
        print(f"  Loaded: {len(df_fh)} rows | Labels: {df_fh['label'].value_counts().sort_index().to_dict()}")
        datasets.append(df_fh)
    else:
        print(f"\n🚨 WARNING: {cfg.data.fakehealth} not found.")
    
    # --- ReCOVery ---
    if os.path.exists(cfg.data.recovery):
        print(f"\n📂 Loading ReCOVery from {cfg.data.recovery}...")
        df_rec = pd.read_csv(cfg.data.recovery)
        df_rec = standardize_recovery(df_rec)
        print(f"  Loaded: {len(df_rec)} rows | Labels: {df_rec['label'].value_counts().sort_index().to_dict()}")
        datasets.append(df_rec)
    else:
        print(f"\n🚨 WARNING: {cfg.data.recovery} not found.")
    
    if not datasets:
        raise ValueError("No datasets found!")
    
    # --- Validate image paths ---
    print("\n" + "=" * 70)
    print("VALIDATING IMAGE PATHS")
    print("=" * 70)
    for i, df in enumerate(datasets):
        df["exists"] = df["image_path"].apply(lambda x: os.path.exists(str(x)))
        missing = (~df["exists"]).sum()
        src = df["source_dataset"].iloc[0]
        print(f"  {src}: {missing}/{len(df)} missing images")
        if missing > 0 and args.skip_missing:
            datasets[i] = df[df["exists"]].drop(columns=["exists"]).reset_index(drop=True)
        elif missing > 0:
            raise ValueError(f"{missing} images missing in {src}. Run with --skip-missing or fix paths.")
        else:
            datasets[i] = df.drop(columns=["exists"]).reset_index(drop=True)
    
    # --- Split each dataset at group level, then merge ---
    print("\n" + "=" * 70)
    print("SPLITTING (ARTICLE/ IMAGE LEVEL)")
    print("=" * 70)
    
    trains, vals, tests = [], [], []
    for df in datasets:
        src = df["source_dataset"].iloc[0]
        print(f"\n  Splitting {src}...")
        tr, va, te = group_split(
            df, cfg.data.val_ratio, cfg.data.test_ratio, cfg.data.random_state
        )
        print(f"    Train: {len(tr)} | Val: {len(va)} | Test: {len(te)}")
        trains.append(tr)
        vals.append(va)
        tests.append(te)
    
    train = pd.concat(trains, ignore_index=True)
    val = pd.concat(vals, ignore_index=True)
    test = pd.concat(tests, ignore_index=True)
    
    # Verify no leakage
    verify_split(train, val, test)
    
    # Save
    train_path = os.path.join(cfg.data.processed_dir, "train_multimodal.csv")
    val_path = os.path.join(cfg.data.processed_dir, "val_multimodal.csv")
    test_path = os.path.join(cfg.data.processed_dir, "test_multimodal.csv")
    
    # Drop group_id before saving (or keep it for debugging)
    train.drop(columns=["group_id"]).to_csv(train_path, index=False)
    val.drop(columns=["group_id"]).to_csv(val_path, index=False)
    test.drop(columns=["group_id"]).to_csv(test_path, index=False)
    
    print(f"\n{'=' * 70}")
    print("SAVED")
    print(f"  Train: {train_path} ({len(train)} samples)")
    print(f"  Val:   {val_path} ({len(val)} samples)")
    print(f"  Test:  {test_path} ({len(test)} samples)")
    print("=" * 70)
    
    # Final sanity check: per-source label distribution
    print("\n📊 FINAL LABEL DISTRIBUTION (per source, per split):")
    for split_name, split_df in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
        print(f"\n  {split_name}:")
        for src in split_df["source_dataset"].unique():
            counts = split_df[split_df["source_dataset"] == src]["label"].value_counts().sort_index().to_dict()
            print(f"    {src}: {counts}")


if __name__ == "__main__":
    main()
