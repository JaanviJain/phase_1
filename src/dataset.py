"""Dataset classes for Phase 1 multimodal training."""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import Callable, Optional, Dict, List


# ============================================================
# CRITICAL: Source-to-label mapping for binary classification
# ============================================================
# honest_ooc  = Real news  → label 0
# recovery    = Fake news  → label 1  
# fakehealth  = Fake news  → label 1
# This ensures ONLY 0 and 1 exist — no label=2 to break BCE loss
SOURCE_TO_BINARY_LABEL = {
    "honest_ooc": 0.0,
    "recovery": 1.0,
    "fakehealth": 1.0,
}


class TripletDataset(Dataset):
    """
    Contrastive triplet dataset using Honest OOC structure.
    Each sample: (image, true_caption, falsified_caption)
    """
    def __init__(self, csv_path: str, tokenizer,
                 image_transform: Optional[Callable] = None, max_length: int = 512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        self.triplets = self._build_triplets()
        print(f"[TripletDataset] Built {len(self.triplets)} triplets from {csv_path}")
        
    def _build_triplets(self) -> List[Dict]:
        triplets = []
        image_col = "image_path" if "image_path" in self.df.columns else "image"
        caption_col = "caption" if "caption" in self.df.columns else "claim_text"
        label_col = "label" if "label" in self.df.columns else "rating"
        
        grouped = self.df.groupby(image_col)
        for img_path, group in grouped:
            true_caps = group[group[label_col] == 1][caption_col].tolist()
            fake_caps = group[group[label_col] == 0][caption_col].tolist()
            if len(true_caps) > 0 and len(fake_caps) > 0:
                triplets.append({
                    "image_path": img_path,
                    "true_caption": str(true_caps[0]),
                    "fake_caption": str(fake_caps[0]),
                })
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        trip = self.triplets[idx]
        pix = self._load_image(trip["image_path"])
        
        true_enc = self.tokenizer(
            trip["true_caption"], max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        fake_enc = self.tokenizer(
            trip["fake_caption"], max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        return {
            "true_ids": true_enc["input_ids"].squeeze(0),
            "true_mask": true_enc["attention_mask"].squeeze(0),
            "fake_ids": fake_enc["input_ids"].squeeze(0),
            "fake_mask": fake_enc["attention_mask"].squeeze(0),
            "pix": pix,
            "image_path": trip["image_path"],
        }
    
    def _load_image(self, path: str):
        try:
            if os.path.exists(str(path)):
                img = Image.open(path).convert("RGB")
                if self.image_transform:
                    return self.image_transform(img)
        except Exception:
            pass
        return torch.zeros(3, 224, 224)


class BinaryMMDataset(Dataset):
    """
    Binary classification dataset for multimodal data.
    Supports FakeHealth, ReCOVery, and Honest OOC (capped).
    """
    def __init__(self, csv_path: str, tokenizer,
                 image_transform: Optional[Callable] = None, max_length: int = 512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        
        required = ["claim_text", "image_path", "source_dataset"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col} in {csv_path}")
        
        # ============================================================
        # FIX 1: Map 3 sources to 2 binary labels using SOURCE_TO_BINARY_LABEL
        # ============================================================
        self.df["binary_label"] = self.df["source_dataset"].map(SOURCE_TO_BINARY_LABEL)
        
        # Drop rows where source is unknown (not in our mapping)
        before_drop = len(self.df)
        self.df = self.df[self.df["binary_label"].notna()].reset_index(drop=True)
        after_drop = len(self.df)
        if before_drop != after_drop:
            print(f"[BinaryMMDataset] Dropped {before_drop - after_drop} rows with unknown source_dataset")
        
        self.df = self.df[self.df["image_path"].notna()].reset_index(drop=True)
        valid_mask = [os.path.exists(str(p)) for p in self.df["image_path"]]
        self.df = self.df[valid_mask].reset_index(drop=True)
        
        # ============================================================
        # FIX 2: Verify only 0 and 1 exist in labels
        # ============================================================
        unique_labels = self.df["binary_label"].unique()
        print(f"[BinaryMMDataset] Loaded {len(self.df)} valid samples from {csv_path}")
        print(f"  Unique labels: {sorted(unique_labels)} (MUST be [0.0, 1.0])")
        print(f"  Sources: {self.df['source_dataset'].value_counts().to_dict()}")
        print(f"  Label distribution: {self.df['binary_label'].value_counts().to_dict()}")
        
        if not set(unique_labels).issubset({0.0, 1.0}):
            raise ValueError(f"CRITICAL: Labels contain values other than 0.0 and 1.0: {unique_labels}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pix = self._load_image(row["image_path"])
        
        enc = self.tokenizer(
            str(row["claim_text"]), max_length=self.max_length, truncation=True,
            padding="max_length", return_tensors="pt",
        )
        source = row.get("source_dataset", "unknown")
        
        # ============================================================
        # FIX 3: Use binary_label instead of raw label column
        # ============================================================
        label_value = float(row["binary_label"])
        
        return {
            "ids": enc["input_ids"].squeeze(0),
            "mask": enc["attention_mask"].squeeze(0),
            "pix": pix,
            "y": torch.tensor(label_value, dtype=torch.float),
            "source": source,
            "image_path": row["image_path"],
        }
    
    def _load_image(self, path: str):
        try:
            if os.path.exists(str(path)):
                img = Image.open(path).convert("RGB")
                if self.image_transform:
                    return self.image_transform(img)
        except Exception:
            pass
        return torch.zeros(3, 224, 224)


def collate_fn_contrastive(batch):
    return {
        "true_ids": torch.stack([b["true_ids"] for b in batch]),
        "true_mask": torch.stack([b["true_mask"] for b in batch]),
        "fake_ids": torch.stack([b["fake_ids"] for b in batch]),
        "fake_mask": torch.stack([b["fake_mask"] for b in batch]),
        "pix": torch.stack([b["pix"] for b in batch]),
        "image_path": [b["image_path"] for b in batch],
    }


def collate_fn_binary(batch):
    return {
        "ids": torch.stack([b["ids"] for b in batch]),
        "mask": torch.stack([b["mask"] for b in batch]),
        "pix": torch.stack([b["pix"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
        "source": [b["source"] for b in batch],
        "image_path": [b["image_path"] for b in batch],
    }