#!/usr/bin/env python3
"""Phase 1 Training Script."""
import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import BertTokenizer
from tqdm import tqdm
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score, 
    roc_auc_score, confusion_matrix, classification_report
)

from src.config import load_config
from src.models import ContrastiveFusion
from src.dataset import TripletDataset, BinaryMMDataset, collate_fn_contrastive, collate_fn_binary
from src.losses import FocalLoss, ContrastiveLoss
from src.transforms import get_train_transforms, get_val_transforms
from src.utils import (
    set_seed, count_parameters, freeze_backbone, unfreeze_top_layers,
    print_metrics, save_checkpoint, EarlyStopping, save_json
)


def build_model(cfg):
    model = ContrastiveFusion(
        biobert_name=cfg.model.biobert,
        vit_name=cfg.model.vit,
        fusion_type=cfg.model.fusion_type,
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
    )
    if cfg.model.freeze_biobert:
        freeze_backbone(model.biobert, "biobert")
    else:
        unfreeze_top_layers(model.biobert, "biobert", cfg.model.unfreeze_biobert_top_n)
    if cfg.model.freeze_vit:
        freeze_backbone(model.vit, "vit")
    else:
        unfreeze_top_layers(model.vit, "vit", cfg.model.unfreeze_vit_top_n)
    print(f"nTrainable parameters: {count_parameters(model):,}")
    return model


def build_dataloaders(cfg, tokenizer):
    mode = cfg.training.mode
    if mode == "contrastive":
        train_ds = TripletDataset(
            os.path.join(cfg.data.processed_dir, "train_multimodal.csv"),
            tokenizer, get_train_transforms(cfg.training.use_augmentation),
        )
        val_ds = TripletDataset(
            os.path.join(cfg.data.processed_dir, "val_multimodal.csv"),
            tokenizer, get_val_transforms(),
        )
        test_ds = TripletDataset(
            os.path.join(cfg.data.processed_dir, "test_multimodal.csv"),
            tokenizer, get_val_transforms(),
        )
        collate_fn = collate_fn_contrastive
    else:
        train_ds = BinaryMMDataset(
            os.path.join(cfg.data.processed_dir, "train_multimodal.csv"),
            tokenizer, get_train_transforms(cfg.training.use_augmentation),
        )
        val_ds = BinaryMMDataset(
            os.path.join(cfg.data.processed_dir, "val_multimodal.csv"),
            tokenizer, get_val_transforms(),
        )
        test_ds = BinaryMMDataset(
            os.path.join(cfg.data.processed_dir, "test_multimodal.csv"),
            tokenizer, get_val_transforms(),
        )
        collate_fn = collate_fn_binary
    
    # ============================================================
    # FIX: WeightedRandomSampler for balanced batches
    # ============================================================
    if mode == "binary":
        # Count Fake (1.0) vs Real (0.0) in training set
        fake_count = 0
        real_count = 0
        for idx in range(len(train_ds)):
            label = train_ds[idx]["y"].item()
            if label == 1.0:
                fake_count += 1
            else:
                real_count += 1
        
        total = fake_count + real_count
        print(f"n[WeightedSampler] Train set: Fake={fake_count}, Real={real_count}")
        
        # Weight: rare class gets high weight, common class gets weight=1
        weight_for_fake = total / (2.0 * fake_count) if fake_count > 0 else 1.0
        weight_for_real = total / (2.0 * real_count) if real_count > 0 else 1.0
        
        print(f"[WeightedSampler] Sample weights: Fake={weight_for_fake:.2f}, Real={weight_for_real:.2f}")
        
        sample_weights = []
        for idx in range(len(train_ds)):
            label = train_ds[idx]["y"].item()
            if label == 1.0:
                sample_weights.append(weight_for_fake)
            else:
                sample_weights.append(weight_for_real)
        
        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
        )
        
        train_loader = DataLoader(
            train_ds, batch_size=cfg.training.batch_size, 
            sampler=sampler,  # NO shuffle when using sampler
            num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory,
            collate_fn=collate_fn, drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.training.batch_size, shuffle=True,
            num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory,
            collate_fn=collate_fn, drop_last=True,
        )
    
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.system.num_workers, pin_memory=cfg.system.pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, test_loader


def train_epoch_contrastive(model, loader, optimizer, criterion, device, scaler, cfg, epoch):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for step, batch in enumerate(pbar):
        true_ids = batch["true_ids"].to(device)
        true_mask = batch["true_mask"].to(device)
        fake_ids = batch["fake_ids"].to(device)
        fake_mask = batch["fake_mask"].to(device)
        pix = batch["pix"].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg.training.use_amp and device.type == "cuda"):
            fused_true, fused_fake = model.forward_contrastive(
                true_ids, true_mask, fake_ids, fake_mask, pix
            )
            loss = criterion(fused_true, fused_fake)
        if scaler is not None:
            scaler.scale(loss).backward()
            if (step + 1) % cfg.training.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (step + 1) % cfg.training.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item()
        pbar.set_postfix({"loss": loss.item()})
    return total_loss / len(loader)


def validate_contrastive(model, loader, criterion, device, cfg):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Validate]"):
            true_ids = batch["true_ids"].to(device)
            true_mask = batch["true_mask"].to(device)
            fake_ids = batch["fake_ids"].to(device)
            fake_mask = batch["fake_mask"].to(device)
            pix = batch["pix"].to(device)
            fused_true, fused_fake = model.forward_contrastive(
                true_ids, true_mask, fake_ids, fake_mask, pix
            )
            loss = criterion(fused_true, fused_fake)
            total_loss += loss.item()
    return total_loss / len(loader)


def train_epoch_binary(model, loader, optimizer, criterion, device, scaler, cfg, epoch):
    model.train()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for step, batch in enumerate(pbar):
        ids = batch["ids"].to(device)
        mask = batch["mask"].to(device)
        pix = batch["pix"].to(device)
        y = batch["y"].to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=cfg.training.use_amp and device.type == "cuda"):
            logits = model(ids, mask, pix)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            if (step + 1) % cfg.training.accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (step + 1) % cfg.training.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
        total_loss += loss.item()
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y.cpu().numpy())
        pbar.set_postfix({"loss": loss.item()})
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)
    
    # ============================================================
    # FIX: Debug prints to verify label distribution
    # ============================================================
    print(f"n[Train Epoch {epoch}] Label distribution: {dict(zip(*np.unique(all_labels, return_counts=True)))}")
    print(f"[Train Epoch {epoch}] Pred distribution:  {dict(zip(*np.unique(preds, return_counts=True)))}")
    
    train_f1 = f1_score(all_labels, preds, zero_division=0)
    return total_loss / len(loader), train_f1


def validate_binary(model, loader, criterion, device, cfg):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    all_sources = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Validate]"):
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            pix = batch["pix"].to(device)
            y = batch["y"].to(device)
            logits = model(ids, mask, pix)
            loss = criterion(logits, y)
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())
            all_sources.extend(batch["source"])
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    
    best_thresh = 0.5
    best_f1 = 0.0
    
    # If all predictions are identical, threshold search is meaningless
    if len(np.unique(all_probs)) < 2:
        print(f"n[WARNING] All validation predictions are identical ({all_probs[0]:.4f}). Model has not learned to separate classes.")
    else:
        for thresh in np.arange(
            cfg.evaluation.threshold_search_min,
            cfg.evaluation.threshold_search_max,
            cfg.evaluation.threshold_search_step,
        ):
            preds_t = (all_probs > thresh).astype(int)
            f1_t = f1_score(all_labels, preds_t, zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_thresh = thresh
    
    val_preds = (all_probs > best_thresh).astype(int)
    
    # ============================================================
    # FIX: More detailed metrics including per-class F1
    # ============================================================
    metrics = {
        "loss": total_loss / len(loader),
        "threshold": best_thresh,
        "accuracy": accuracy_score(all_labels, val_preds),
        "precision": precision_score(all_labels, val_preds, zero_division=0),
        "recall": recall_score(all_labels, val_preds, zero_division=0),
        "f1": f1_score(all_labels, val_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
    }
    
    # Per-class precision/recall
    if len(np.unique(all_labels)) == 2:
        metrics["precision_fake"] = precision_score(all_labels, val_preds, pos_label=1, zero_division=0)
        metrics["recall_fake"] = recall_score(all_labels, val_preds, pos_label=1, zero_division=0)
        metrics["precision_real"] = precision_score(all_labels, val_preds, pos_label=0, zero_division=0)
        metrics["recall_real"] = recall_score(all_labels, val_preds, pos_label=0, zero_division=0)
    
    source_acc = {}
    for src in set(all_sources):
        mask = np.array([s == src for s in all_sources])
        if mask.sum() > 0:
            source_acc[src] = accuracy_score(all_labels[mask], val_preds[mask])
    metrics["per_source"] = source_acc
    return metrics


def test_binary(model, loader, device, threshold, cfg):
    model.eval()
    all_probs = []
    all_labels = []
    all_sources = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="[Test]"):
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            pix = batch["pix"].to(device)
            y = batch["y"].to(device)
            logits = model(ids, mask, pix)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.cpu().numpy())
            all_sources.extend(batch["source"])
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    test_preds = (all_probs > threshold).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(all_labels, test_preds),
        "precision": precision_score(all_labels, test_preds, zero_division=0),
        "recall": recall_score(all_labels, test_preds, zero_division=0),
        "f1": f1_score(all_labels, test_preds, zero_division=0),
        "auc": roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0.0,
    }
    source_acc = {}
    for src in set(all_sources):
        mask = np.array([s == src for s in all_sources])
        if mask.sum() > 0:
            source_acc[src] = accuracy_score(all_labels[mask], test_preds[mask])
    metrics["per_source"] = source_acc
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, test_preds)
    
    # Classification Report
    report = classification_report(
        all_labels, test_preds,
        target_names=["Fake (0)", "Real (1)"],
        digits=4,
        output_dict=True
    )
    
    return metrics, all_probs, all_labels, cm, report


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Multimodal Encoder Training")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--mode", choices=["contrastive", "binary"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--fusion", choices=["concat", "cross_attn", "bilinear"])
    parser.add_argument("--freeze_backbones", action="store_true")
    parser.add_argument("--unfreeze_biobert", type=int)
    parser.add_argument("--unfreeze_vit", type=int)
    parser.add_argument("--resume", type=str)
    parser.add_argument("--checkpoint_dir", type=str, help="Override checkpoint directory for this run")
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    if args.mode:
        cfg.training.mode = args.mode
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.fusion:
        cfg.model.fusion_type = args.fusion
    if args.freeze_backbones:
        cfg.model.freeze_biobert = True
        cfg.model.freeze_vit = True
    if args.unfreeze_biobert is not None:
        cfg.model.freeze_biobert = False
        cfg.model.unfreeze_biobert_top_n = args.unfreeze_biobert
    if args.unfreeze_vit is not None:
        cfg.model.freeze_vit = False
        cfg.model.unfreeze_vit_top_n = args.unfreeze_vit
    
    # Override checkpoint dir if provided (used by run_experiments.py)
    ckpt_dir = args.checkpoint_dir if args.checkpoint_dir else cfg.logging.checkpoint_dir
    log_dir = os.path.join(ckpt_dir, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    set_seed(cfg.system.seed)
    device = torch.device(cfg.system.device if torch.cuda.is_available() else "cpu")
    print(f"nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    print(f"nLoading tokenizer: {cfg.model.biobert}")
    tokenizer = BertTokenizer.from_pretrained(cfg.model.biobert)
    
    print(f"nBuilding dataloaders (mode={cfg.training.mode})...")
    train_loader, val_loader, test_loader = build_dataloaders(cfg, tokenizer)
    
    print(f"nBuilding model (fusion={cfg.model.fusion_type})...")
    model = build_model(cfg).to(device)
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.training.lr, weight_decay=cfg.training.weight_decay,
    )
    
    total_steps = len(train_loader) * cfg.training.epochs // cfg.training.accum_steps
    warmup_steps = int(total_steps * cfg.training.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=warmup_steps, T_mult=2
    )
    
    if cfg.training.mode == "contrastive":
        criterion = ContrastiveLoss(margin=cfg.training.contrastive_margin)
    else:
        # ============================================================
        # FIX: Compute pos_weight from actual training data
        # Then pass it to FocalLoss
        # ============================================================
        # We need to peek at the train_loader dataset to count classes
        train_dataset = train_loader.dataset
        fake_count = 0
        real_count = 0
        for idx in range(len(train_dataset)):
            label = train_dataset[idx]["y"].item()
            if label == 1.0:
                fake_count += 1
            else:
                real_count += 1
        
        # pos_weight = num_neg / num_pos = Real / Fake
        # If Fake is 10x rarer, pos_weight = 10.0
        if fake_count > 0:
            pos_weight_value = real_count / fake_count
        else:
            pos_weight_value = 1.0
        
        pos_weight_tensor = torch.tensor([pos_weight_value], dtype=torch.float32).to(device)
        print(f"n[Loss Setup] Fake={fake_count}, Real={real_count}")
        print(f"[Loss Setup] pos_weight (Real/Fake ratio) = {pos_weight_value:.2f}")
        print(f"[Loss Setup] This means missing a Fake sample is penalized {pos_weight_value:.1f}x more than missing a Real sample.")
        
        criterion = FocalLoss(
            alpha=cfg.training.focal_alpha, 
            gamma=cfg.training.focal_gamma,
            pos_weight=pos_weight_tensor
        )
    
    scaler = torch.cuda.amp.GradScaler() if cfg.training.use_amp and device.type == "cuda" else None
    
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(log_dir, run_name))
    
    early_stop = EarlyStopping(patience=cfg.training.early_stopping_patience, mode="max")
    
    start_epoch = 1
    best_val_f1 = 0.0
    best_threshold = 0.5
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_f1 = checkpoint.get("metrics", {}).get("f1", 0.0)
        print(f"nResumed from {args.resume} at epoch {start_epoch}")
    
    print("n" + "=" * 70)
    print(f"STARTING TRAINING: {cfg.training.mode.upper()} MODE")
    print(f"Epochs: {cfg.training.epochs} | Batch: {cfg.training.batch_size} | Fusion: {cfg.model.fusion_type}")
    print(f"Checkpoint dir: {ckpt_dir}")
    print("=" * 70)
    
    for epoch in range(start_epoch, cfg.training.epochs + 1):
        epoch_start = time.time()
        
        if cfg.training.mode == "contrastive":
            train_loss = train_epoch_contrastive(
                model, train_loader, optimizer, criterion, device, scaler, cfg, epoch
            )
            val_loss = validate_contrastive(model, val_loader, criterion, device, cfg)
            print(f"nEpoch {epoch}/{cfg.training.epochs} | Time: {time.time()-epoch_start:.1f}s")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            if val_loss < getattr(main, "best_val_loss", float("inf")):
                main.best_val_loss = val_loss
                save_checkpoint(
                    model, optimizer, epoch, {"val_loss": val_loss},
                    os.path.join(ckpt_dir, f"best_contrastive_{run_name}.pt"),
                    is_best=True,
                )
        else:
            train_loss, train_f1 = train_epoch_binary(
                model, train_loader, optimizer, criterion, device, scaler, cfg, epoch
            )
            val_metrics = validate_binary(model, val_loader, criterion, device, cfg)
            print(f"nEpoch {epoch}/{cfg.training.epochs} | Time: {time.time()-epoch_start:.1f}s")
            print(f"  Train Loss: {train_loss:.4f} | Train F1: {train_f1:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f} | Val F1: {val_metrics['f1']:.4f} (thresh={val_metrics['threshold']:.2f})")
            print(f"  Val Acc:    {val_metrics['accuracy']:.4f} | Val AUC: {val_metrics['auc']:.4f}")
            print(f"  Val Fake Precision: {val_metrics.get('precision_fake', 0):.4f} | Val Fake Recall: {val_metrics.get('recall_fake', 0):.4f}")
            print(f"  Per-source: {val_metrics['per_source']}")
            
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("F1/train", train_f1, epoch)
            writer.add_scalar("F1/val", val_metrics["f1"], epoch)
            writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
            writer.add_scalar("AUC/val", val_metrics["auc"], epoch)
            writer.add_scalar("Threshold/best", val_metrics["threshold"], epoch)
            
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_threshold = val_metrics["threshold"]
                save_checkpoint(
                    model, optimizer, epoch, val_metrics,
                    os.path.join(ckpt_dir, f"best_binary_{run_name}.pt"),
                    is_best=True,
                )
            
            if early_stop(val_metrics["f1"]):
                print(f"nEarly stopping triggered at epoch {epoch}")
                break
        
        if epoch % cfg.logging.save_every_n_epochs == 0:
            save_checkpoint(
                model, optimizer, epoch, {},
                os.path.join(ckpt_dir, f"epoch_{epoch}_{run_name}.pt"),
            )
        scheduler.step()
    
    writer.close()
    
    # Final test evaluation
    if cfg.training.mode == "binary":
        print("n" + "=" * 70)
        print("FINAL TEST EVALUATION")
        print("=" * 70)
        print(f"Using best threshold: {best_threshold:.2f}")
        
        test_metrics, test_probs, test_labels, cm, report = test_binary(
            model, test_loader, device, best_threshold, cfg
        )
        
        print(f"nTest Macro-F1: {test_metrics['f1']:.4f}")
        print_metrics(test_metrics, prefix="Test ")
        
        print(f"nPer-source Test Accuracy:")
        for src, acc in test_metrics["per_source"].items():
            print(f"  {src:15s}: {acc:.4f}")
        
        # Confusion Matrix
        print(f"nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"                 Fake    Real")
        print(f"Actual Fake      {cm[0][0]:4d}    {cm[0][1]:4d}")
        print(f"Actual Real      {cm[1][0]:4d}    {cm[1][1]:4d}")
        
        # Classification Report
        print(f"nClassification Report:")
        print(classification_report(
            test_labels, (test_probs > best_threshold).astype(int),
            target_names=["Fake (0)", "Real (1)"],
            digits=4
        ))
        
        # Save everything to JSON
        results = {
            "config": {
                "fusion": cfg.model.fusion_type,
                "epochs": cfg.training.epochs,
                "batch_size": cfg.training.batch_size,
                "frozen_biobert": cfg.model.freeze_biobert,
                "frozen_vit": cfg.model.freeze_vit,
            },
            "best_val_f1": best_val_f1,
            "best_threshold": best_threshold,
            "test_metrics": test_metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }
        save_json(results, os.path.join(ckpt_dir, f"test_results_{run_name}.json"))
        print(f"nSaved full results to: {os.path.join(ckpt_dir, f'test_results_{run_name}.json')}")
    
    # Save projection layers for Phase 4
    proj_path = os.path.join(ckpt_dir, f"best_fusion_{run_name}.pt")
    model.save_projection_layers(proj_path)
    
    print("n" + "=" * 70)
    print("TRAINING COMPLETE")
    print(f"Best checkpoint: {os.path.join(ckpt_dir, f'best_binary_{run_name}.pt')}")
    print(f"Projection layers for Phase 4: {proj_path}")
    print("=" * 70)
    
    # Return metrics for experiment runner
    if cfg.training.mode == "binary":
        return test_metrics
    return None


if __name__ == "__main__":
    main()