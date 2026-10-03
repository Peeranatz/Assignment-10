"""
train.py: Training Pipeline for Custom Lane Segmentation Neural Network from Scratch
Author: Peeranat Chunhok (Student ID: 6710110295)
Course: 241-353 Artificial Intelligence Ecosystem Module
Department: Artificial Intelligence Engineering, Faculty of Engineering, Prince of Songkla University
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import get_dataloaders
from model import CustomLaneUNet, get_model


class DiceLoss(nn.Module):
    """
    Soft Dice Loss for Binary Segmentation.
    Maximizes the intersection over union between continuous predicted probabilities and ground truth masks.
    """
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        cardinality = probs_flat.sum() + targets_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice


class CombinedBCEDiceLoss(nn.Module):
    """
    Combined Loss: Weighted sum of Binary Cross-Entropy and Soft Dice Loss.
    L_total = alpha * L_BCE + beta * L_Dice
    Combines pixel-level classification accuracy with global boundary overlap optimization.
    """
    def __init__(self, alpha: float = 0.5, beta: float = 0.5, pos_weight: float = 1.0):
        super(CombinedBCEDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        pos_weight_tensor = torch.tensor([pos_weight]) if pos_weight != 1.0 else None
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.dice_loss = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        total = self.alpha * bce + self.beta * dice
        return total, bce, dice


def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> Tuple[float, float, float, float]:
    """
    Computes standard binary segmentation metrics: IoU, Dice/F1, Precision, and Recall.
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum().item()
    union = (preds_flat + targets_flat).clamp(0, 1).sum().item()
    total_pred = preds_flat.sum().item()
    total_target = targets_flat.sum().item()
    
    eps = 1e-6
    iou = (intersection + eps) / (union + eps)
    dice = (2.0 * intersection + eps) / (total_pred + total_target + eps)
    precision = (intersection + eps) / (total_pred + eps)
    recall = (intersection + eps) / (total_target + eps)
    
    return iou, dice, precision, recall


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    tb_writer: SummaryWriter,
    global_step: int
) -> Tuple[float, float, float, float, float, int]:
    model.train()
    total_loss, total_bce, total_dice = 0.0, 0.0, 0.0
    total_iou, total_f1 = 0.0, 0.0
    
    pbar = tqdm(dataloader, desc=f"Train Epoch {epoch:02d}", leave=False)
    for imgs, masks, _ in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            logits = model(imgs)
            loss, bce, dice = criterion(logits, masks)
            
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Calculate batch metrics
        with torch.no_grad():
            iou, f1, _, _ = compute_metrics(logits, masks)
            
        total_loss += loss.item()
        total_bce += bce.item()
        total_dice += dice.item()
        total_iou += iou
        total_f1 += f1
        
        # Log to TensorBoard per iteration
        tb_writer.add_scalar("train_iter/loss", loss.item(), global_step)
        tb_writer.add_scalar("train_iter/iou", iou, global_step)
        tb_writer.add_scalar("train_iter/lr", optimizer.param_groups[0]["lr"], global_step)
        global_step += 1
        
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "IoU": f"{iou:.4f}"})
        
    n = len(dataloader)
    return total_loss / n, total_bce / n, total_dice / n, total_iou / n, total_f1 / n, global_step


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float, float, float, float, float, float]:
    model.eval()
    total_loss, total_bce, total_dice = 0.0, 0.0, 0.0
    total_iou, total_f1, total_prec, total_rec = 0.0, 0.0, 0.0, 0.0
    
    with torch.no_grad():
        for imgs, masks, _ in dataloader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits = model(imgs)
                loss, bce, dice = criterion(logits, masks)
                
            iou, f1, prec, rec = compute_metrics(logits, masks)
            
            total_loss += loss.item()
            total_bce += bce.item()
            total_dice += dice.item()
            total_iou += iou
            total_f1 += f1
            total_prec += prec
            total_rec += rec
            
    n = len(dataloader)
    return (total_loss / n, total_bce / n, total_dice / n,
            total_iou / n, total_f1 / n, total_prec / n, total_rec / n)


def plot_and_save_loss_curves(history: Dict, save_path: Path):
    """Generates clean publication-ready training & validation curves."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), dpi=200)
    
    # 1. Total Loss Curve
    axes[0].plot(epochs, history["train_loss"], 'r-o', label="Train Loss", linewidth=2, markersize=4)
    axes[0].plot(epochs, history["val_loss"], 'b--s', label="Val Loss", linewidth=2, markersize=4)
    axes[0].set_title("Total Loss (BCE + Dice)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Epoch", fontsize=10)
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    
    # 2. IoU Metric Curve
    axes[1].plot(epochs, history["train_iou"], 'g-o', label="Train IoU", linewidth=2, markersize=4)
    axes[1].plot(epochs, history["val_iou"], 'm--s', label="Val IoU", linewidth=2, markersize=4)
    axes[1].set_title("Mean IoU over Epochs", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Epoch", fontsize=10)
    axes[1].set_ylabel("IoU Score (0.0 - 1.0)", fontsize=10)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()
    
    # 3. Learning Rate Scheduler
    axes[2].plot(epochs, history["lr"], 'c-d', label="Learning Rate", linewidth=2, markersize=4)
    axes[2].set_title("Cosine Annealing LR Schedule", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Epoch", fontsize=10)
    axes[2].set_ylabel("Learning Rate", fontsize=10)
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()
    
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Loss curves saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Train CustomLaneUNet from scratch on PSU Reservoir dataset")
    parser.add_argument("--data-dir", type=str, default=r"D:\work_AJfern\work4\psu_reservoir_dataset", help="Path to PSU dataset")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs (>= 30)")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--img-size", type=int, default=128, help="Target image resolution (H, W)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
    parser.add_argument("--base-c", type=int, default=32, help="Base feature channels in U-Net")
    parser.add_argument("--save-dir", type=str, default="saved_models", help="Directory to save model weights")
    parser.add_argument("--log-dir", type=str, default="runs/lane_unet", help="TensorBoard log directory")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory for output graphs and reports")
    args = parser.parse_args()

    # Create output directories
    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    results_dir = Path(args.results_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Set up hardware device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting Training from Scratch ===")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Dataset root: {args.data_dir}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, Image Size: {args.img_size}x{args.img_size}")

    # Load dataset with 65% Train : 35% Test split
    train_loader, test_loader, train_samples, test_samples = get_dataloaders(
        dataset_root=args.data_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        train_ratio=0.65,
        seed=42,
        num_workers=0
    )
    print(f"Train samples (65%): {len(train_samples)}, Test samples (35%): {len(test_samples)}")

    # Instantiate model from scratch
    model = get_model(in_channels=3, num_classes=1, base_c=args.base_c).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: CustomLaneUNet (Params: {total_params:,}, Memory: {total_params * 4 / (1024**2):.2f} MB)")

    # Criterion, Optimizer, Scheduler, Scaler
    criterion = CombinedBCEDiceLoss(alpha=0.5, beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    # TensorBoard SummaryWriter
    tb_writer = SummaryWriter(str(log_dir))
    
    # Track training history
    history = {
        "train_loss": [], "train_bce": [], "train_dice": [], "train_iou": [], "train_f1": [],
        "val_loss": [], "val_bce": [], "val_dice": [], "val_iou": [], "val_f1": [],
        "val_precision": [], "val_recall": [], "lr": []
    }

    best_val_iou = 0.0
    global_step = 0
    t0 = time.time()

    print("\nTraining Progress:")
    for epoch in range(1, args.epochs + 1):
        # 1. Train epoch
        tr_loss, tr_bce, tr_dice, tr_iou, tr_f1, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch, tb_writer, global_step
        )
        
        # 2. Evaluate on 35% test split
        val_loss, val_bce, val_dice, val_iou, val_f1, val_prec, val_rec = evaluate(
            model, test_loader, criterion, device
        )
        
        # Step LR Scheduler
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        
        # Record history
        history["train_loss"].append(tr_loss)
        history["train_bce"].append(tr_bce)
        history["train_dice"].append(tr_dice)
        history["train_iou"].append(tr_iou)
        history["train_f1"].append(tr_f1)
        history["val_loss"].append(val_loss)
        history["val_bce"].append(val_bce)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_f1"].append(val_f1)
        history["val_precision"].append(val_prec)
        history["val_recall"].append(val_rec)
        history["lr"].append(current_lr)
        
        # TensorBoard epoch logging
        tb_writer.add_scalar("train/loss", tr_loss, epoch)
        tb_writer.add_scalar("train/bce_loss", tr_bce, epoch)
        tb_writer.add_scalar("train/dice_loss", tr_dice, epoch)
        tb_writer.add_scalar("train/iou", tr_iou, epoch)
        tb_writer.add_scalar("val/loss", val_loss, epoch)
        tb_writer.add_scalar("val/bce_loss", val_bce, epoch)
        tb_writer.add_scalar("val/dice_loss", val_dice, epoch)
        tb_writer.add_scalar("val/iou", val_iou, epoch)
        tb_writer.add_scalar("val/f1_dice", val_f1, epoch)
        tb_writer.add_scalar("val/precision", val_prec, epoch)
        tb_writer.add_scalar("val/recall", val_rec, epoch)
        tb_writer.add_scalar("lr/epoch", current_lr, epoch)
        
        # Save Best Model Checkpoint
        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_iou": val_iou,
                "val_f1": val_f1,
                "config": vars(args)
            }, save_dir / "best_model.pt")
            
        # Save Last Model Checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "val_iou": val_iou,
            "config": vars(args)
        }, save_dir / "last_model.pt")
        
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] "
              f"Train Loss: {tr_loss:.4f}, Train IoU: {tr_iou:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}, Val F1: {val_f1:.4f} | "
              f"LR: {current_lr:.6f} {'[BEST SAVED]' if is_best else ''}")

    elapsed = time.time() - t0
    print(f"\n=== Training Finished in {elapsed/60:.2f} mins ===")
    print(f"Best Validation IoU: {best_val_iou:.4f}")
    
    # Save training history and loss plot
    with open(results_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
        
    plot_and_save_loss_curves(history, results_dir / "loss_curves.png")
    tb_writer.close()


if __name__ == "__main__":
    main()
