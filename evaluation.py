"""
evaluation.py - Evaluation script for CustomLaneUNet on PSU Reservoir test set.
Computes IoU, Dice/F1, Precision, Recall, Pixel Accuracy, Detection Rate (IoU >= 0.6),
and Average IoU of detected positives.
"""

import os
import argparse
import json
import torch
import numpy as np
from tqdm import tqdm

from model import CustomLaneUNet
from dataset import get_dataloaders


def compute_batch_metrics(preds, targets, threshold=0.5, eps=1e-7):
    """
    Compute pixel-level metrics for batch of predictions and targets.
    preds: (B, 1, H, W) probabilities or binary predictions
    targets: (B, 1, H, W) binary ground truth (0 or 1)
    """
    preds_bin = (preds > threshold).float()
    targets_bin = targets.float()

    batch_size = preds.shape[0]
    sample_ious = []
    sample_dices = []
    sample_precisions = []
    sample_recalls = []
    sample_accuracies = []

    for i in range(batch_size):
        p = preds_bin[i].view(-1)
        t = targets_bin[i].view(-1)

        tp = (p * t).sum().item()
        fp = (p * (1.0 - t)).sum().item()
        fn = ((1.0 - p) * t).sum().item()
        tn = ((1.0 - p) * (1.0 - t)).sum().item()

        # IoU
        intersection = tp
        union = tp + fp + fn
        iou = (intersection + eps) / (union + eps) if union > 0 else 1.0

        # Dice / F1
        dice = (2.0 * intersection + eps) / (2.0 * intersection + fp + fn + eps)

        # Precision, Recall, Accuracy
        precision = (tp + eps) / (tp + fp + eps)
        recall = (tp + eps) / (tp + fn + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)

        sample_ious.append(iou)
        sample_dices.append(dice)
        sample_precisions.append(precision)
        sample_recalls.append(recall)
        sample_accuracies.append(accuracy)

    return sample_ious, sample_dices, sample_precisions, sample_recalls, sample_accuracies


def evaluate_model(model, dataloader, device, iou_threshold_det=0.6):
    model.eval()
    all_ious = []
    all_dices = []
    all_precisions = []
    all_recalls = []
    all_accuracies = []

    with torch.no_grad():
        for images, masks, _ in tqdm(dataloader, desc="Evaluating Test Set"):
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            preds = torch.sigmoid(logits)

            ious, dices, precs, recs, accs = compute_batch_metrics(preds, masks)
            all_ious.extend(ious)
            all_dices.extend(dices)
            all_precisions.extend(precs)
            all_recalls.extend(recs)
            all_accuracies.extend(accs)

    all_ious = np.array(all_ious)
    all_dices = np.array(all_dices)
    all_precisions = np.array(all_precisions)
    all_recalls = np.array(all_recalls)
    all_accuracies = np.array(all_accuracies)

    # Detection Flag: IoU >= 0.6
    detected_mask = all_ious >= iou_threshold_det
    detection_count = np.sum(detected_mask)
    total_count = len(all_ious)
    detection_rate = (detection_count / total_count) * 100.0 if total_count > 0 else 0.0

    # Average IoU of detected positives
    avg_iou_positives = np.mean(all_ious[detected_mask]) if detection_count > 0 else 0.0

    results = {
        "total_test_samples": int(total_count),
        "mean_iou": float(np.mean(all_ious)),
        "std_iou": float(np.std(all_ious)),
        "mean_dice_f1": float(np.mean(all_dices)),
        "mean_precision": float(np.mean(all_precisions)),
        "mean_recall": float(np.mean(all_recalls)),
        "pixel_accuracy": float(np.mean(all_accuracies)),
        "detection_threshold_iou": float(iou_threshold_det),
        "detected_samples_count": int(detection_count),
        "detection_rate_pct": float(detection_rate),
        "average_iou_detected_positives": float(avg_iou_positives)
    }

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CustomLaneUNet on Test Set")
    parser.add_argument("--data-dir", type=str, default=r"D:\work_AJfern\work4\psu_reservoir_dataset", help="Dataset directory")
    parser.add_argument("--weights", type=str, default="saved_models/best_model.pt", help="Model checkpoint path")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=128, help="Image size")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataloader
    _, test_loader, _, _ = get_dataloaders(args.data_dir, batch_size=args.batch_size, img_size=(args.img_size, args.img_size), train_ratio=0.65, num_workers=0)

    # Load model
    model = CustomLaneUNet(in_channels=3, num_classes=1, base_c=32).to(device)
    if os.path.exists(args.weights):
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        ep = checkpoint.get('epoch', 'N/A')
        vl = checkpoint.get('val_loss', 0.0)
        print(f"Loaded weights from {args.weights} (Epoch {ep}, Val Loss: {vl:.4f})")
    else:
        print(f"Warning: Weights not found at {args.weights}, evaluating uninitialized model.")

    results = evaluate_model(model, test_loader, device)

    # Print summary
    print("\n" + "=" * 55)
    print("           EVALUATION RESULTS (TEST SET: 350 SAMPLES)")
    print("=" * 55)
    print(f"Mean IoU (Jaccard Index)         : {results['mean_iou']:.4f} (+/- {results['std_iou']:.4f})")
    print(f"Dice Coefficient (F1-Score)      : {results['mean_dice_f1']:.4f}")
    print(f"Precision                        : {results['mean_precision']:.4f}")
    print(f"Recall (Sensitivity)             : {results['mean_recall']:.4f}")
    print(f"Pixel Accuracy                   : {results['pixel_accuracy']:.4f} ({results['pixel_accuracy']*100:.2f}%)")
    print(f"Detection Threshold (IoU >= 0.6) : {results['detection_threshold_iou']:.2f}")
    print(f"Detection Rate                   : {results['detection_rate_pct']:.2f}% ({results['detected_samples_count']}/{results['total_test_samples']})")
    print(f"Average IoU (Detected Positives) : {results['average_iou_detected_positives']:.4f}")
    print("=" * 55)

    # Save JSON
    json_path = os.path.join(args.output_dir, "evaluation_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
    print(f"Saved metrics to: {json_path}")

    # Save Text Report
    report_path = os.path.join(args.output_dir, "evaluation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 55 + "\n")
        f.write("      LANE SEGMENTATION EVALUATION REPORT (TEST SET)\n")
        f.write("=" * 55 + "\n")
        f.write(f"Model Architecture               : CustomLaneUNet (Scratch)\n")
        f.write(f"Dataset                          : PSU Reservoir Dataset (35% Test Split = {results['total_test_samples']} images)\n")
        f.write(f"Mean IoU (Jaccard Index)         : {results['mean_iou']:.4f}\n")
        f.write(f"Dice Coefficient (F1-Score)      : {results['mean_dice_f1']:.4f}\n")
        f.write(f"Precision                        : {results['mean_precision']:.4f}\n")
        f.write(f"Recall                           : {results['mean_recall']:.4f}\n")
        f.write(f"Pixel Accuracy                   : {results['pixel_accuracy']:.4f}\n")
        f.write(f"Detection Rate (IoU >= 0.60)     : {results['detection_rate_pct']:.2f}%\n")
        f.write(f"Average IoU of Positives (>=0.6) : {results['average_iou_detected_positives']:.4f}\n")
        f.write("=" * 55 + "\n")
    print(f"Saved text report to: {report_path}")


if __name__ == "__main__":
    main()
