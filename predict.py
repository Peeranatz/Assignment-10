"""
predict.py - Inference and visualization script for CustomLaneUNet.
Generates 4-panel visual snapshots (Original | Ground Truth | Prediction | Overlay)
and saves them to results/snapshots/.
"""

import os
import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import CustomLaneUNet
from dataset import get_dataloaders


def overlay_mask_on_image(image_rgb, mask, color=(0, 255, 0), alpha=0.45):
    """
    Overlays a binary mask onto an RGB image with transparency and lane contours.
    image_rgb: (H, W, 3) uint8 RGB image
    mask: (H, W) uint8 binary mask (0 or 255)
    """
    overlay = image_rgb.copy()
    colored_mask = np.zeros_like(image_rgb)
    colored_mask[mask > 127] = color

    cv2.addWeighted(colored_mask, alpha, overlay, 1 - alpha, 0, overlay)

    # Draw lane boundary contour for crisp visualization
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 0), 1)

    return overlay


def run_inference_and_visualize(model, dataloader, device, output_dir="results/snapshots", num_samples=12, threshold=0.5):
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    saved_count = 0
    composite_items = []

    with torch.no_grad():
        for batch_idx, (images, masks, img_names) in enumerate(dataloader):
            images_gpu = images.to(device)
            logits = model(images_gpu)
            preds = torch.sigmoid(logits).cpu().numpy()

            images_cpu = images.numpy()
            masks_cpu = masks.numpy()

            for i in range(len(img_names)):
                if saved_count >= num_samples:
                    break

                img_name = img_names[i]
                img_rgb = (images_cpu[i].transpose(1, 2, 0) * 255).astype(np.uint8)
                gt_mask = (masks_cpu[i, 0] * 255).astype(np.uint8)
                pred_prob = preds[i, 0]
                pred_mask = ((pred_prob > threshold) * 255).astype(np.uint8)

                # Compute sample IoU & Dice
                p_bin = (pred_prob > threshold).astype(np.float32)
                t_bin = (masks_cpu[i, 0] > 0.5).astype(np.float32)
                intersection = np.sum(p_bin * t_bin)
                union = np.sum(p_bin) + np.sum(t_bin) - intersection
                iou = (intersection + 1e-7) / (union + 1e-7) if union > 0 else 1.0
                dice = (2.0 * intersection + 1e-7) / (np.sum(p_bin) + np.sum(t_bin) + 1e-7)

                overlay_img = overlay_mask_on_image(img_rgb, pred_mask, color=(0, 255, 0), alpha=0.5)

                # Plot 4-panel snapshot
                fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), dpi=150)
                
                axes[0].imshow(img_rgb)
                axes[0].set_title(f"Original Input\n({img_name})", fontsize=11, fontweight="bold")
                axes[0].axis("off")

                axes[1].imshow(gt_mask, cmap="gray")
                axes[1].set_title("Ground Truth Mask\n(PSU Reservoir)", fontsize=11, fontweight="bold")
                axes[1].axis("off")

                im3 = axes[2].imshow(pred_prob, cmap="magma", vmin=0.0, vmax=1.0)
                axes[2].set_title(f"Predicted Probability Map\n(Threshold={threshold})", fontsize=11, fontweight="bold")
                axes[2].axis("off")
                fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

                axes[3].imshow(overlay_img)
                axes[3].set_title(f"Lane Overlay\nIoU: {iou:.3f} | Dice: {dice:.3f}", fontsize=11, fontweight="bold", color="darkgreen" if iou >= 0.6 else "darkred")
                axes[3].axis("off")

                plt.tight_layout()
                save_path = os.path.join(output_dir, f"sample_{saved_count+1:02d}_{os.path.splitext(img_name)[0]}.png")
                plt.savefig(save_path, bbox_inches="tight")
                plt.close(fig)

                composite_items.append((img_rgb, gt_mask, pred_mask, overlay_img, img_name, iou, dice))
                saved_count += 1

            if saved_count >= num_samples:
                break

    print(f"Successfully generated and saved {saved_count} snapshot comparisons to: {output_dir}")

    # Generate multi-sample composite overview image (3x4 grid)
    if composite_items:
        grid_rows = min(4, len(composite_items))
        fig, axes = plt.subplots(grid_rows, 4, figsize=(16, 3.8 * grid_rows), dpi=150)
        plt.subplots_adjust(wspace=0.05, hspace=0.2)

        for row_idx in range(grid_rows):
            img_rgb, gt_mask, pred_mask, overlay_img, name, iou, dice = composite_items[row_idx]
            
            axes[row_idx, 0].imshow(img_rgb)
            if row_idx == 0:
                axes[row_idx, 0].set_title("Original Image", fontsize=12, fontweight="bold")
            axes[row_idx, 0].set_ylabel(name, fontsize=9)
            axes[row_idx, 0].set_xticks([]); axes[row_idx, 0].set_yticks([])

            axes[row_idx, 1].imshow(gt_mask, cmap="gray")
            if row_idx == 0:
                axes[row_idx, 1].set_title("Ground Truth Mask", fontsize=12, fontweight="bold")
            axes[row_idx, 1].axis("off")

            axes[row_idx, 2].imshow(pred_mask, cmap="gray")
            if row_idx == 0:
                axes[row_idx, 2].set_title("Predicted Mask", fontsize=12, fontweight="bold")
            axes[row_idx, 2].axis("off")

            axes[row_idx, 3].imshow(overlay_img)
            if row_idx == 0:
                axes[row_idx, 3].set_title(f"Segmentation Overlay", fontsize=12, fontweight="bold")
            axes[row_idx, 3].set_title(f"IoU: {iou:.3f} | Dice: {dice:.3f}", fontsize=10, color="darkgreen" if iou>=0.6 else "darkred")
            axes[row_idx, 3].axis("off")

        plt.tight_layout()
        composite_path = os.path.join(os.path.dirname(output_dir), "inference_grid_summary.png")
        plt.savefig(composite_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved inference grid summary to: {composite_path}")


def main():
    parser = argparse.ArgumentParser(description="Inference and visual snapshot generator")
    parser.add_argument("--data-dir", type=str, default=r"D:\work_AJfern\work4\psu_reservoir_dataset", help="Dataset directory")
    parser.add_argument("--weights", type=str, default="saved_models/best_model.pt", help="Model checkpoint path")
    parser.add_argument("--output-dir", type=str, default="results/snapshots", help="Output snapshots directory")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of snapshot samples to generate")
    parser.add_argument("--img-size", type=int, default=128, help="Image size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataloader (test set)
    _, test_loader, _, _ = get_dataloaders(args.data_dir, batch_size=8, img_size=(args.img_size, args.img_size), train_ratio=0.65, num_workers=0)

    # Load model
    model = CustomLaneUNet(in_channels=3, num_classes=1, base_c=32).to(device)
    if os.path.exists(args.weights):
        checkpoint = torch.load(args.weights, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded weights from {args.weights}")
    else:
        print(f"Warning: Weights not found at {args.weights}")

    run_inference_and_visualize(model, test_loader, device, output_dir=args.output_dir, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
