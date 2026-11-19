
from pathlib import Path

import torch
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader

from csdnet.models.csdnet_base import CSDNetBase
from csdnet.datasets.levirdataset import LEVIRDataset


def denormalize(image):

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image.cpu() * std + mean
    image = image.clamp(0, 1)

    return image.permute(1, 2, 0).numpy()


def calculate_iou(prediction, target):

    prediction = prediction.bool()
    target = target.bool()

    intersection = (prediction & target).sum().item()
    union = (prediction | target).sum().item()

    return intersection / (union + 1e-7)


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    test_dataset = LEVIRDataset(
        root="data/LEVIR-CD",
        split="test",
        patch_size=256,
        augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    model = CSDNetBase().to(device)

    checkpoint = torch.load(
        "checkpoints/csdnet/best.pth",
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    output_dir = Path("outputs/csdnet_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    max_samples = 8

    with torch.no_grad():

        for index, (image_a, image_b, mask) in enumerate(test_loader):

            image_a = image_a.to(device)
            image_b = image_b.to(device)

            outputs = model(image_a, image_b)

            prediction = (
                outputs["prediction_probabilities"] > 0.5
            ).float()

            prediction = prediction[0, 0].cpu()
            target = mask[0, 0]

            # Skip patches without actual changes.
            if target.sum().item() == 0:
                continue

            patch_iou = calculate_iou(prediction, target)

            image_t1 = denormalize(image_a[0])
            image_t2 = denormalize(image_b[0])

            ground_truth = target.numpy()
            predicted_mask = prediction.numpy()

            fig, axes = plt.subplots(1, 4, figsize=(16, 5))

            axes[0].imshow(image_t1)
            axes[0].set_title("Image T1")

            axes[1].imshow(image_t2)
            axes[1].set_title("Image T2")

            axes[2].imshow(ground_truth, cmap="gray", vmin=0, vmax=1)
            axes[2].set_title("Ground Truth")

            axes[3].imshow(predicted_mask, cmap="gray", vmin=0, vmax=1)
            axes[3].set_title(f"Prediction (IoU: {patch_iou:.3f})")

            for axis in axes:
                axis.axis("off")

            plt.suptitle(
                f"CSD-Net Change Detection - Test Patch {index}"
            )

            plt.tight_layout()

            save_path = output_dir / f"sample_{saved_count + 1:02d}.png"

            plt.savefig(
                save_path,
                dpi=150,
                bbox_inches="tight"
            )

            plt.close(fig)

            saved_count += 1

            print(
                f"Saved {saved_count}/{max_samples}: "
                f"{save_path} | IoU: {patch_iou:.4f}"
            )

            if saved_count >= max_samples:
                break

    print("\nVisualization completed.")
    print("Total images saved:", saved_count)
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
