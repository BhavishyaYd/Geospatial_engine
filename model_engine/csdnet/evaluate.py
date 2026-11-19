
import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from csdnet.models.csdnet_base import CSDNetBase
from csdnet.datasets.levirdataset import LEVIRDataset


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = device.type == "cuda"

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    test_dataset = LEVIRDataset(
        root="data/LEVIR-CD",
        split="test",
        patch_size=256,
        augment=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp
    )

    print("Test patches:", len(test_dataset))

    model = CSDNetBase().to(device)

    checkpoint = torch.load(
        "checkpoints/csdnet/best.pth",
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():

        for batch_index, (image_a, image_b, masks) in enumerate(test_loader):

            image_a = image_a.to(device, non_blocking=True)
            image_b = image_b.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):

                outputs = model(image_a, image_b)

            predictions = (
                outputs["prediction_probabilities"] > 0.5
            ).float()

            total_tp += (predictions * masks).sum().item()
            total_fp += (predictions * (1 - masks)).sum().item()
            total_fn += ((1 - predictions) * masks).sum().item()

            if (batch_index + 1) % 100 == 0:
                print(
                    f"Processed {batch_index + 1}/{len(test_loader)} batches"
                )

    epsilon = 1e-7

    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    iou = total_tp / (total_tp + total_fp + total_fn + epsilon)

    print("\nCSD-Net Test Results")
    print("--------------------")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"IoU:       {iou:.4f}")

    print("\nTest evaluation completed.")


if __name__ == "__main__":
    main()
