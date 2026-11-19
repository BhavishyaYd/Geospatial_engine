
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from csdnet.models.csdnet_base import CSDNetBase
from csdnet.datasets.levirdataset import LEVIRDataset
from csdnet.utils.losses import ChangeDetectionLoss


def calculate_metrics(logits, targets):

    predictions = (torch.sigmoid(logits) > 0.5).float()

    tp = (predictions * targets).sum().item()
    fp = (predictions * (1 - targets)).sum().item()
    fn = ((1 - predictions) * targets).sum().item()

    epsilon = 1e-7

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    iou = tp / (tp + fp + fn + epsilon)

    return tp, fp, fn, precision, recall, f1, iou


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    use_amp=False,
    max_batches=None
):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_dice = 0.0

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    progress = tqdm(loader, leave=False)

    for batch_index, (image_a, image_b, masks) in enumerate(progress):

        if max_batches is not None and batch_index >= max_batches:
            break

        image_a = image_a.to(device, non_blocking=True)
        image_b = image_b.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

            with autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp
            ):

                outputs = model(image_a, image_b)

                detection_loss, bce_loss, dice_loss = criterion(
                    outputs["prediction_logits"],
                    masks
                )

                reconstruction_loss_t1 = F.smooth_l1_loss(
                    outputs["csd_outputs"]["reconstructed_t1"],
                    outputs["features_t1"][3].detach()
                )

                reconstruction_loss_t2 = F.smooth_l1_loss(
                    outputs["csd_outputs"]["reconstructed_t2"],
                    outputs["features_t2"][3].detach()
                )

                reconstruction_loss = (
                    reconstruction_loss_t1 + reconstruction_loss_t2
                ) / 2

                loss = detection_loss + 0.1 * reconstruction_loss

            if training:

                scaler.scale(loss).backward()

                if batch_index == 0:

                    csd_missing_gradients = [
                        name
                        for name, parameter in model.csd.named_parameters()
                        if parameter.requires_grad and parameter.grad is None
                    ]

                    csd_total_parameters = sum(
                        1 for parameter in model.csd.parameters()
                        if parameter.requires_grad
                    )

                    csd_received_gradients = (
                        csd_total_parameters - len(csd_missing_gradients)
                    )

                    print(
                        f"CSD parameters receiving gradients: "
                        f"{csd_received_gradients}/{csd_total_parameters}"
                    )

                    if csd_missing_gradients:
                        print("CSD parameters without gradients:")
                        for name in csd_missing_gradients:
                            print(name)

                scaler.unscale_(optimizer)

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0
                )

                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item()
        total_bce += bce_loss.item()
        total_dice += dice_loss.item()

        tp, fp, fn, _, _, _, _ = calculate_metrics(
            outputs["prediction_logits"].detach(),
            masks
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        progress.set_postfix(loss=f"{loss.item():.4f}")

    batches = (
        min(len(loader), max_batches)
        if max_batches is not None
        else len(loader)
    )

    epsilon = 1e-7

    precision = total_tp / (total_tp + total_fp + epsilon)
    recall = total_tp / (total_tp + total_fn + epsilon)
    f1 = 2 * precision * recall / (precision + recall + epsilon)
    iou = total_tp / (total_tp + total_fp + total_fn + epsilon)

    return {
        "loss": total_loss / batches,
        "bce": total_bce / batches,
        "dice_loss": total_dice / batches,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--smoke-test", action="store_true")

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    use_amp = device.type == "cuda"

    print("Device:", device)
    print("Mixed precision:", use_amp)
    print("Batch size:", args.batch_size)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_dataset = LEVIRDataset(
        root="data/LEVIR-CD",
        split="train",
        patch_size=256,
        augment=True
    )

    val_dataset = LEVIRDataset(
        root="data/LEVIR-CD",
        split="val",
        patch_size=256,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_amp
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_amp
    )

    print("Training patches:", len(train_dataset))
    print("Validation patches:", len(val_dataset))

    model = CSDNetBase().to(device)

    criterion = ChangeDetectionLoss().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-7
    )

    scaler = GradScaler(
        "cuda",
        enabled=use_amp
    )

    checkpoint_dir = Path("checkpoints/csdnet")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch = 0
    best_iou = 0.0

    if args.resume:

        checkpoint = torch.load(
            args.resume,
            map_location=device,
            weights_only=False
        )

        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])

        start_epoch = checkpoint["epoch"] + 1
        best_iou = checkpoint["best_iou"]

        print("Resumed from epoch:", start_epoch)

    if args.smoke_test:

        print("\nRunning one-batch training test...")

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            max_batches=1
        )

        print("\nTraining smoke test:")
        print(train_metrics)

        print("\nRunning one-batch validation test...")

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_amp=use_amp,
            max_batches=1
        )

        print("\nValidation smoke test:")
        print(val_metrics)

        print("\nTraining pipeline smoke test successful.")

        if device.type == "cuda":
            print(
                "Peak GPU memory (MB):",
                round(
                    torch.cuda.max_memory_allocated() / (1024 ** 2),
                    2
                )
            )

        return

    for epoch in range(start_epoch, args.epochs):

        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp
        )

        val_metrics = run_epoch(
            model,
            val_loader,
            criterion,
            device,
            use_amp=use_amp
        )

        scheduler.step(val_metrics["loss"])

        print("\nTraining:", train_metrics)
        print("Validation:", val_metrics)

        print(
            "Learning rate:",
            optimizer.param_groups[0]["lr"]
        )

        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(),
            "best_iou": max(best_iou, val_metrics["iou"])
        }

        torch.save(
            checkpoint,
            checkpoint_dir / "latest.pth"
        )

        if val_metrics["iou"] > best_iou:

            best_iou = val_metrics["iou"]

            checkpoint["best_iou"] = best_iou

            torch.save(
                checkpoint,
                checkpoint_dir / "best.pth"
            )

            print("Best checkpoint saved.")

        if device.type == "cuda":
            print(
                "GPU memory allocated (MB):",
                round(
                    torch.cuda.memory_allocated() / (1024 ** 2),
                    2
                )
            )

    print("\nTraining completed.")
    print("Best validation IoU:", best_iou)


if __name__ == "__main__":
    main()
