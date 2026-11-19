
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.view(
            probabilities.size(0), -1
        )

        targets = targets.view(
            targets.size(0), -1
        )

        intersection = (probabilities * targets).sum(dim=1)

        dice_score = (
            (2.0 * intersection + self.smooth)
            /
            (
                probabilities.sum(dim=1)
                + targets.sum(dim=1)
                + self.smooth
            )
        )

        return 1.0 - dice_score.mean()


class ChangeDetectionLoss(nn.Module):

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        total_loss = (
            self.bce_weight * bce_loss
            + self.dice_weight * dice_loss
        )

        return total_loss, bce_loss, dice_loss


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    criterion = ChangeDetectionLoss().to(device)

    predictions = torch.randn(
        4, 1, 256, 256,
        device=device,
        requires_grad=True
    )

    targets = torch.randint(
        0, 2, (4, 1, 256, 256), device=device
    ).float()

    total, bce, dice = criterion(predictions, targets)

    print("Device:", device)
    print("Total loss:", total.item())
    print("BCE loss:", bce.item())
    print("Dice loss:", dice.item())

    total.backward()

    print("Gradient calculation successful.")
    print("Change detection loss test successful.")
