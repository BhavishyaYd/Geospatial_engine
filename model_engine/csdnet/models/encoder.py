
import torch
import torch.nn as nn
from torchvision.models import resnet18

from csdnet.models.perturbation import PerturbationModule


class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        batch, channels, _, _ = x.shape

        weights = self.pool(x).view(batch, channels)
        weights = self.fc(weights).view(
            batch, channels, 1, 1
        )

        return x * weights


class FeatureEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        backbone = resnet18(weights=None)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layers = nn.ModuleList([
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        ])

        self.se_blocks = nn.ModuleList([
            SEBlock(64),
            SEBlock(128),
            SEBlock(256),
            SEBlock(512)
        ])

       

    def stem(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        return x

    def stage(self, x, index):

        x = self.layers[index](x)
        x = self.se_blocks[index](x)

        return x


class SiameseEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.encoder = FeatureEncoder()

        self.perturbation = nn.ModuleList([
            PerturbationModule(64),
            PerturbationModule(128),
            PerturbationModule(256),
            PerturbationModule(512)
        ])

    def forward(self, image_t1, image_t2):

        x1 = self.encoder.stem(image_t1)
        x2 = self.encoder.stem(image_t2)

        features_t1 = []
        features_t2 = []
        differences = []
        attentions = []

        for i in range(4):

            x1 = self.encoder.stage(x1, i)
            x2 = self.encoder.stage(x2, i)

            x1, x2, difference, attention = (
                self.perturbation[i](x1, x2)
            )

            features_t1.append(x1)
            features_t2.append(x2)
            differences.append(difference)
            attentions.append(attention)

        return (
            features_t1,
            features_t2,
            differences,
            attentions
        )


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SiameseEncoder().to(device)
    model.eval()

    image_t1 = torch.randn(
        1, 3, 256, 256, device=device
    )

    image_t2 = torch.randn(
        1, 3, 256, 256, device=device
    )

    with torch.inference_mode():

        features_t1, features_t2, differences, attentions = (
            model(image_t1, image_t2)
        )

    print("Device:", device)

    for i in range(4):

        print(
            f"Stage {i + 1}: "
            f"T1 {tuple(features_t1[i].shape)} | "
            f"T2 {tuple(features_t2[i].shape)} | "
            f"D {tuple(differences[i].shape)} | "
            f"A {tuple(attentions[i].shape)}"
        )

    print("Interleaved Siamese encoder test successful.")
